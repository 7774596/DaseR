# FragPack decode scheduling under skewed concurrency

**Date:** 2026-10-25

**Baseline:** `research/lossless-kv-e2e@d7b2d052`

**Benchmark:** `benchmarks/research/fragpack_step_bench.py`

## Problem and implementation

The previous decode budget was `max(batch, 2 * SMs // KV_heads)`. On the
measured H800 with four KV heads, one wave is 66 units per head. With 65
short requests and one long request, each request consumes one unit: the
long request cannot split, even when most CTAs finish much earlier.
Zero-length CUDA Graph padding also consumed units.

The new work-unit capacity is `batch + wave - 1`. The device planner
subtracts padding rows and their reservations together, so padding neither
consumes work units nor changes the live batch's split budget. Empty rows
produce zero outputs, including after a graph replay with nonempty rows.

Additional capacity is used only when the original budget cannot bound
chunk length to `2 * ceil(total_pages / original_budget)`. Otherwise the
original split count is retained. This threshold is a heuristic to avoid
excessive splits in ordinary batches, not a globally optimal scheduler.
The plan stores the chosen budget. The static CTA grid stays
`(max(batch, wave), KV_heads)`; each CTA processes subsequent units of its
own head at a grid-sized stride. Each CTA handles at most two units. Partial
results use the work-unit capacity as their stride. Shared-memory reuse is
ordered by CTA barriers, and completed TMA barriers are invalidated before
the next unit initializes them, as required by the
[PTX mbarrier lifecycle](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-init).
All layers reuse the first layer's plan without a host synchronization.

For this four-head geometry, the maximum extra float workspace is about
1.03 MiB, plus roughly 1 KiB of plan storage. It is shared across layers.
No encoded record format, KV values, or compression policy changes.

## Benchmark repairs

- Recreate valid open-block staging before each write/encode sample through
  the public `prepare`, `write`, and `commit` API. The previous benchmark
  filled and committed all blocks, then wrote at offset 5 without reclaiming
  staging; that could measure a no-op. Measure both a middle token and the
  token that closes a block.
- Check encoded-record counter deltas and bit-exact gathered KV for every
  sample. Alternate the appended token to detect stale writes. Reclaim
  blocks between samples so repeated close encodes cannot exhaust overflow
  storage. The raw-record test exercises that recycling as well.
- Distinguish eager event timings, which include host launch gaps, from
  `--cuda-graph` replay timings. Stateful write/encode graphs contain exactly
  one step, with external CUDA event nodes and untimed setup/commit/checks.
- Poison and validate each decode configuration's actual eager/graph
  output before timing, including every `--units` value and replanning.
  Validate prefill gather bit-for-bit and its attention output against FA3.
- Prewarm each repeatable decode/prefill arm for at least `--warmup-ms`
  CUDA-event milliseconds after capture and validation (default 100 ms).
  Stateful write/encode instead uses three independently prepared steps.
  Retain every sample, median, mean, standard deviation, and p10/p90.
- Label `us_plan_delta_estimate` as a difference of medians, not an isolated
  planner kernel measurement. Planning is once per engine step. Prefill's
  component-sum difference is also an estimate, not an end-to-end latency.

## Measurement method

Both revisions run the same corrected benchmark; only the imported
production implementation changes. Run order is old/new/new/old. Each run
uses 15 rounds, 10 calls per captured decode sample, and at least 100 ms of
prewarming per arm. Report the mean of the two run medians for each revision.
GPU clock/power/temperature samples are retained; no clocks are locked.

- Idle NVIDIA H800, 132 SMs; the H800 PCIe device is excluded.
- CUDA 12.9, PyTorch 2.13.0+cu129, vLLM 0.30.0, BF16, four KV heads,
  GQA 8, head dimension 128, block size 128.
- `moe_agent.npy`, uint16 BF16 bits, shape `[32, 48, 2, 128, 4, 128]`;
  SHA-256 `1302f31ac929ea3f623db0edcae7fa5869e50449c4b572dcff4ac17e7ae39322`.
  Layer 24, first eight pages for calibration, 24 held-out pages repeated
  into 8192 cache blocks. Queries and block placement use seed 0.
- This is captured KV with synthetic query/length workloads, not a native
  FP8 serving trace. Timings cover one layer's fused decode and combine,
  excluding the once-per-step planner. They do not establish TTFT, TPOT,
  full-model throughput, transfer performance, or total memory savings.

## Results and remaining regression

These are microseconds per layer, excluding planning. A skewed batch has
`B - 1` requests at 1024 tokens and one at 131072 tokens.

| Workload | Old us | New us | Old / new |
| --- | ---: | ---: | ---: |
| One request, 1024 tokens | 9.1 | 9.4 | 0.96x |
| One request, 16000 tokens | 20.3 | 20.5 | 0.99x |
| One request, 64000 tokens | 56.6 | 57.5 | 0.98x |
| Ragged eight-request batch | 480.1 | 470.9 | 1.02x |
| Skewed B=64 | 1147.0 | 170.0 | 6.75x |
| Skewed B=66 | 3356.2 | 183.6 | 18.28x |
| Skewed B=96 | 2377.0 | 230.2 | 10.33x |
| Skewed B=132 | 2393.3 | 266.3 | 8.99x |
| Skewed B=256 | 2492.9 | 351.1 | 7.10x |
| Uniform B=64, 8192 tokens each | 416.8 | 501.4 | 0.83x |
| Uniform B=65, 8192 tokens each | 472.4 | 493.9 | 0.96x |
| Uniform B=66, 8192 tokens each | 596.8 | 502.2 | 1.19x |
| Uniform B=132, 4096 tokens each | 433.6 | 523.2 | 0.83x |
| Skewed B=66 padded to 96 | 2355.8 | 214.1 | 11.00x |
| One 64000-token request padded to 96 | 1087.1 | 60.8 | 17.89x |

The severe skew cliff is reduced, but this is **not a regression-free
replacement**. Uniform B=64 and B=132 are respectively about 20.3% and
20.7% slower in the final full matrix. Uniform B=65 is 4.5% slower; the
single-1024 case adds about 0.33 us. Keep this as an optimization branch
pending resolution of those regressions before merging or enabling it in
general serving workloads.

Earlier exploratory matrices and isolated unit-count sweeps show substantial
variation in ordinary workloads. Telemetry also records changing SM clocks
and power draw; the current evidence does not isolate compiler, memory
layout, scheduling, and DVFS effects. Equal-duration warmup improves the
method but does not make those effects disappear. The final table retains
the complete workload matrix, including slower cases; it does not select
the best exploratory measurements.

## Reproduction

Select an idle non-PCIe H800 by UUID. Use an external scratch directory for
CUDA builds and output. The high-concurrency cases can be generated with:

```bash
cases="$(python -c 'print(",".join("/".join(["1024"] * (b - 1) + ["131072"]) for b in (64, 66, 96, 132, 256)))')"
python -m benchmarks.research.fragpack_step_bench \
  --dump KV_DUMP.npy --layer 24 --heads 0:4 --blocks 8192 \
  --rounds 15 --warmup-ms 100 --batch 64 \
  --decode "$cases" --prefill '' --cuda-graph --out RESULT.json
```

The full matrix additionally checks one request at 1024/16000/64000 tokens;
an eight-request batch at 4000/8000/16000/32000/64000/100000/128000/160000;
64/65/66 requests at 8192 tokens; 132 requests at 4096 tokens; and padding to
96 rows for the 66-request skew case and for a single 64000-token request.
Separate raw-record graph and eager runs use five rounds, a two-request
1024/8192 batch, unit counts 2/66/132, and an 8192-token prefill with a
128-token chunk.

## Validation

The scheduling regression tests use a separate FP32 attention reference,
two layers, batches around the wave boundary and above 256 rows, and one
graph replayed with changed lengths, padding, an entirely empty batch, and
sparse live rows. They cover both automatic capacity and the minimum legal
capacity (`units == batch`). Existing HBM store integration tests also run.

Benchmark fault injection intentionally omitting the measured write or
close encode must fail its bit-exact or counter check, respectively.

Completed checks: 1064 CPU tests passed (five skipped), all 12 targeted GPU
tests passed, the complete six-run A/B/raw/eager matrix passed its output
checks, and both omitted-operation injections were rejected. Repository
Ruff 0.15.10 lint/format checks and `git diff --check` passed.

Compute Sanitizer `synccheck` passed both graph cases with zero errors.
Default `memcheck` reported `CUDA_ERROR_INVALID_HANDLE` in
`cuKernelGetFunction` during HBM prepare on both the unchanged baseline and
this branch, although the tests completed. With CUDA API error reporting
disabled, device memory checking passed both graph cases with zero errors.
This is a qualified device-access check, not a claim that the default
sanitizer run was clean; both original diagnostics are retained.
