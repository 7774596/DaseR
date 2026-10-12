# Explicit batch-invariant cache correctness benchmarks

## Problem and scope

On the original `fix/transfer-connector-bundle` revision
`dc536011c6ca6875125237fd0e41a73c33ac0bf5`, greedy sampling with temperature zero
and seed 42 does not guarantee identical generated text between cold and warm
requests. Cache reuse changes the amount of uncached computation and can alter
batch shapes. CUDA graph policy also affects the observed outputs. This can
confound an exact-text check intended to detect cache corruption.

This change adds an explicit diagnostic option to the original benchmark tools.
It does not change the connector, transfer implementation, default graph policy
or vLLM source, and does not include any layerwise implementation or fixes.
The first numerically divergent engine operator has not been isolated.

## Change

Both `benchmarks/run_bench.py` and `benchmarks/bench_start_servers.py` accept
`--batch-invariant`. The startup manager sets vLLM's existing
`VLLM_BATCH_INVARIANT=1` in the engine subprocess environment before imports and
kernel selection, for every selected benchmark backend. Explicit opt-in wins
over an inherited zero, and an inherited one remains supported. The parent
process environment is not mutated.

The effective mode is recorded in `manifest.json` and the manifest embedded in
`results.json` as `batch_invariant`. The flag is off by default. Performance
comparisons should use the same mode on every arm, and should not substitute
these diagnostic timings for default-mode performance measurements.

Example (replace model, dataset and approved scratch locations):

```bash
python benchmarks/run_bench.py \
  --backend daser-prefix,daser-chunk \
  --batch-invariant \
  --dataset longbench \
  --longbench-dir /path/to/longbench/data \
  --datasets qasper,hotpotqa_e \
  --model /path/to/Qwen3-8B \
  --store-dir /path/to/approved-scratch \
  --gpu-id 0 --gpu-util 0.85 \
  --max-num-seqs 4 --max-num-batched-tokens 512 \
  --max-model-len 8192 --max-context-tokens 7168 \
  --max-inflight 2 --max-samples 3 --gen-max-tokens 128
```

For a memory-limited GPU, use a graph capture ceiling appropriate to its KV
capacity. This verification uses a ceiling of eight; the option above does not
change graph capture settings or force PIECEWISE.

## Original-source diagnostic controls (2026-10-12)

Verified with vLLM 0.30.0, Qwen3-8B BF16, RTX 4090, raw L1-only KV, concurrency
two and six samples (three each from qasper/hotpotqa_e). Prompt context is
capped at 7168, maximum model length is 8192, maximum sequences four and
maximum batched tokens 512. Services and stores are fresh for each arm.
The original default graph policy is FULL_AND_PIECEWISE.

| Original-source configuration | Output limit | Prefix cold/warm exact matches |
|---|---:|---:|
| Default graph, normal numerics | 64 | 4/6 |
| Default graph, normal numerics | 128 | 5/6 |
| PIECEWISE, normal numerics | 16 | 5/6 |
| PIECEWISE, normal numerics | 64 | 3/6 |
| PIECEWISE, normal numerics | 128 | 4/6 |
| Default graph, batch-invariant | 64 | 6/6 |
| Default graph, batch-invariant | 128 | 6/6 |

The original PIECEWISE warm outputs match all six corresponding layerwise
warm outputs at limits 64 and 128 in the earlier investigation. Waiting for
all layers before computation does not remove the default numerical variation.
These observations motivate controlling engine numerics for cache correctness
rather than treating every text mismatch as proof of an asynchronous copy bug.

Batch-invariant mode is a diagnostic control, not a blanket correctness
promise: incorrect cached tensors or stochastic sampling can still produce
different outputs. Chunk mode has a document-upload cold phase, so it has no
cold/warm generated-text comparison; its outputs require a separate reference.

## New CLI end-to-end verification

Ran the preceding six-sample workload through the new `--batch-invariant`
option with the parent `VLLM_BATCH_INVARIANT` variable unset. No external hook
sets determinism or changes connector configuration; the only external launch
adaptation caps graph capture at eight to fit the 4090. Graph policy remains
the original FULL_AND_PIECEWISE.

- Prefix: six cold and six warm requests succeed; cold/warm text matches 6/6.
  Warm output totals 589 tokens and warm-phase wall time is 14.634 seconds.
- Chunk: six warm inference requests succeed; outputs match the previous
  original-source batch-invariant reference 6/6. Warm output totals 605 tokens
  and warm-phase wall time is 14.963 seconds.
- Both manifests and their embedded result copies record
  `batch_invariant: true`. No layerwise settings or source changes are present.

For context, the previous original normal-numerics runs took 6.343/6.708 seconds
for prefix/chunk and generated 573/601 tokens. The output counts differ, so this
is not a precise equal-token slowdown ratio. It still demonstrates why the
control is opt-in and why its measurements must be labeled separately.
Stores/sockets and owned services are cleaned; result JSON and logs remain as
local evidence. Machine-specific resource paths are not committed.

Environment isolation regression cases cover an unset variable, inherited
zero, inherited one, and explicit opt-in overriding zero. They start the public
engine service API with mocked process/HTTP boundaries, verify the actual
engine environment, persist/read the manifest, and confirm the parent
environment remains unchanged.

The submission check completes with **584 passed, 1 skipped, 136 deselected**
using only the 4090 and `-m "not integration and not slow"`, with the previously
stalling IPC-client module excluded. The skipped test requires two visible
CUDA devices. Ruff checks/formatting and whitespace validation also pass.
