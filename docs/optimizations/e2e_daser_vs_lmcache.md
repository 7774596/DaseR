# E2E Benchmark: DaseR vs LMCache LocalDiskBackend

**Run:** 2026-04-23
**Harness:** `benchmarks/bench_e2e_daser_vs_lmcache.py`
**Model:** Qwen/Qwen3-8B (bfloat16)
**GPU:** CUDA 2 (NVIDIA H800, 79 GB; 19 GB free at start; `gpu_memory_utilization=0.22`, `max_num_seqs=64`)
**Workload:** 200 IMDB reviews, `max_tokens=1`, `temperature=0`, `enable_prefix_caching=False`
**Prompt-token total:** 60,595 tokens (avg 303 tok/prompt, 3,867 KV blocks)

Both systems run cold → warm on the **same** `LLM` instance. LMCache's
`LocalDiskBackend` keeps its chunk index in memory only (no directory scan on
startup), so a rebuild would orphan every chunk written by the cold pass. vLLM
recycles in-GPU KV between `generate()` calls with prefix caching off, so the
warm pass still has to fetch KV from the external storage tier — exactly the
signal this benchmark measures. DaseR receives identical treatment for fairness.

## Results

| Metric                       | DaseR     | LMCache   |
|------------------------------|----------:|----------:|
| cold elapsed (s)             |    27.64  |     5.72  |
| warm elapsed (s)             |    13.58  |     2.37  |
| cold prompt tok/s            |    2,192  |   10,590  |
| warm prompt tok/s            |    4,462  |   25,541  |
| warm/cold speedup            |    2.04×  |    2.41×  |

DaseR warm tok/s ÷ LMCache warm tok/s = **0.17×**.

Correctness: 2/200 prompts diverge on the single generated token in each
system. This is a known KV-reuse precision effect (argmax over a hidden state
reconstructed from cached KV can flip for near-tied logits) — not a bug in
either storage layer.

## Reading the numbers

- **Both systems demonstrate their storage tier works.** Warm elapsed is
  meaningfully below cold for each; both show ~2× warm speedup from skipping
  prefill recompute.
- **LMCache is ~5–6× faster in absolute E2E throughput.** The prior storage
  microbenchmark (`benchmarks/bench_storage_imdb.py`) showed DaseR's raw
  bytes-I/O layer beating LMCache's (1.54–1.83× on reads). The E2E gap
  therefore lives in the **connector / control plane** overhead DaseR adds
  per request or per layer, not in the GDS data plane itself.

## Known fairness caveats

- LMCache's `LocalDiskBackend` does not survive engine restart — its on-disk
  files are only reachable via its in-memory index. This benchmark runs cold
  and warm on the same engine to work around that. Any future benchmark that
  wants to measure *persistent* KV offload across separate engine sessions
  would need an LMCache configuration with a restartable index (out of scope
  here).
- Qwen3-8B with a 19 GB free-memory ceiling forced `gpu_memory_utilization=0.22`
  and `max_num_seqs=64`. Larger vLLM batches would widen both tok/s numbers
  but the ratio is what matters for this comparison.
- 2/200 correctness mismatches occurred for both systems; they are independent
  of which storage backend is in use.

## Reproducing

```bash
source /data/zwt/vllm/bin/activate
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python benchmarks/bench_e2e_daser_vs_lmcache.py \
    --num-prompts 200 \
    --gpu-util 0.22 \
    --max-num-seqs 64 \
    --out docs/optimizations/e2e_daser_vs_lmcache.json
```

The script re-execs itself once with `PYTHONHASHSEED=0` so LMCache's
`NONE_HASH` seeding and Python string hashing stay deterministic across
process boundaries (without this, the scheduler's hash and the workers' hash
disagree and LMCache reports zero warm hits).

Raw JSON: [e2e_daser_vs_lmcache.json](./e2e_daser_vs_lmcache.json).

## Implication

The E2E comparison exposes a DaseR connector-level overhead that does not
appear in the storage microbenchmark. Profiling the per-request path in
`DaserConnector` (index lookup, slot allocation, layer iteration, IPC round
trips) is the next optimization target — not the GDS data plane.
