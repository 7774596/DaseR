# Next-session prefetch

## Problem

Agent workloads interleave several long-lived sessions. Each session sends a
request, waits for a tool call, and returns with a prompt that extends its
previous one. When the working set overflows the GPU, the returning session's
prefix is in L1 or on the SSD. The lookup-triggered readahead
(`DASER_LOAD_READAHEAD`) only starts reading after vLLM schedules the request,
so the SSD read sits on the request's TTFT.

The return order is predictable from the lookup stream alone. The session
idle the longest is usually the next to come back. The server can read that
session's stored prefix while the current request computes.

## Mechanism

All of it is server-side research behavior behind environment switches. Every
switch is off by default, and with all of them off the load path is unchanged.

- **Session predictor** (`daser/server/session_predictor.py`, enabled with
  `DASER_SESSION_PREFETCH=1`).
  - Matching: a lookup whose prompt repeats the leading `match_fraction` of an
    earlier prompt continues that session. Any other lookup opens a new
    session.
  - Prediction: `predict(k)` returns the `k` sessions idle the longest,
    soonest first. Sessions idle for more than `stale_factor` average return
    gaps are presumed finished and left out.
  - Inputs: both `lookup` and `lookup_prefetch` feed the predictor, so it
    works with scheduler lease prefetch on or off.
- **Speculative readahead** (`IPCServer._session_prefetch_readahead`).
  - When it runs: after each load, the server resolves the predicted
    sessions' stored prefixes to physical spans.
  - Dropping stale reads: it first calls
    `transfer.drop_speculative_except` to drop speculative readaheads of
    sessions that are no longer predicted.
  - Read order: for each predicted session, soonest first, it waits for the
    in-flight readahead to go idle and then calls
    `start_readahead(spans, speculative=True)`. The current request's own
    reads always win the SSD.
- **Depth and budget.** `DASER_SESSION_PREFETCH_DEPTH` sets how many sessions
  are read ahead (default 1). Readahead leases may hold at most
  `DASER_READAHEAD_BUDGET_FRACTION` of L1 (default 0.5). A speculative
  readahead that does not fit the remaining budget is skipped, never forced.
  Fixed depth plus the budget check therefore acts as a byte-budgeted greedy
  depth.
- **Device stash** (`DASER_DEVICE_STASH_BYTES`, `DASER_DEVICE_STASH_SESSIONS`).
  - What it holds: an optional server-owned GPU buffer for the stored bytes
    of the soonest `DASER_DEVICE_STASH_SESSIONS` predicted sessions.
  - How it fills: the buffer is filled from L1 after those sessions'
    readaheads finish.
  - How loads use it: loads whose spans the stash holds are served by
    device-to-device copy instead of host-to-device copy.
  - Skip option: `DASER_READAHEAD_SKIP_STASHED=1` makes readaheads skip spans
    the stash already holds.

## Interaction with lossless compression

Stored bytes are codec output, so compression shrinks every quantity the
mechanism budgets: SSD bytes per session, L1 lease bytes, and stash bytes. A
fixed readahead budget therefore covers about 1/ratio more sessions. This
matters most for depth above 1: with raw bytes, the second predicted session
often no longer fits the budget and its readahead is skipped.

## Measured effect

Research runs used agent-session traces with 256k-token contexts, the working
set spilling to L1/L2, one request at a time, and paired warm TTFT.

| Setting (budget 0.75) | Compressed: depth 2 / no session prefetch | Raw: depth 2 / no session prefetch |
|---|---|---|
| Dense 4B | 0.75 | 0.87 |
| MoE 30B-A3B FP8 | 0.81 | 0.90 |

On the dense model the raw baseline is its strongest configuration (depth 1,
budget 0.5); raw depth 2 equals raw depth 1 there.

- **Why depth 2 gains under compression.** The second-session readaheads
  mostly start under compression and are mostly skipped under raw bytes.
- **Device stash.** It adds a small gain of about 0.94–0.97× that does not
  depend on compression.
- **Stash variants that did not help.** Skipping stashed spans gave no gain.
  Splitting HBM between KV and the stash gave no gain. A two-session stash was
  slower: its fill overlaps the next load.
- **Layer-wise loading.** It cannot hide these loads, because each layer's
  copy of a 256k-token prefix outlasts that layer's suffix compute.

## Limitations

- **Session order is a heuristic.** Prediction relies on prompt-prefix
  continuation and round-robin return order. Workloads where sessions return
  in an order unrelated to idle time get no benefit, but the speculative reads
  still compete for the budget.
- **Stash fills cost PCIe bandwidth.** A stash fill that overlaps the next
  request's load competes with it for PCIe bandwidth.
