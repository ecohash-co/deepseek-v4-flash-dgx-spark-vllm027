# Measurement methodology — including the instruments that lied

Every harness in `bench/` is here, including the ones that produced wrong answers, because knowing
*which* measurement to distrust was most of the work.

---

## Instruments that produced wrong answers

### `benchmarks/longctx.py` — invalid on 0.27.1

It derives decode throughput as `warm_N − warm_1`, assuming the warm (prefix-cached) prefill is
stable. On 0.27.1 the warm prefill for an **identical** call ranged **0.38 s … 17.18 s**.
Subtracting two noisy numbers of similar magnitude produced artifacts up to **1.5×10⁸ tok/s** and
spurious "COLLAPSE" flags.

We reported a decode regression from this before discarding it. **Do not use differencing
harnesses on a stack whose prefill timing you have not first shown to be stable.**

**Use instead:** `bench/decode_direct.py` — stream the response, divide emitted tokens by wall
time *after the first token*. No subtraction, no assumptions.

### Any single-run decode number

The same 128K configuration measured **18.3 ±2.2 tok/s** in one session and **37.0 tok/s** in
another. Speculative acceptance varies enough run-to-run to swamp most effects you would want to
measure.

**Use instead:** `bench/accept_by_depth.py`, which scrapes vLLM's `spec_decode_*` counters
immediately before and after a **single** request, so the delta is attributable to that request
alone, and reports throughput and acceptance from the *same* call. If both move together, the
model got less certain; nothing in the stack changed.

Read `p0` (first draft position) specifically: a genuine draft-path bug shows as **`p0` collapse**.
Gentle positional decay (p0 88% → p4 43%) is ordinary uncertainty.

### Failure rates measured under unknown load

Our JSON-degeneracy rate for the identical cell was **13/20 serial** and **20/20 serial-on-an-idle-box**
and **0/10 under concurrency 4**. Ambient load changes the result. An early write-up compared a
loaded measurement against an idle one and briefly concluded the new version had made things
twice as bad.

**Always state the concurrency at which a rate was measured.** See
[06-the-json-non-bug.md](06-the-json-non-bug.md).

### The profiler's own decoy

`py-spy` showed **88.66%** of samples in `sched_yield` at `vllm/distributed/utils.py:48`. That is
not a bottleneck — it is vLLM's *intentional* tight polling loop (~3×10⁻⁷ s/iteration, documented
in a comment on the line above it), i.e. the "waiting for the peer node" bucket. A spin-wait
dominates any sampling profile by construction.

Read the *shape* of a profile, not its top line.

---

## Harnesses

| Script | Answers |
|---|---|
| `needle.py` | Is long-context retrieval **correct**? Verbatim needle at 8K/128K/512K. |
| `decode_direct.py` | What is decode throughput, without differencing? |
| `accept_by_depth.py` | Is a throughput change explained by speculative acceptance? |
| `conc12.py` | Does it survive saturation, including the >64-token verify pass? |
| `json_rate_2x2.py` | Degeneracy rate across thinking × sampling (2×2). |
| `serial_json.py` | Serial vs concurrent — is a failure batch-dependent? |
| `batch_threshold.py` | Where is the batch threshold, and is it prompt-specific? |
| `logprob_tie.py` | Is the first token contested? (top-2 logprob gap) |
| `cachebust.py` | Is prefix caching involved, or does it survive cold prefills? |
| `profagg.py` | Aggregate `py-spy` raw output by frame. |

All take `URL` / `MODEL` / `DEPTHS` / `N` from the environment and hit the OpenAI-compatible
endpoint directly. Point them at your own server; nothing is site-specific except the defaults.

---

## Correctness before performance

`needle.py` embeds a distinctive fact at a controlled depth and requires it back **verbatim**,
including a numeric constant, a proper name, and a label. It is the gate everything else depends
on: a stack that is fast and subtly wrong at 300K tokens is worse than one that is slow.

Verified: **3/3 verbatim at 377,594 prompt tokens.**

Do this *before* you benchmark, and again after any attention-path change. Our patch touches
attention index widths — exactly the sort of change that could plausibly corrupt long-context
retrieval while leaving short prompts perfect.

---

## Statistics, briefly

For rate comparisons (does patch X change the failure rate?) we use **Fisher's exact test** on the
2×2 and report the p-value. Small-n LLM measurements produce differences that look dramatic and
are not: 4/20 vs 0/20 is p=0.106 — **not** significant, despite looking like a clean win.

For paired before/after latency across depths, a **sign test** across depths is more honest than a
t-test on pooled runs, because the depths are not exchangeable.

Our one statistically clean result — the MiaAI Issue #22 one-line change — was **6/6 depths
improved, p=0.016**, with decode neutral and warm prefill **−16.1%**.
