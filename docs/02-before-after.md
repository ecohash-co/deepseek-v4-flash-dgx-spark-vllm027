# Before / after: vendor-overlay 0.21 vs upstream 0.27.1

Both stacks serve the **same checkpoint** (`deepseek-ai/DeepSeek-V4-Flash-0731`) on the **same two
machines** with the **same** `--tensor-parallel-size 2 --max-model-len 1048576 --max-num-seqs 12
--max-num-batched-tokens 8192 --gpu-memory-utilization 0.80` and the same DSpark spec-decode
config (`k=5`, probabilistic draft sampling).

## Configuration delta

| | **Before** (0.21 + overlay) | **After** (upstream 0.27.1) |
|---|---|---|
| Base image | `vllm-dspark-runtime:dspark-nvfp4-stage-c` (vendor overlay, locally built, unreproducible if lost) | `vllm/vllm-openai:v0.27.1` (official) + 1 rebuilt component |
| vLLM | `0.21.1rc1.dev339+g1967a5627bc3` | `0.27.1` |
| FlashInfer | 0.6.12 | 0.6.16.post3 |
| KV cache dtype | `nvfp4_ds_mla` (overlay-only) | `fp8_ds_mla` (**upstream has no nvfp4 variant**) |
| MoE backend | B12X (overlay did an out-of-tree NVFP4 expert conversion) | `DEEPGEMM_MXFP4` |
| Patches carried | 4 vendor patches + bind-mounts | 1 functional + 1 diagnostic, **baked into the image** |
| Python paths | `/opt/env/...` (micromamba) | `/usr/local/...` (system) |

## Measured

| Metric | Before | After | Δ |
|---|---|---|---|
| **Prefill, needle @ 512K** | 276.3 s | **233.8 s** | **−15.4%** ✅ |
| **KV cache tokens** | 2,034,816 – 2,122,679 | **1,266,988** | **−38%** ❌ |
| KV memory available | 13.86 – 14.19 GiB | 13.47 GiB | −2.8% |
| **Per-token KV footprint** | ~7.3 KB | ~11.4 KB | **+56%** ❌ |
| Max concurrency @ 1M ctx | 1.94 – 2.02× | 1.21× | ❌ |
| Correctness @ 377,594 tok | pass | **pass (3/3 verbatim)** | = |
| Boot to health | ~13 min | **~6 min** | ✅ |
| Asserts @ 12 concurrency | — | **0** | ✅ |

### The KV regression is the real cost of going upstream

At essentially the same GPU memory (−2.8%), the token pool fell 38%. That is a **~1.56× larger
per-token KV footprint**, and it is almost certainly the KV dtype: upstream has no
`nvfp4_ds_mla`, so we run `fp8_ds_mla`.

`fp8_ds_mla` on DSV4 is **584 B/token/layer** (448 NoPE + 128 RoPE + 8 fp8 scale). If the overlay's
`nvfp4_ds_mla` stored the NoPE component at 4 bits instead of 8, that is 224 + 128 + scales ≈
360 B — a ratio of ~1.62, which lines up with the measured 1.56. **We have not read the overlay's
kernel, so treat the mechanism as a strong hypothesis and the measurement as fact.**

Practical impact: at 1M context you get **1.21 concurrent requests instead of ~2**. If you serve
long contexts to more than one caller at a time, this is the thing that will bite you, and it is
the strongest argument for staying on the overlay until upstream lands an NVFP4 MLA KV dtype.

### Do not transplant the overlay's "B12X is 2× faster" advice

The overlay reached the B12X MoE backend only because it performed an **out-of-tree NVFP4 expert
conversion**. This checkpoint's experts are natively **MXFP4**, and upstream's
`FlashInferB12xExperts` asserts `quant_dtype == "nvfp4"`, so `--kernel-config
'{"moe_backend":"flashinfer_b12x"}'` **fails at boot** on the stock image. `DEEPGEMM_MXFP4` is the
correct in-tree choice here. Reaching B12X upstream would require an offline NVFP4 requant of the
experts — a project, not a flag.

## Decode: measured, attributed, and *not* directly comparable

| depth | decode tok/s | accepted/draft | accept % | per-position accept |
|---|---|---|---|---|
| 8K | 52.0 | 3.17 | 63.3% | p0 88 · p1 71 · p2 60 · p3 55 · p4 43 |
| 128K | 37.0 | 2.11 | 42.2% | p0 80 · p1 53 · p2 36 · p3 25 · p4 17 |
| 512K | 42.3 | 2.83 | 56.5% | p0 85 · p1 73 · p2 56 · p3 38 · p4 31 |

**Throughput tracks speculative acceptance.** `p0` stays at 80%+ at every depth, which is the
signature of ordinary model uncertainty rather than a broken draft path — a genuine draft bug
shows up as `p0` collapse, not a gentle positional decay.

**We deliberately do not publish a before/after decode number.** The two stacks were measured with
different instruments and different sampling, and run-to-run acceptance variance on this model is
large enough to swamp the comparison: we measured the *same* 128K config at **18.3 ±2.2 tok/s** in
one session and **37.0 tok/s** in another. Publishing a delta from that would be noise dressed as a
result. See [04-measurement.md](04-measurement.md).

## Saturation (upstream 0.27.1)

| concurrency | wall | total out | aggregate | median TTFT |
|---|---|---|---|---|
| 1 | 10.7 s | 300 | 27.94 tok/s | 10.73 s |
| 4 | 32.1 s | 952 | 29.67 tok/s | 25.01 s |
| 8 | 21.8 s | 1908 | 87.43 tok/s | 17.13 s |
| 12 | 30.9 s | 2865 | **92.61 tok/s** | 24.97 s |

These **include prefill**, so they are a stability/saturation result, not a decode benchmark.

Concurrency 12 matters for a specific reason: at 12 sequences the speculative verify pass is
12 × (1+5) = **72 tokens**, which exceeds the 64-token threshold, so attention routes to the
**prefill orchestrator** — the *opposite* side of the dispatch gate patched in
[01-blockers.md](01-blockers.md). It needed its own proof that it does not assert. It does not.

## Verdict

Going upstream is worth it **if** your workload is long-context and mostly serial, or you value
a reproducible base image over peak concurrency. It is **not** yet worth it if you need 2×
concurrent 1M-context streams — the KV regression is real and unfixed. See
[07-roadmap.md](07-roadmap.md).
