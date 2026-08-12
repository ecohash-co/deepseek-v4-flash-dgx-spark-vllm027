# DeepSeek-V4-Flash on 2× DGX Spark (GB10) — running it on **upstream vLLM 0.27.1**

Getting `deepseek-ai/DeepSeek-V4-Flash-0731` with **DSpark speculative decoding** serving on a
pair of NVIDIA DGX Sparks (GB10, `sm_121`, compute capability 12.1) using the **stock upstream
`vllm/vllm-openai:v0.27.1` image** instead of a vendor overlay.

Out of the box that does not work. Three stacked defects have to be cleared first, and the last
one — a dispatch-gate mismatch between vLLM and FlashInfer — makes **every DSpark configuration
with `num_speculative_tokens >= 1` crash on the first decode step on SM120/SM121.**

This repo is the writeup, the patches, the Dockerfiles, and every benchmark harness, including
the ones that **produced wrong answers** and how we caught them.

> ### Credit
> This work stands on **[MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)**,
> the community recipe we originally deployed from and still measure against. Our 2-node
> bring-up, NCCL/CX-7 fabric settings, and speculative-decode configuration all derive from it.
> Where we deviate, [docs/02-before-after.md](docs/02-before-after.md) says so and says why.
> Thanks also to the `tonyd2wild` thread for the loop-vs-temperature root cause we rely on below.

---

## Status

| | |
|---|---|
| **Serving** | `vllm/vllm-openai:v0.27.1` + SM120 DeepGEMM + a DSpark index-width patch |
| **Hardware** | 2× DGX Spark GB10 (`sm_121`), ConnectX-7 point-to-point @ 200 Gb/s, RDMA verified |
| **Model** | `deepseek-ai/DeepSeek-V4-Flash-0731`, TP=2 across both nodes |
| **Spec decode** | DSpark, `num_speculative_tokens=5`, probabilistic draft sampling |
| **Context** | 1,048,576 · verified correct at **377,594** prompt tokens |
| **KV pool** | **1,368,558 tokens** @ 1M ctx, `--gpu-memory-utilization 0.80` (1,499,595 @ 512K / 0.88) |
| **Stability** | 0 asserts at the 12-concurrency saturation point; ~3.3× throughput scaling c=1→c=12 |
| **Tool calling** | 20/20 on an agent-shaped gate; **49/49 tool calls** in a real agent session — [read the incident first](docs/08-the-tool-calling-incident.md) |
| **Status** | **serving production agent traffic** as of 2026-08-12, after a rollback and redeploy |

**What we gained:** upstream base image (no unreproducible vendor overlay), **15% faster prefill
at 512K**, and DeepSeek-V4 improvements that only exist in 0.27.x.

**What we lost, honestly:** **~33% of the KV token pool** (2.03M → 1.37M tokens at 1M context),
because upstream has no `nvfp4_ds_mla` KV dtype, and **decode kernel autotuning for DSpark decode
shapes**, which upstream does not generate. Both are quantified in
[docs/02-before-after.md](docs/02-before-after.md) and both are fixable — see
[docs/07-roadmap.md](docs/07-roadmap.md).

> ### ⚠️ Read this before you copy our config
>
> **1. Test tool calling at your agent's sampling parameters.** This stack passed every benchmark
> we had and still broke our production agent within minutes. vLLM samples DSML *structural*
> tokens at the request temperature, so a stack that is perfect at greedy defaults can emit
> invented tool-call syntax at `temperature=1.0`. Our old smoke test — no temperature, no
> streaming, one toy tool — passed 5/5 throughout the outage. Full postmortem, and the gate that
> would have caught it: **[docs/08-the-tool-calling-incident.md](docs/08-the-tool-calling-incident.md)**.
> Harness: [`harness/toolcall_tars.py`](harness/toolcall_tars.py).
>
> **2. Pin `--revision`.** We found our own production service running unpinned, resolving
> `refs/main` to a *different* checkpoint than the one MiaAI-Lab tested
> (`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`). Both snapshots sat in the cache and `main` chose.
> A re-pull can swap weights under a running service, and cross-repo comparisons are meaningless
> without it.
>
> **3. If you front vLLM with LiteLLM**, its `hosted_vllm` transform recursively strips `strict`
> and `additionalProperties` from every tool, so 0.27's DSML grammar — gated behind
> `tool_choice=="auto" and _any_tool_strict(tools)` — can never be enabled client-side.

---

## The three blockers, in the order you will hit them

Full detail with source locations: **[docs/01-blockers.md](docs/01-blockers.md)**

### 1. DeepGEMM has no `arch_major == 12` dispatch

```
RuntimeError: Assertion error (deepgemm-src/csrc/apis/hyperconnection.hpp:56):
Unsupported architecture
```

vLLM 0.27.1 pins `vllm-project/DeepGEMM @ e21c821f`, which dispatches only `arch_major` 9 and 10.
GB10 reports **12**. `VLLM_USE_DEEP_GEMM=0` does **not** bypass it — the DeepSeek-V4 path calls
DeepGEMM regardless.

**Fix:** rebuild that one component from branch `nv_dev+situ+0810`
(`9e8903799beb0b65d88e5ca08940dd5cd712c7d2`) — the only branch carrying both SM120 kernels and the
`hyperconnection`/`einsum` APIs 0.27.1 calls. See [docker/Dockerfile.vllm027-gb10](docker/Dockerfile.vllm027-gb10).

⚠️ DeepGEMM's `CMakeLists.txt` **overwrites `TORCH_CUDA_ARCH_LIST`** from its own `CUDA_ARCH_LIST`
(default 9.0). Setting the env var alone silently builds the wrong arch. The Dockerfile asserts
the built artifact actually contains SM120 support.

### 2. FlashInfer's SM120 decode kernel is never dispatched  ← *the interesting one*

```
Check failed: num_tokens > 64 (5 vs 64)
```

vLLM computes the DSpark non-causal SWA index width as

```python
noncausal_index_width = cdiv(sliding_window + num_speculative_tokens, 128) * 128
#                     = cdiv(128 + 5, 128) * 128
#                     = 256
```

That "round up to a multiple of 128" matches **FlashMLA on SM100**. But FlashInfer's SM120
decode kernel is only instantiated for **`topk ∈ {128, 512, 1024}`**
(`csrc/sparse_mla_sm120_decode_dsv4.cu`). There is no 256 kernel, so the dispatch gate falls
through to the **prefill-only** orchestrator, which then rejects a 5-token decode batch.

**This is an upstream bug.** Any DSpark config with `k >= 1` on SM120/SM121 hits it immediately.

**Fix** (semantics-preserving, ~15 lines): round up to the next *instantiated* width, not the next
multiple of 128. Padding slots are already `-1` and capped by `swa_topk_lens`, so the extra width
only adds empty split-K chunks. Measured cost **< 0.5 ms/step**. With speculation off it still
yields 128. See [patches/patch-dspark-sm120.py](patches/patch-dspark-sm120.py).

### 3. A silent `torch` downgrade that breaks everything

`pip install flashinfer-python==0.6.12` quietly pulls **torch 2.13.0 → 2.10.0**, breaking every
compiled vLLM extension with errors that point nowhere near the cause. `--no-deps` is mandatory,
and the build asserts torch is unchanged.

---

## Quickstart

```bash
# 1. Build the SM120 DeepGEMM base (on EACH node — no registry between them)
docker build -f docker/Dockerfile.vllm027-gb10 -t vllm027-gb10:sm120 .

# 2. Bake the DSpark patches on top
cp patches/patch-dspark-sm120.py .
docker build -f docker/Dockerfile.vllm027-patched -t vllm027-gb10:patched .

# 3. Start worker first, then head
#    (edit fabric IPs / NCCL device names in the compose files for your machines)
ssh worker 'docker compose -f docker-compose.pollux.yml up -d'
ssh head   'docker compose -f docker-compose.castor.yml  up -d'

# 4. Verify (do not skip — a broken build looks healthy until the first decode)
python3 bench/needle.py                    # correctness at depth
DEPTHS=8000,128000 python3 bench/needle.py
python3 bench/conc12.py                    # saturation, no assert
```

Boot to `/health` 200 is **~6 minutes**.

We apply the patcher at build time rather than `COPY`ing finished files **on purpose**: its
anchors are exact strings from vLLM 0.27.1 / FlashInfer 0.6.16.post3, so a changed base image
**fails the build** instead of silently shipping upstream's unpatched file.

---

## Repo map

| Path | What |
|---|---|
| [docs/01-blockers.md](docs/01-blockers.md) | The three blockers with source locations and reasoning |
| [docs/02-before-after.md](docs/02-before-after.md) | Vendor-overlay 0.21 vs upstream 0.27.1, measured |
| [docs/03-what-i-wish-id-had.md](docs/03-what-i-wish-id-had.md) | The things that would have saved days |
| [docs/04-measurement.md](docs/04-measurement.md) | Methodology — **including the instruments that lied** |
| [docs/05-dead-ends.md](docs/05-dead-ends.md) | What we tried that does not work, so you can skip it |
| [docs/06-the-json-non-bug.md](docs/06-the-json-non-bug.md) | A "model bug" that turned out to be batch-size numerics |
| [docs/07-roadmap.md](docs/07-roadmap.md) | Where the remaining performance is, ranked |
| [docs/08-the-tool-calling-incident.md](docs/08-the-tool-calling-incident.md) | **How this stack passed every gate and still broke production** — and the gate that catches it |
| [harness/toolcall_tars.py](harness/toolcall_tars.py) | Agent-shaped tool-calling gate: temp 1.0, streaming, real toolset, n≥20, with `--self-test` |
| `docker/` | Both Dockerfiles + both compose files, as deployed |
| `patches/` | The anchored, idempotent patcher |
| `bench/` | Every harness, including the ones that produced wrong answers |

## License

MIT for everything original here. The patches are derivative of vLLM (Apache 2.0) and FlashInfer
(Apache 2.0); upstreaming them is the intent — see [docs/07-roadmap.md](docs/07-roadmap.md).

## Reproducibility caveat

Every number here comes from **one specific pair of machines**. Spec-decode acceptance varies
enough run-to-run that we measured the *same* config at 18.3 tok/s and 37.0 tok/s in different
sessions. Treat single-run numbers here as indicative, and read
[docs/04-measurement.md](docs/04-measurement.md) before comparing against your own.
