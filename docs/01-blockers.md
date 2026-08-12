# The three blockers, with reasoning

Each of these presents as a different error, at a different layer, at a different point in boot.
Cleared in order, because each one hides the next.

---

## Blocker 1 — DeepGEMM: no `arch_major == 12` dispatch

### Symptom

```
RuntimeError: Assertion error (deepgemm-src/csrc/apis/hyperconnection.hpp:56):
Unsupported architecture
```

Fires during model init, before any request.

### Why

GB10 is compute capability **12.1** (`sm_121`, family 120). vLLM 0.27.1 pins
`vllm-project/DeepGEMM @ e21c821f`, whose host-side dispatcher branches on `arch_major` and knows
only **9** (Hopper) and **10** (datacenter Blackwell).

Two things make this more annoying than it looks:

1. **`VLLM_USE_DEEP_GEMM=0` does not save you.** The DeepSeek-V4 path calls into DeepGEMM
   regardless of that flag. Neither does `VLLM_MOE_USE_DEEP_GEMM=0`.
2. **The published arm64 image ships DeepGEMM built for SM100a only.** Confirm with
   `strings _C.cpython-312-aarch64-linux-gnu.so | grep -o '1[02]0a'` — a working build shows
   both `100a` and `120a`; the stock image shows only `100a`. vLLM's own platform gate
   (`vllm/platforms/cuda.py`) reports DeepGEMM as *supported* on capability family 120, so vLLM
   confidently calls a kernel set that was never compiled.

DeepGEMM upstream already supports the arch — `cmake/external_projects/deepgemm.cmake` lists
`12.0a` and `12.1a`. The published image just intersected `CUDA_ARCHS` down to datacenter
Blackwell. So only that one component needs rebuilding.

### Fix

Build branch **`nv_dev+situ+0810`** at `9e8903799beb0b65d88e5ca08940dd5cd712c7d2`. We tried
several refs; this is the only one carrying **both** the SM120 kernels **and** the
`hyperconnection` / `einsum` APIs that vLLM 0.27.1 calls. Older SM120-capable refs fail to link
against 0.27.1; the pinned ref lacks SM120.

Install it as a **top-level `deep_gemm`** package: `vllm/utils/deep_gemm.py` imports
site-packages `deep_gemm` first and only falls back to `vllm.third_party.deep_gemm`, so this
shadows the under-built vendored copy without patching the wheel.

### Two traps in the build itself

- **The image has `nvcc` but no CUDA dev headers.** The build dies on
  `fatal error: cusparse.h: No such file`. Install `cuda-libraries-dev-13-0` (torch here is cu130,
  so the 13-0 series is the matching one).
- **DeepGEMM's `CMakeLists.txt` overwrites `TORCH_CUDA_ARCH_LIST`** from its own `CUDA_ARCH_LIST`
  (default 9.0). Exporting the env var is silently ignored. This produced a *successful build of
  the wrong architecture* — which then fails at runtime exactly like no fix at all. The Dockerfile
  now asserts SM120 support is present in the built artifact, at build time.

---

## Blocker 2 — FlashInfer SM120 sparse-MLA decode is never dispatched

**This is the one worth reading. It is an upstream bug, not a configuration problem.**

### Symptom

```
Check failed: num_tokens > 64 (5 vs 64)
```

Fires on the **first decode step** — the model loads, warms up, captures CUDA graphs, answers
`/health`, and then dies the moment a real generation starts. Prefill-only workloads look fine.

### Why

vLLM computes the DSpark non-causal sliding-window index width in
`vllm/v1/attention/backends/mla/sparse_swa.py`:

```python
self.noncausal_index_width = (
    cdiv(self.window_size + self.num_speculative_tokens, 128) * 128
    if self.is_dspark
    else 0
)
```

With `window_size=128` and `num_speculative_tokens=5`, that is `cdiv(133, 128) * 128` = **256**.

"Round up to a multiple of 128" is correct for **FlashMLA on SM100**. It is wrong for FlashInfer
on SM120, whose decode kernel is only instantiated for a fixed set of `topk` values —
**{128, 512, 1024}** — in the launch switch of `csrc/sparse_mla_sm120_decode_dsv4.cu`.

There is no 256 kernel. The dispatch gate therefore does not match, control falls through to the
**prefill-only orchestrator**, and that orchestrator asserts because a decode batch (5 tokens) is
smaller than its 64-token floor.

So: **every DSpark configuration with `num_speculative_tokens >= 1` crashes on SM120/SM121 at the
first decode.** `k=5` gives 133→256. Even `k=1` gives 129→256. The only way to land on an
instantiated width by accident is `k=0`, i.e. speculation disabled.

### Fix

Round up to the next **instantiated** width instead of the next multiple of 128:

```python
if self.is_dspark:
    _min_width = self.window_size + self.num_speculative_tokens
    for _width in (128, 512, 1024):
        if _min_width <= _width:
            break
    else:
        raise ValueError(...)
    self.noncausal_index_width = _width
else:
    self.noncausal_index_width = 0
```

**Why this is safe, not a fudge:** the index buffer's unused slots are already written as `-1`
and the *active* length is capped separately by `swa_topk_lens`. Widening 256→512 therefore adds
only empty split-K chunks, which the kernel early-exits — `decode_dsv4_kernel.cuh` computes its
chunk count from the *effective* length, not the buffer width. Measured cost **< 0.5 ms/step**.
With speculation off the expression still yields 128, so non-speculative and non-DSpark paths are
untouched. FlashMLA/SM100 keeps seeing a multiple of 128 because the branch is gated on
`is_dspark`.

### Do NOT "fix" this by shrinking the sliding window

Setting `sliding_window` to 123 makes 123+5=128 fit an instantiated width and the crash goes
away. **Don't.** `sliding_window` is a trained architecture parameter; shrinking it silently
truncates every layer's attention span. You get a quality regression with no error message.

### A diagnostic worth keeping

We also replaced the opaque C++ `ICHECK` with a Python error that prints **every** dispatch-gate
term (`num_tokens`, `num_heads`, `topk`, `d_qk`, `kv_page_block_size`, tensor shapes). It has
never fired since the real fix — but working out *which* of six gate terms was wrong, from
`Check failed: num_tokens > 64`, cost hours. Included in the patcher.

---

## Blocker 3 — the silent `torch` downgrade

### Symptom

Undefined symbols / ABI errors from vLLM's compiled extensions, pointing nowhere near the cause.

### Why

```bash
pip install flashinfer-python==0.6.12     # quietly downgrades torch 2.13.0 -> 2.10.0
```

Every compiled vLLM extension in the image was built against 2.13. Nothing warns you.

### Fix

`--no-deps`, always — plus a build-time assertion that torch is still what the base image shipped.
Cheap to add, and it converts a confusing runtime failure into an obvious build failure.

---

## Why the order matters

Each blocker masks the next, and each fails at a *later* stage of boot than the one before:

| # | Fails at | Looks like |
|---|---|---|
| 1 | model init | "unsupported architecture" — clearly an arch problem |
| 2 | first decode | a healthy server that dies on first use |
| 3 | import time | unrelated ABI noise |

If you fix 2 before 1 you will never see 2 work, and you may conclude the patch is wrong.
Fix them in order and verify each with a real generation, not a `/health` check.
