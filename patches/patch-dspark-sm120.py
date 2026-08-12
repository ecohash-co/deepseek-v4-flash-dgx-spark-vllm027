#!/usr/bin/env python3
"""Patch vLLM 0.27.1 + FlashInfer 0.6.16/0.6.17 for DSv4 DSpark decode on SM120/SM121.

Usage:  python3 patch-dspark-sm120.py <sparse_swa.py> <_sparse_mla_sm120.py>

Edits (exact-string, fails loudly if anchors are missing):
1. vllm/v1/attention/backends/mla/sparse_swa.py
   Round the DSpark non-causal SWA index width up to a FlashInfer
   decode-dsv4-instantiated topk (128/512/1024) instead of the next
   multiple of 128 (which yields 256 for window=128, k=5 — a width the
   SM120 decode kernel is not compiled for).
2. flashinfer/mla/_sparse_mla_sm120.py
   Replace the silent decode->prefill-orchestrator fallthrough (which dies
   in a prefill-only C++ ICHECK) with a Python error that prints every
   dispatch-gate term.
"""
import sys

VLLM_OLD = """        self.noncausal_index_width = (
            cdiv(self.window_size + self.num_speculative_tokens, 128) * 128
            if self.is_dspark
            else 0
        )
"""

VLLM_NEW = """        # PATCH(dspark-sm120): FlashInfer's SM120 decode-dsv4 kernel is only
        # instantiated for topk in {128, 512, 1024}
        # (csrc/sparse_mla_sm120_decode_dsv4.cu launch switch), so round the
        # non-causal index width up to the next instantiated width instead of
        # the next multiple of 128. Slots past swa_len are written as -1
        # (skipped by the kernel) and swa_lens caps the active length, so the
        # extra width only adds empty split-K chunks. FlashMLA (SM100) still
        # sees a multiple of 128.
        if self.is_dspark:
            _min_width = self.window_size + self.num_speculative_tokens
            for _width in (128, 512, 1024):
                if _min_width <= _width:
                    break
            else:
                raise ValueError(
                    f"DSpark non-causal SWA index width {_min_width} exceeds "
                    "the largest instantiated sparse-MLA decode topk (1024)"
                )
            self.noncausal_index_width = _width
        else:
            self.noncausal_index_width = 0
"""

FI_OLD = """        module.sparse_mla_sm120_paged_attention(
"""

FI_NEW = """        # PATCH(gate-diagnostic): decode-sized calls reaching this point would
        # die in the prefill-only orchestrator with an opaque C++ ICHECK
        # ("Decode (num_tokens <= 64) must go through ..."). Fail in Python
        # instead, with every dispatch-gate term visible.
        if num_tokens <= _DECODE_MAX_TOKENS:
            raise RuntimeError(
                "SM120 sparse-MLA: decode-sized call matched no decode kernel: "
                f"num_tokens={num_tokens}, num_heads={num_heads}, topk={topk}, "
                f"d_qk={d_qk}, kv_pbs={kv_pbs}, extra_topk={extra_topk}, "
                f"model_type={model_type}, "
                f"kv_cache.shape={tuple(kv_cache.shape)}, "
                f"indices.shape={tuple(indices.shape)}; decode-dsv4 needs "
                "d_qk=512, kv page_block_size=64, (num_heads, topk) in "
                "{8,16,32,64,128} x {128,512,1024}."
            )
        module.sparse_mla_sm120_paged_attention(
"""


def apply(path: str, old: str, new: str, expected_count: int) -> None:
    with open(path) as f:
        src = f.read()
    n = src.count(old)
    if new.strip("\n") in src:
        print(f"[skip] {path}: already patched")
        return
    if n != expected_count:
        sys.exit(f"[FAIL] {path}: anchor found {n}x, expected {expected_count}. "
                 "File does not match vLLM 0.27.1 / FlashInfer 0.6.16-0.6.17.")
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    compile(src, path, "exec")  # syntax check
    print(f"[ok] patched {path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    apply(sys.argv[1], VLLM_OLD, VLLM_NEW, 1)
    apply(sys.argv[2], FI_OLD, FI_NEW, 1)
