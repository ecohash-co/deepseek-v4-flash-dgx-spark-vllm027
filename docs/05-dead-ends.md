# Dead ends — do not spend a boot on these

Each of these cost us at least one 6–13 minute restart cycle. Written so you can skip them.

---

## `VLLM_USE_DEEP_GEMM=0` does not bypass DeepGEMM

The DeepSeek-V4 path calls into DeepGEMM regardless of the flag. Same for
`VLLM_MOE_USE_DEEP_GEMM=0`. You must actually build a DeepGEMM that supports your architecture.

## B12X MoE backend is unreachable for this checkpoint

`--kernel-config '{"moe_backend":"flashinfer_b12x"}'` **fails at boot**. The experts are natively
**MXFP4**; `FlashInferB12xExperts` asserts `quant_dtype == "nvfp4"`
(`fused_moe/experts/flashinfer_b12x_moe.py`), and the MXFP4 oracle has no b12x entry.

The vendor overlay reached B12X only via an **out-of-tree NVFP4 expert conversion**, so any
"B12X is much faster" advice from overlay-based recipes **does not transplant**. `DEEPGEMM_MXFP4`
is the correct in-tree choice. Getting to B12X upstream needs an offline NVFP4 requant
(`moe_quant_algo: NVFP4`) — a project, not a flag.

## `VLLM_USE_BREAKABLE_CUDAGRAPH=0` would be actively harmful

It is **auto-enabled** for `DeepseekV4ForCausalLM` (`config/vllm.py`) precisely because that class
lacks `@support_torch_compile`. Recipes that pin it to 0 are targeting *other* model classes.
Leave it alone.

## FlashInfer 0.6.17 does not fix the SM120 dispatch

Identical assert at the identical source line as 0.6.16.post3 — verified by reading the source,
not by booting. Save yourself the build.

And **0.6.12 is not a fallback either**: it lacks `swa_topk_lens`, which vLLM 0.27.1 passes.
The version window that works with 0.27.1 is narrow.

## Do not shrink `sliding_window` to make the index width fit

`sliding_window=123` makes `123+5=128` land on an instantiated kernel and the crash disappears.
It is a trained architecture parameter. Shrinking it silently truncates every layer's attention
span — quality regression, no error. See [01-blockers.md](01-blockers.md).

## `--block-size 64` does not change the dispatch gate

We guessed this, reasoning that the failing gate term was the page block size. Wrong:
`_packed_kv_page_block_size()` reads from the **tensor layout**, not `--block-size`. The result
was trading a clear assert for a `ZeroDivisionError` deeper in `kv_cache_interface.py`.

This is the entry in this list we are least proud of — it was a guess where a measurement was
available. Print the gate terms (see [03-what-i-wish-id-had.md](03-what-i-wish-id-had.md) §3)
instead of theorizing about which one is wrong.

## Memoizing FlashInfer's `tactic=-1` fallback

Tempting one-line change: `if int(tactic) > 0:` → `if True:` in `_sparse_mla_sm120.py`, so the hot
cache stops re-resolving on every call.

Profiling says the dispatch costs nothing (0 of 6005 samples). And more importantly `tactic=-1`
is the **heuristic fallback for an untuned shape** — caching it just means reaching the wrong
kernel faster. The real fix is to get the shape tuned. See [07-roadmap.md](07-roadmap.md).

## `nohup` / `screen -dmS` under `sudo -n -u` without a tty (macOS)

Silently fails to detach. If you are orchestrating long benchmark runs on a remote box this way,
you will think the job died. Run in the foreground of a persistent SSH session instead.

## Two containers with `restart: unless-stopped` on the same port

Both will try to start at boot and race for the port and the GPU. The loser's logs do not say
"someone else has the port". Set the retired container to `restart=no` **explicitly** — do not
rely on it being stopped.
