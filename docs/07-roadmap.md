# Where the remaining performance is, ranked

Ranked by expected value. Items 1 and 2 are, in our view, the two things standing between upstream
vLLM and parity with a vendor overlay on GB10.

---

## 1. Upstream the index-width fix  ·  *highest value to everyone else*

**Status: not yet filed. Patch is written and running in production.**

vLLM's DSpark non-causal index-width formula
(`cdiv(sliding_window + num_speculative_tokens, 128) * 128`) matches FlashMLA/SM100's contract but
not FlashInfer SM120's instantiation set `{128, 512, 1024}`. Consequence: **every DSpark
configuration with `num_speculative_tokens >= 1` crashes on the first decode on SM120/SM121.**

Our fix is a ~15-line strict subset of what upstream should merge — it only changes the rounding
target, only for `is_dspark`, and leaves FlashMLA untouched. See
[01-blockers.md](01-blockers.md) §2 and `patches/patch-dspark-sm120.py`.

A more complete upstream fix would query FlashInfer for its instantiated widths rather than
hardcoding `{128, 512, 1024}`, so the two stay in sync when new kernels are compiled.

## 2. Generate DSpark uniform-decode autotune shapes

**Status: diagnosed, not implemented. Probably the largest available decode win.**

vLLM 0.27.1's `flashinfer_sparse_mla_decode_autotune_warmup()` autotunes exactly **one** shape —
`_SPARSE_MLA_MIXED_WARMUP_TOKENS`, clamped to `max_num_batched_tokens`. It does not enumerate the
**uniform-decode** shapes that DSpark actually runs, which are
`cudagraph_capture_size × (1 + num_speculative_tokens)`.

The vendor overlay did (`Including 2 DSpark uniform-decode sparse MLA autotune shapes`). Upstream
does not. The consequence appears only as a `WARNING`, mid-progress-bar, during CUDA graph capture:

```
No tuned config covers sparse_mla_sm120_decode_dsv4
  input_shapes=((60, 32, 512), (60, 128), (60, 32, 130, 512), ...);
  falling back to runner=SparseMlaDecodeV3Runner tactic=-1.
  This shape is outside the tuning bucket range -- expand tuning_buckets / max_num_tokens
  during the next tuning pass to avoid this perf cliff.
```

We see this for shapes **60, 24, 12, 6** — i.e. every decode shape we actually run. vLLM is
telling you, in its own words, that you are on a perf cliff.

**Implementation sketch:** in `flashinfer_sparse_mla_decode_autotune_warmup`, when the DSpark
backend is selected, additionally autotune `n × (1 + k)` for each configured cudagraph capture
size. Cost is a one-off increase in warmup time; the result persists to the autotune cache under
`VLLM_CACHE_ROOT/flashinfer_autotune_cache/`.

**Do not** instead memoize the `tactic=-1` fallback (`if int(tactic) > 0:` → `if True:`). We
profiled that path: the dispatch overhead is **0 of 6005 samples**, because resolution happens at
CUDA-graph capture. Caching `-1` only reaches the *untuned* kernel faster. The tempting one-liner
treats the symptom's shadow.

**Unquantified.** We have not measured the win, because doing so needs a patched warmup and a boot.
Quantify it before you believe it — the discipline in [04-measurement.md](04-measurement.md)
applies to our own hypotheses too.

## 3. Recover the KV pool: an NVFP4 MLA KV dtype upstream

> **Do the free thing first.** Check whether you are leaving unified memory idle before treating
> this as a kernel problem. We were: at `--gpu-memory-utilization 0.80`, ~12 GiB per node sat
> unused, and moving to 0.88 nearly doubled KV (13,793 → 25,417 blocks). vLLM prints the headroom
> at startup. Only the ~1.56× per-token footprint difference is inherent.

Upstream's only DSV4 MLA KV dtype is `fp8_ds_mla` at **584 B/token/layer**. The overlay's
`nvfp4_ds_mla` fit **~1.56× more tokens** in the same memory — measured, mechanism inferred.

That is worth **2.03M vs 1.27M KV tokens**, i.e. **~2× vs 1.21× concurrency at 1M context**. For
multi-tenant long-context serving this is the single biggest functional gap between the two stacks.

Getting it upstream is a real kernel contribution, not a patch. Until then, if you need concurrent
1M-context streams, the overlay is still the better choice — say so honestly when recommending
this migration.

## 4. Apples-to-apples decode comparison against the overlay

We deliberately did not publish a decode delta, because the two stacks were measured with
different instruments and acceptance variance swamps the difference
([04-measurement.md](04-measurement.md)). Running `bench/accept_by_depth.py` against **both**
stacks, same prompts, same sampling, several repetitions, would settle it.

This is the measurement we most want and have not made. It also gates item 2: if decode is already
at parity, the autotune cliff matters less than the warning implies.

## 5. Loop and structured-output behaviour under agent traffic

This checkpoint has a documented repetition-loop mode on local GB10 stacks, gated almost entirely
by **sampling temperature** rather than by anything in the serving stack (community finding across
two independent 2×GB10 labs; loop rate invariant across spec on/off, k=3/5, KV dtype, and vLLM
version). The checkpoint's own generation config specifies temperature 1.0; serving it at
temperature 0 is what triggers loops.

If you drive this model with agents, pin `temperature: 1.0 / top_p: 0.95` at your gateway and
leave thinking on. Separately, note that **thinking-on plus a small `max_tokens` returns empty
`content`**, because reasoning tokens consume the budget — give probes 500+.

## 6. Smaller, known

- **Client `stop` sequences can decapitate reasoning** on some builds — the detokenizer matches
  stops inside the think stream, yielding null content. Send no stops, or patch.
- **Do not add repetition/frequency penalties** on this model — documented illegal-memory-access
  crash risk.
- The container ships `nvcc` without CUDA dev headers; anything you build in-image needs
  `cuda-libraries-dev-13-0`.

---

## Contributing

If you improve on any of this — especially items 1–3 — please open a PR or an issue. The most
useful contribution is a **measurement that contradicts something here**; several claims in this
repo are hypotheses labelled as such, and we would rather be corrected than cited.
