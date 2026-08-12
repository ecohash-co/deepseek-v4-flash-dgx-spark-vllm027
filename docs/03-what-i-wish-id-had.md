# The things I wish I'd had

Written for the next person. Roughly ordered by how much time each would have saved.

---

## 1. "A healthy server is not a working server"

The single most expensive misconception. On this stack the model loads, warms up, captures CUDA
graphs, and serves `/health` **200** — and then dies on the first decode step. Every quick check
that stops at `/health` will tell you the deploy succeeded.

**Have this instead:** a smoke test that does a real generation of >10 tokens at >1 batch size,
run automatically after every boot. `bench/needle.py` is ours. Nothing else counts as "up".

## 2. A build that fails loudly instead of shipping something subtly wrong

Three separate times a build "succeeded" and produced a broken artifact:

- DeepGEMM compiled for the **wrong architecture**, because its `CMakeLists.txt` overwrites
  `TORCH_CUDA_ARCH_LIST` from its own variable.
- A `pip install` that silently **downgraded torch** and broke every compiled extension.
- A patch that didn't apply because a file had shifted, leaving upstream's original in place.

**Have this instead:** build-time assertions for every property you depend on. Ours check for
SM120 support in the built object, torch's version being unchanged, both patch markers being
present, and `ast.parse` on both patched files. They cost seconds and each one has already caught
a real failure.

Corollary: **apply patches with anchored, exact-string patchers, not `COPY` of finished files.**
A `COPY` silently ships a stale copy of upstream's file when the base image moves. An anchored
patcher fails the build.

## 3. The dispatch-gate terms, printed

`Check failed: num_tokens > 64 (5 vs 64)` tells you one term of a **six-term** gate. Working
backwards from it to "the topk is 256 and only {128, 512, 1024} are compiled" took hours of
reading CUDA template instantiation lists.

**Have this instead:** before you theorize, replace the opaque assert with something that prints
every gate term — `num_tokens`, `num_heads`, `topk`, `d_qk`, `kv_page_block_size`, and the actual
tensor shapes. That one change turns a research project into a lookup. It's in our patcher and
has never fired since.

## 4. Someone telling me to measure before optimizing

We received a specific, expert, code-accurate analysis: FlashInfer refuses to memoize the
`tactic=-1` heuristic fallback, so `AutoTuner.choose_one` should re-run on every decode call —
predicted ~90+ calls/step on the Grace CPU. The premise checked out. The patch was one line.

Profiling first (`py-spy`, 6005 samples over 25 s under decode load) showed
**`choose_one` in 0 samples**. The dispatch happens during CUDA-graph capture, not per step. The
patch would have bought nothing, cost a production restart, and been "confirmed" by any A/B whose
noise exceeds the effect — which, given acceptance variance here, is all of them.

**But the story does not end there**, and this is the real lesson: profiling refuted the
*mechanism* while the *symptom* was pointing at something real. `tactic=-1` is not a caching
failure — it means the shape **was never tuned**. See [07-roadmap.md](07-roadmap.md), item 2.
Memoizing `-1` would have cached the wrong kernel faster.

Two things to take from that: profile before applying a performance patch, and when you refute
someone's mechanism, **check whether their symptom still needs explaining.**

Note for profiling in containers: plain `docker exec py-spy` fails with
`No python processes found`. You need `docker exec --privileged` for ptrace.

## 5. A benchmark I could trust, and a list of the ones I couldn't

We reported a decode regression that did not exist, from `benchmarks/longctx.py`, which derives
decode throughput as `warm_N − warm_1`. That assumes stable cached prefill. On 0.27.1 the warm
prefill for an *identical* call ranged **0.38 s … 17.18 s**, producing artifacts up to
**1.5×10⁸ tok/s** and false "COLLAPSE" flags.

**Have this instead:** measure decode by streaming and dividing tokens by wall time *after the
first token* (`bench/decode_direct.py`), and scrape speculative-acceptance counters on the
identical request (`bench/accept_by_depth.py`). If throughput moves and acceptance moves with it,
you have not found a stack bug — you have found the model being less certain.

See [04-measurement.md](04-measurement.md) for the full list of instruments that lied.

## 6. Knowing that greedy decoding is not batch-invariant

We spent real effort on a "degenerate JSON output bug" that reproduced **10/10 at batch 1 and
0/10 at batch 4**. It was not a bug in anything. Full writeup:
[06-the-json-non-bug.md](06-the-json-non-bug.md).

**Have this instead:** before blaming the stack for a bad output, vary batch size, try other
prompts, and look at the first-token logprobs. A contested first token (top-2 gap of ~1 nat) that
resolves differently under different execution shapes is model behaviour, not a defect.

And: **quote the concurrency of any failure rate you report.** Ours changed from "13/20" to
"20/20" for the identical cell purely because one harness ran serially and the other ran under
ambient load. That artifact briefly looked like a version regression.

## 7. A map of which upstream tuning the vendor overlay was silently doing

The overlay generated **DSpark uniform-decode autotune shapes**; upstream generates only a single
mixed-token shape. Nobody documents this, and the consequence appears only as a `WARNING` buried
in CUDA-graph capture output:

```
No tuned config covers sparse_mla_sm120_decode_dsv4 ... falling back to tactic=-1.
This shape is outside the tuning bucket range -- expand tuning_buckets / max_num_tokens
during the next tuning pass to avoid this perf cliff.
```

vLLM is literally telling you it is on a perf cliff, at `WARNING` level, in the middle of a
progress bar. **Grep your boot logs for `WARNING` before declaring a migration complete.**

## 8. Rollback that was tested, not just written down

We kept the previous container on both nodes, stopped, with `restart=no` so it could not collide
on the port at boot, plus an image snapshot on disk. We exercised the rollback twice. That is the
only reason experimenting on a production inference service was reasonable.

Two sharp edges worth stealing:
- `docker compose` reports the old container as an **orphan** of the project. `--remove-orphans`
  would delete your rollback.
- Set the old container's restart policy to `no` **explicitly**. Two containers with
  `unless-stopped` bound to the same port will race on reboot, and the loser's logs are unhelpful.

## 9. `docker compose config` before every deploy

An edit that removed two bind-mounts also silently swallowed the `environment:` key — every NCCL
setting, master address, and cache path. Rendering the config caught it before deploy.

Better still: **diff the rendered config**, old versus new, and confirm the only differences are
the ones you intended. That is a five-second check that makes compose edits boring.
