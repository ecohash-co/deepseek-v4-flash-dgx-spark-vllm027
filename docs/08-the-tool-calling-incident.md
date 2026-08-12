# The tool-calling incident: how a stack passed every gate and still broke production

**TL;DR** — 0.27.1 passed needle recall at 377,594 tokens, arithmetic, JSON mode, 12-way
concurrency saturation and zero asserts, then broke our agent so badly it was unusable inside
minutes. We rolled back, redeployed, and eventually **reproduced it: roughly 1% of tool calls emit
invalid DSML as content instead of a structured call.** That rate is invisible to benchmarks and
brutal for agents — at 1%/call, a 50-call session has a ~40% chance of at least one malformed call.

**It is fixed structurally.** vLLM 0.27 already ships a DSML grammar that makes invalid markup
unsamplable; it is gated off for exactly the case agents use. One anchored patch ungates it:
[`patches/patch-dsml-grammar.py`](../patches/patch-dsml-grammar.py).

If you take one thing from this page: **test tool calling at the sampling parameters your agent
actually uses, not at the defaults your benchmark uses.**

> **Revision history of this page**, because being wrong in public is part of the record: an earlier
> version said the failure *did not reproduce*, based on a clean redeploy plus a 49-call agent
> session. That was a sampling artifact — at ~1% per call, a 49-call session is clean about 60% of
> the time. It reproduced on the next run. **"Not reproduced" is not "absent"**, and we said so too
> confidently.

---

## What happened

| | |
|---|---|
| 08-11 evening | Cut production to 0.27.1. All benchmarks green. |
| 08-11 → 08-12 00:26 | Agent traffic degrades badly. Operator: *"it is hella busted and can't do tool calling."* |
| 08-12 07:00 | Rolled back to the 0.21 vendor-overlay stack. Service restored. |
| 08-12 ~11:00 | Redeployed 0.27.1 on the **exact failing config**. Ran a real agent workload. |
| | **49 tool calls, 48/48 results OK, zero malformed output.** Not reproduced. |

The symptom, from agent transcripts captured during the bad window, was **invalid tool-call
markup** — the model emitting invented grammar rather than the checkpoint's DSML:

```
<｜DSML｜parameter name="exec">        # a parameter tag used as an invoke
<｜DSML｜exec_command>                 # a tag that does not exist
</｜DSML｜tool_calls></｜DSML｜tool_calls></｜DSML｜tool_calls>   # stray repeated closers
```

No parser can rescue invented syntax. vLLM correctly passed it through as content, so the agent
received prose where it expected a callable tool, and lost the thread immediately.

## What it was *not*

We spent real effort eliminating the obvious suspects. Recording them because each one looked
plausible and each one was wrong:

| Suspect | Verdict | How it was killed |
|---|---|---|
| Tool-call parser changed between versions | ✗ | Replayed a valid completion through 0.27's streaming parser chunk-by-chunk — clean extraction. |
| Chat template / tool declaration differences | ✗ | Rendered the same request through both images CPU-only: **byte-identical prompt strings and token IDs**. |
| The checkpoint's dict-args encoder bug | ✗ | Never reached. The API rejects dict `arguments` at Pydantic validation first — spec-correct, since OpenAI types it as a string. |
| A dropped `--reasoning-config` flag | ✗ | 0.27's parser **hardcodes** `DSML_THINK_START`/`_END` as module constants; the flag only feeds thinking-*budget* machinery, which our agent never uses. |
| Structured-output grammar regression | ✗ | Grammar was off in *both* stacks — see below. It could not have regressed because it never ran. |

That fourth row deserves a note, because it was our leading hypothesis for several hours and it
was *elegant*: 0.21's compose passed `--reasoning-config` with explicit `<think>`/`</think>`
delimiters and 0.27's did not, while both forced `thinking:true`. It fit the symptom exactly. It
was also wrong — reading the parser source took ten minutes and would have saved all of it.

**Conclusion: the failure was in generation, not in parsing or plumbing.** Everything downstream
of the sampler is exonerated by evidence rather than argument.

## The structural-token trap (why sampling params are the whole story)

MiaAI-Lab's [`docs/DSML_SYNTAX_TEMP_ASYMMETRY.md`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
documents the mechanism independently: **vLLM samples DSML *structural* tokens at the request
temperature.** The tags that frame a tool call are drawn from the same distribution as prose. At
`temperature=1.0` the syntax itself can derail.

We run temp 1.0 / top_p 0.95 deliberately — it is the mitigation for a *separate* repetition-loop
problem. So our agent traffic sits exactly where structural sampling is most fragile, while every
benchmark we had ran at defaults.

vLLM 0.27.1 ships a complete DSML xgrammar structural tag that would constrain this. **It is
gated off for the case that needs it most:**

```python
# vllm/tool_parsers/abstract_tool_parser.py
if not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:      # defaults True
    return None
# vllm/tool_parsers/structural_tag_registry.py
if tool_choice == "auto" and not _any_tool_strict(tools):
    return None                                    # ← agents live here
```

`deepseek_v4` **is** in `XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS`, so the grammar exists and is
reachable. But agents send `tool_choice="auto"` and rarely mark tools `strict`, so it silently
does nothing.

⚠️ **If you front vLLM with LiteLLM, you cannot fix this client-side.** LiteLLM's `hosted_vllm`
transform calls `_remove_strict_from_schema()`, which **recursively deletes every `strict` key**
(and `_remove_additional_properties()` alongside it). `strict: true` never reaches vLLM. We found
this only after building a probe around it.

## The gate

[`harness/toolcall_tars.py`](../harness/toolcall_tars.py) — the test we should have had. It differs
from a typical tool-calling smoke test in four ways that all turned out to matter:

1. **Agent sampling** — `temperature=1.0`, `top_p=0.95`. A greedy probe cannot see this failure class.
2. **Streaming** — tool calls assembled from deltas, as an agent receives them.
3. **A realistic toolset** — 5 tools with enum / array / nested-object params, not one toy function.
4. **n≥20** — the failure is stochastic; a single sample proves nothing either way.

It checks the real signature, not merely "did a tool call come back": arguments must parse as JSON
and satisfy the declared `required` keys, and **no raw DSML may leak into content**. One leak is
disqualifying.

```
PASS RATE 20/20 = 100%
```

Two practices that made it trustworthy, both learned the hard way in the same week:

- **Calibrate against a known-good stack before you trust it.** We ran it against the old 0.21
  stack first (20/20, via proxy and direct). A gate never shown to pass a good stack proves
  nothing when it passes a new one.
- **Test the instrument.** `--self-test` runs the detector against the real malformed strings from
  the incident. It immediately caught that our first regex **missed stray closing tags**, because
  it did not allow `/` after `<`. The gate was broken when written, and only the self-test knew.

We had already been bitten by a bad instrument days earlier: a multi-turn probe reported a
checkpoint encoder bug that did not exist, because it swallowed an HTTP 400 into a generic failure
branch. The 400 was correct API behavior. **Verify your instruments before you trust their verdicts.**

## Reproducibility warning: pin your revision

While writing this up we found our production service running **unpinned**:

```
revision=None
refs/main -> 7872f01b1d1fe23eabc4c98b48bffcef5a386062
```

MiaAI-Lab's tested and documented revision is `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`. Both
snapshots were in our cache; `main` silently decided which one we served.

Two consequences worth internalizing:

1. **A re-pull can swap weights under a running service.** This is an entire class of "why did it
   change overnight" bug, and it is invisible unless you look.
2. **Cross-repo comparisons are void without it.** Every MiaAI finding we lean on — including the
   temperature analysis above — was validated on a different checkpoint than we were running.

Add `--revision <sha>` to your serve command. We did not, and it cost us the ability to say
whether our results and theirs are even about the same weights.

## The reproduction

Sampling across one afternoon, `temperature=1.0`, `top_p=0.95`, streaming, 5-tool set:

| Stack | Samples | DSML leaks |
|---|---|---|
| 0.21 vendor overlay (baseline) | 40 | 0 |
| **0.27.1, unpatched** | **~140** | **1** |
| 0.27.1 + grammar patch | 80 | 0 |

The one leak was a run of stray closing tags in `content`:

```
</｜DSML｜invoke> </｜DSML｜tool_calls>
```

Two things worth noting about how nearly we missed it:

1. **It is a *closing* tag.** Our detector's first regex required `[<\[]` followed by the pipe, so
   it did not match `</｜DSML｜…`. The `--self-test` caught that; nothing else would have. The run
   that produced this leak would otherwise have been scored 20/20 clean.
2. **The very next 40-sample run was clean.** A ~1% defect hides trivially in small samples. Do not
   conclude "fixed" from one green run — we did exactly that earlier in this same investigation and
   had to retract it.

## The fix: ungate the grammar vLLM already has

[`patches/patch-dsml-grammar.py`](../patches/patch-dsml-grammar.py) ·
[`docker/Dockerfile.vllm027-patched-r2`](../docker/Dockerfile.vllm027-patched-r2)

One anchored edit to `structural_tag_registry.get_model_structural_tag()`, gated behind an env var
so upstream behavior is one restart away:

```python
if tool_choice == "auto" and not _any_tool_strict(tools):
    import os as _os
    if _os.environ.get("VLLM_DSML_GRAMMAR_ON_AUTO", "1") != "1":
        return None
```

Kill switches, neither needing a rebuild: `VLLM_DSML_GRAMMAR_ON_AUTO=0` (this patch only) or
`VLLM_ENFORCE_STRICT_TOOL_CALLING=0` (upstream, disables all structural tags).

**Verify engagement structurally — do not infer it from a clean run.** xgrammar's matcher only
exists when a grammar was compiled, so its log lines are a direct on/off signal:

```
unpatched  : 0 grammar_matcher messages during live traffic
patched-r2 : 32
```

That is the actual proof the fix is doing something. The 80 clean samples are corroboration, not
evidence — 80 samples cannot distinguish 1% from 0%.

Expect one benign warning under speculative decoding:
`matcher has terminated after accepting the stop token, but is trying to accept new token` —
draft tokens arriving past the stop token, which the matcher correctly rejects.

## Honest status

0.27.1 + the grammar patch serves our production agent traffic. But be clear about what is and is
not established:

- **The root cause was never identified.** The grammar makes invalid *syntax* unsamplable; it does
  not explain why the model wanted to emit it. Remaining suspects: the upstream DSpark spec-decode
  reimplementation (none of the vendor overlay's tuning knobs exist upstream) and numerics on the
  fp8/MXFP4 path.
- **Grammar constrains syntax, not semantics.** If numerics are the underlying problem, you now get
  well-formed tool calls with wrong arguments — and every gate on this page would pass them.
- **We have one leak, not a distribution.** The 1% figure is a single event over ~140 samples. Treat
  it as an order of magnitude, not a measurement.

If you hit malformed DSML under agent traffic on 0.27.x, we would genuinely like to hear about it —
open an issue. A second data point is worth more to both of us than another week of our own
guessing.
