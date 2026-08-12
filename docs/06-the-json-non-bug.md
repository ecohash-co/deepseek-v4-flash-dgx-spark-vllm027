# A "model bug" that was neither a model bug nor a stack bug

Included because the *method* generalizes, and because we nearly patched an inference stack that
was behaving correctly.

## The symptom

Prompt:

```
Output a JSON array of 60 objects, each exactly
{"id":N,"name":"user_N","active":true}. JSON only.
```

Sometimes the model returns the array. Sometimes it returns, in full:

```json
["json array of 60 objects"]
```

9 tokens, `finish_reason: stop`. This had been tracked for weeks as a suspected
speculative-decoding defect ("DSpark spec-decode appears not verification-exact").

## What it actually is

### It is batch-size dependent

Same request, same `temperature: 0`, same server, back to back:

| batch size | degenerate |
|---|---|
| **1** | **10/10** |
| 2 | 0/8 |
| 3 | 0/8 |
| 4 | 0/10 |

The threshold is exactly 1→2.

### It is prompt-specific

Of three prompts run at every batch size, only this one flips. "Count from 1 to 50" and "write 8
sentences about a lighthouse keeper" are 0/8 at every batch size.

### The output is not corruption

`["json array of 60 objects"]` is a **coherent, literal-minded parse** of "output a JSON array of
60 objects" — an array containing that description. It is a defensible reading of an ambiguous
instruction, not garbage.

### The first token is genuinely contested

Requesting `logprobs` on the first token:

| prompt | top-2 gap | tokens in contention |
|---|---|---|
| the failing one | **0.25 – 1.75 nats** | `'[\n'` (real array) vs `'["'` (literal reading) |
| a stable one | **13.5 nats** | `'1'` vs `'2'` |

A ~1-nat gap is a real fork in the model's distribution, and which branch wins moves with
execution shape.

### Prefix caching is not the cause

With a unique suffix per request (so every request is a **cold** prefill), the failure still
occurs — 4/6 at batch 1 — and the completion lengths scatter wildly: 5, 16, 242, 903, 967 tokens.

The eerie part of the original observation — every failure being *exactly* 9 tokens, every success
*exactly* 903 — appears only when all requests share one cached prefix. **That determinism was a
caching artifact, not evidence of a deterministic bug.**

## Conclusion

An ambiguous instruction sitting on a contested continuation, where execution shape decides the
winner. Not DSpark. Not speculative decoding. Not the vLLM version.

**Nothing to patch.** For our deployment the mitigation was already in place: with the model's
thinking mode **on**, the rate is 0/40 across the 2×2 and 0/8 at the failing batch size, because
the reasoning tokens move the context off the fork before the content branch is reached.

## Why the rates looked like a version regression

Reported failure rates for the identical cell:

| how it was measured | rate |
|---|---|
| serially, on a loaded box | 13/20 |
| serially, on an idle box | 20/20 |
| at concurrency 4 | 0/10 |

Comparing a loaded measurement of the old version against an idle measurement of the new one made
it look like the upgrade had doubled the failure rate. It had not. **State the concurrency of any
rate you publish.**

## The generalizable method

Before blaming an inference stack for a bad output:

1. **Vary batch size** (1, 2, 4). A failure that vanishes at higher concurrency is not a stack bug.
2. **Try other prompts** at the same batch sizes. Prompt-specific ⇒ not a decode fault.
3. **Look at first-token logprobs.** `logprobs: true, top_logprobs: 5, max_tokens: 4`. A top-2 gap
   around a nat means the model is genuinely torn.
4. **Cache-bust** with a unique suffix. If it survives cold prefills, caching is not involved.
5. Only then suspect the stack.

Harnesses: `bench/serial_json.py`, `bench/batch_threshold.py`, `bench/logprob_tie.py`,
`bench/cachebust.py`, `bench/json_rate_2x2.py`.
