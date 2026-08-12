#!/usr/bin/env python3
"""Needle-in-haystack correctness probe at context depth.

A wrong KV layout (e.g. routing nvfp4_ds_mla through the fp8 kernel path when the
layouts are NOT actually identical) corrupts long-context reads specifically.
Short prompts would still look fine, so this is the test that matters.
"""
import json, os, sys, time, urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8888/v1")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")
DEPTHS = [int(x) for x in os.environ.get("DEPTHS", "8000,128000,512000").split(",")]

FILLER = ("The distributed scheduler interleaves prefill and decode within a single step. "
          "Long prompts dominate the step budget and delay in-flight decodes. "
          "Speculative drafts are verified by the target model before acceptance. ")

# Distinctive, unguessable facts placed mid-document.
NEEDLE = ("REMEMBER THIS: the calibration constant for the Wichita array is 74915, "
          "the technician on duty was Marguerite Delacroix, and the sealed crate "
          "was labelled ORANGE-VECTOR-12. ")
CHECKS = [("74915", "constant"), ("delacroix", "technician"), ("orange-vector-12", "crate label")]


def ask(prompt, max_tokens=600):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95}
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r["choices"][0]["message"].get("content") or "", r["usage"], time.time() - t0


failures = 0
for d in DEPTHS:
    hay = (FILLER * max(1, d // 20))[: d * 4]
    mid = len(hay) // 2
    doc = hay[:mid] + NEEDLE + hay[mid:]
    prompt = (doc + "\n\nQuestion: In the passage above, what was the calibration constant, "
              "who was the technician on duty, and what was the crate labelled? "
              "Answer in one short sentence.")
    text, usage, dt = ask(prompt)
    low = text.lower()
    hits = [(tok in low) for tok, _ in CHECKS]
    ok = all(hits)
    if not ok:
        failures += 1
    miss = ", ".join(name for (tok, name), h in zip(CHECKS, hits) if not h) or "-"
    print(f"depth={d:>7} prompt_tok={usage['prompt_tokens']:>7} {dt:>7.1f}s "
          f"{'PASS' if ok else 'FAIL'} missing=[{miss}]")
    print(f"    answer: {text.strip()[:200]!r}")

print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
sys.exit(1 if failures else 0)
