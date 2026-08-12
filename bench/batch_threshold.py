#!/usr/bin/env python3
"""Characterise the batch-size-dependent degenerate-output bug on 0.27.1/DSpark.

Established: thinking=False, temp 0, this prompt -> 10/10 degenerate at batch 1,
0/10 at batch 4. Questions this answers:
  1. Where is the threshold (batch 1/2/3/4)?
  2. Is it prompt-specific, or does any prompt truncate at batch 1?
  3. Does thinking=True mask it at batch 1 (production config)?
"""
import json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("URL", "http://127.0.0.1:8888/v1/chat/completions")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")

PROMPTS = {
    "json60": ('Output a JSON array of 60 objects, each exactly '
               '{"id":N,"name":"user_N","active":true}. JSON only.'),
    "count50": "Count from 1 to 50, one number per line. Numbers only, nothing else.",
    "story": "Write exactly 8 sentences about a lighthouse keeper who collects stamps.",
}


def ask(prompt, think, temp):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000, "temperature": temp,
            "chat_template_kwargs": {"thinking": think}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["usage"]["completion_tokens"], (r["choices"][0]["message"].get("content") or "")


def run(prompt, think, temp, batch, n):
    with ThreadPoolExecutor(max_workers=batch) as ex:
        res = list(ex.map(lambda _: ask(prompt, think, temp), range(n)))
    toks = [t for t, _ in res]
    bad = sum(1 for t in toks if t < 50)
    return bad, min(toks), max(toks), res[0][1][:46]


N = 8
print(f"n={N} per row; 'degen' := completion_tokens < 50\n")
print(f"{'prompt':>9} {'think':>6} {'temp':>5} {'batch':>6} {'degen':>7} {'tok min':>8} {'tok max':>8}  sample")
for name, p in PROMPTS.items():
    for batch in (1, 2, 3, 4):
        bad, lo, hi, s = run(p, False, 0.0, batch, N)
        print(f"{name:>9} {'off':>6} {0.0:>5} {batch:>6} {bad:>4}/{N} {lo:>8} {hi:>8}  {s!r}")
    print()

print("--- production config (thinking on) at the failing batch size ---")
for name, p in PROMPTS.items():
    bad, lo, hi, s = run(p, True, 0.0, 1, N)
    print(f"{name:>9} {'ON':>6} {0.0:>5} {1:>6} {bad:>4}/{N} {lo:>8} {hi:>8}  {s!r}")
