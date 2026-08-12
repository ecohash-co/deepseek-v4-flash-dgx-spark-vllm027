#!/usr/bin/env python3
"""Test the knife-edge-tie hypothesis for the batch-dependent degenerate output.

Hypothesis: the json60 prompt has two near-tied continuations at the first content
token ('[\\n{' -> the real array, vs '["' -> the literal-minded ["json array..."]).
Greedy decoding is NOT batch-invariant (matmul reduction order changes with batch
size), so a near-tie flips when batch size changes. That would make this a prompt
ambiguity exposed by numerics, NOT a DSpark/spec-decode correctness defect.

Prediction if true: top-2 logprobs on the first token are within a few thousandths
of a nat for json60, and far apart for count50 (which never flips).
"""
import json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("URL", "http://127.0.0.1:8888/v1/chat/completions")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")

PROMPTS = {
    "json60 (flips)": ('Output a JSON array of 60 objects, each exactly '
                       '{"id":N,"name":"user_N","active":true}. JSON only.'),
    "count50 (stable)": "Count from 1 to 50, one number per line. Numbers only, nothing else.",
}


def ask(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4, "temperature": 0.0,
            "chat_template_kwargs": {"thinking": False},
            "logprobs": True, "top_logprobs": 5}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    ch = r["choices"][0]
    lp = (ch.get("logprobs") or {}).get("content") or []
    return lp[0] if lp else None


def show(prompt, batch):
    with ThreadPoolExecutor(max_workers=batch) as ex:
        res = list(ex.map(lambda _: ask(prompt), range(batch)))
    first = res[0]
    if not first:
        print(f"    batch={batch}: no logprobs returned")
        return
    top = first.get("top_logprobs") or []
    chosen = first.get("token")
    parts = [f"{t['token']!r}:{t['logprob']:.5f}" for t in top[:3]]
    gap = (top[0]["logprob"] - top[1]["logprob"]) if len(top) > 1 else float("nan")
    print(f"    batch={batch}  chosen={chosen!r:12}  top2 gap={gap:.6f}   " + "  ".join(parts))


for name, p in PROMPTS.items():
    print(f"\n{name}")
    for b in (1, 2, 4):
        show(p, b)
