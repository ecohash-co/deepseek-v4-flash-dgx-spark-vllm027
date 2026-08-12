#!/usr/bin/env python3
"""Separate prefix-caching from batch numerics as the cause of the flip.

Every earlier run reused an identical prompt, so requests after the first hit the
prefix cache. If the batch dependence survives cache-busting (unique prompt per
request, so every one is a cold full prefill), the cause is batch numerics.
If it vanishes, the cause was cached-vs-uncached prefill.
"""
import json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("URL", "http://127.0.0.1:8888/v1/chat/completions")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")
BASE = ('Output a JSON array of 60 objects, each exactly '
        '{"id":N,"name":"user_N","active":true}. JSON only.')


def ask(i, bust):
    # A unique trailing token forces a cold prefill without changing the task.
    p = f"{BASE} (request {i})" if bust else BASE
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}],
            "max_tokens": 2000, "temperature": 0.0,
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["usage"]["completion_tokens"]


N = 6
print(f"{'cache':>10} {'batch':>6} {'degen':>7}  tokens")
for bust in (True, False):
    for batch in (1, 4):
        with ThreadPoolExecutor(max_workers=batch) as ex:
            toks = list(ex.map(lambda i: ask(i, bust), range(1000 if bust else 0,
                                                             (1000 if bust else 0) + N)))
        bad = sum(1 for t in toks if t < 50)
        label = "BUSTED" if bust else "shared"
        print(f"{label:>10} {batch:>6} {bad:>4}/{N}  {toks}")
