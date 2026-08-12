#!/usr/bin/env python3
"""Is the degenerate JSON output load-dependent?

The 2x2 probe scored thinking=False/temp0 at 20/20 degenerate running SERIALLY.
A single identical request issued while OTHER traffic was in flight produced a
clean 903-token answer. If that reproduces, the failure is batch//concurrency
dependent -- which would make it a spec-decode verify-path suspect, not a
sampling-temperature artifact.
"""
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("URL", "http://127.0.0.1:8888/v1/chat/completions")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")
P = ('Output a JSON array of 60 objects, each exactly '
     '{"id":N,"name":"user_N","active":true}. JSON only.')


def ask(_=None, think=False, temp=0.0):
    body = {"model": MODEL, "messages": [{"role": "user", "content": P}],
            "max_tokens": 2000, "temperature": temp,
            "chat_template_kwargs": {"thinking": think}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["usage"]["completion_tokens"], (r["choices"][0]["message"].get("content") or "")


N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
print(f"A) SERIAL (batch size 1), thinking=False temp=0, n={N}")
bad = 0
for i in range(N):
    ct, c = ask()
    d = ct < 50
    bad += d
    print(f"  {i+1:2d}  tokens={ct:5d}  {'DEGENERATE' if d else 'ok'}  {c[:50]!r}")
print(f"  -> {bad}/{N} degenerate\n")

print(f"B) CONCURRENT (4 at once), same request, n={N}")
bad2 = 0
with ThreadPoolExecutor(max_workers=4) as ex:
    for i, (ct, c) in enumerate(ex.map(ask, range(N))):
        d = ct < 50
        bad2 += d
        print(f"  {i+1:2d}  tokens={ct:5d}  {'DEGENERATE' if d else 'ok'}  {c[:50]!r}")
print(f"  -> {bad2}/{N} degenerate\n")
print(f"SUMMARY  serial {bad}/{N}   concurrent(4) {bad2}/{N}")
