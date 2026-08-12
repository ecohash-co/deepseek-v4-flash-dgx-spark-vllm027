#!/usr/bin/env python3
"""Saturation test at --max-num-seqs 12 on the baked 0.27.1 image.

Why 12 specifically: at 11-12 in-flight sequences the DSpark speculative verify
pass exceeds 64 tokens (12 seqs x (1+5) drafts = 72 > 64), so attention routes to
the PREFILL orchestrator by design rather than the decode-dsv4 kernel. That is the
opposite side of the dispatch gate we patched, so it needs its own proof that it
does not assert. Killed twice in earlier sessions before completing.
"""
import json, os, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("URL", "http://127.0.0.1:8888/v1/chat/completions")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")
NTOK = int(os.environ.get("NTOK", "300"))

FILLER = ("The distributed scheduler interleaves prefill and decode within a single step. "
          "Long prompts dominate the step budget and delay in-flight decodes. ")


def one(i, depth):
    # unique suffix per request so they do not collapse onto one cached prefix
    prompt = (FILLER * max(1, depth // 20))[: depth * 4] + \
             f"\n\n(request {i}) Summarize the passage above in one short paragraph."
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": NTOK, "temperature": 1.0, "top_p": 0.95,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            p = line[6:]
            if p == "[DONE]":
                break
            ev = json.loads(p)
            if ev.get("choices"):
                d = ev["choices"][0].get("delta") or {}
                if (d.get("content") or d.get("reasoning_content")) and first is None:
                    first = time.perf_counter()
            if ev.get("usage"):
                usage = ev["usage"]
    end = time.perf_counter()
    out = (usage or {}).get("completion_tokens", 0)
    win = end - (first or end)
    return {"ttft": (first or end) - t0, "out": out,
            "tps": out / win if win > 0 else 0.0, "elapsed": end - t0}


for conc in (1, 4, 8, 12):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(lambda i: one(i, 2000), range(conc)))
    el = time.perf_counter() - t0
    tot = sum(r["out"] for r in res)
    print(f"concurrency={conc:>3}  wall={el:7.1f}s  total_out={tot:>6}  "
          f"aggregate={tot/el:7.2f} tok/s  median_per_stream={statistics.median(r['tps'] for r in res):6.2f} tok/s  "
          f"median_ttft={statistics.median(r['ttft'] for r in res):6.2f}s")
    sys.stdout.flush()
print("DONE")
