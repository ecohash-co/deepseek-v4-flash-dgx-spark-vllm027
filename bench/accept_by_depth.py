#!/usr/bin/env python3
"""Attribute the 128K decode cliff: speculative acceptance per depth.

Scrapes vLLM's spec_decode counters immediately before/after a SINGLE request at
each depth, so the delta is attributable to that request alone. If acceptance
collapses at 128K, the cliff is a draft-path problem; if acceptance is flat, the
cliff is elsewhere (step time) and the next suspect list changes.

Also reports decode tok/s from the same request, so acceptance and throughput are
measured on the identical call.
"""
import json, os, re, time, urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8888/v1")
METRICS = os.environ.get("METRICS", "http://127.0.0.1:8888/metrics")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")
NTOK = int(os.environ.get("NTOK", "200"))
DEPTHS = [int(x) for x in os.environ.get("DEPTHS", "8000,128000,512000").split(",")]

FILLER = ("The distributed scheduler interleaves prefill and decode within a single step. "
          "Long prompts dominate the step budget and delay in-flight decodes. "
          "Speculative drafts are verified by the target model before acceptance. ")

PATS = {
    "drafts": r'vllm:spec_decode_num_drafts_total\{[^}]*\}\s+([0-9.]+)',
    "draft_tokens": r'vllm:spec_decode_num_draft_tokens_total\{[^}]*\}\s+([0-9.]+)',
    "accepted": r'vllm:spec_decode_num_accepted_tokens_total\{[^}]*\}\s+([0-9.]+)',
}


def scrape():
    txt = urllib.request.urlopen(METRICS, timeout=30).read().decode()
    out = {}
    for k, p in PATS.items():
        m = re.search(p, txt)
        out[k] = float(m.group(1)) if m else 0.0
    pos = {}
    for m in re.finditer(r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"\}\s+([0-9.]+)', txt):
        pos[int(m.group(1))] = float(m.group(2))
    out["pos"] = pos
    return out


def stream_once(prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(),
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
                if (d.get("content") or d.get("reasoning_content") or d.get("reasoning")) and first is None:
                    first = time.perf_counter()
            if ev.get("usage"):
                usage = ev["usage"]
    end = time.perf_counter()
    out = (usage or {}).get("completion_tokens", 0)
    win = end - (first or end)
    return (first or end) - t0, win, out, (out / win if win > 0 else 0.0)


print(f"{'depth':>8} {'ttft_s':>8} {'tok/s':>7} {'drafts':>7} {'dtok':>7} {'acc':>7} {'acc/draft':>10} {'accept%':>8}  per-position accept%")
for d in DEPTHS:
    hay = (FILLER * max(1, d // 20))[: d * 4]
    prompt = hay + "\n\nSummarize the passage above in one short paragraph."
    b = scrape()
    ttft, win, out, rate = stream_once(prompt, NTOK)
    a = scrape()
    dd = a["drafts"] - b["drafts"]
    dt = a["draft_tokens"] - b["draft_tokens"]
    da = a["accepted"] - b["accepted"]
    per = []
    for i in sorted(set(list(a["pos"]) + list(b["pos"]))):
        dpi = a["pos"].get(i, 0) - b["pos"].get(i, 0)
        per.append(f"p{i}={100*dpi/dd:.0f}%" if dd else f"p{i}=-")
    print(f"{d:>8} {ttft:>8.2f} {rate:>7.2f} {dd:>7.0f} {dt:>7.0f} {da:>7.0f} "
          f"{(da/dd if dd else 0):>10.2f} {(100*da/dt if dt else 0):>7.1f}%  {' '.join(per)}")
