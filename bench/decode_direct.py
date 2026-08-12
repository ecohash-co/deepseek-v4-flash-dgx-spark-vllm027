#!/usr/bin/env python3
"""Direct streaming decode measurement — no prefix-cache differencing.

benchmarks/longctx.py derives decode as (warm_N - warm_1), which assumes a stable
cached prefill. On vLLM 0.27.1 the warm prefill is erratic (0.38s..17.18s for the
same call), so that derivation produces artifacts (incl. 1.5e8 tok/s). This measures
decode the unambiguous way: stream, and divide emitted tokens by the wall time
AFTER the first token.

Sampling matches production (temp 1.0 / top_p 0.95 pinned at LiteLLM) and caps
max_tokens so an occasional repetition-loop can't hang the run.
"""
import json, os, time, urllib.request, statistics

URL = os.environ.get("URL", "http://127.0.0.1:8888/v1")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")
NTOK = int(os.environ.get("NTOK", "200"))
REPS = int(os.environ.get("REPS", "3"))
DEPTHS = [int(x) for x in os.environ.get("DEPTHS", "8000,128000,512000").split(",")]

FILLER = ("The distributed scheduler interleaves prefill and decode within a single step. "
          "Long prompts dominate the step budget and delay in-flight decodes. "
          "Speculative drafts are verified by the target model before acceptance. ")


def stream_once(prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("choices"):
                d = ev["choices"][0].get("delta") or {}
                if (d.get("content") or d.get("reasoning_content") or d.get("reasoning")) and first is None:
                    first = time.perf_counter()
            if ev.get("usage"):
                usage = ev["usage"]
    end = time.perf_counter()
    out = (usage or {}).get("completion_tokens", 0)
    ttft = (first or end) - t0
    win = end - (first or end)
    return ttft, win, out, (out / win if win > 0 else 0.0)


print(f"{'depth':>8} {'prompt_tok':>10} {'rep':>4} {'ttft_s':>9} {'decode_win_s':>13} {'out_tok':>8} {'DECODE t/s':>11}")
for d in DEPTHS:
    hay = (FILLER * max(1, d // 20))[: d * 4]
    prompt = hay + "\n\nSummarize the passage above in one short paragraph."
    rates = []
    for rep in range(1, REPS + 1):
        ttft, win, out, rate = stream_once(prompt, NTOK)
        rates.append(rate)
        print(f"{d:>8} {'':>10} {rep:>4} {ttft:>9.2f} {win:>13.2f} {out:>8} {rate:>11.2f}")
    print(f"{d:>8} {'':>10} {'mean':>4} {'':>9} {'':>13} {'':>8} {statistics.mean(rates):>11.2f}"
          + (f"  (sd {statistics.stdev(rates):.2f})" if len(rates) > 1 else ""))
