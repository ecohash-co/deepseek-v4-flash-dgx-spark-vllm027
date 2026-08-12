#!/usr/bin/env python3
"""Is the degenerate ["json"] real at our CURRENT sampling settings?

2026-07-31 measured 4/20 (thinking off) vs 0/20 (thinking on) -- but at temperature 0,
and Fisher two-tailed p=0.106 (not significant). Since then we pin temp 1.0 / top_p 0.95
at LiteLLM because the loop root-cause is temperature-gated. So the real question is a 2x2:

    thinking {off,on} x sampling {temp 0, temp 1.0/top_p 0.95}

If the degenerate output vanishes at temp 1.0 regardless of thinking, then it was another
symptom of temp-0 chaos, and thinking-on is not earning its keep for THIS bug.
"""
import json, os, urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8888/v1")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")
N = int(os.environ.get("N", "20"))
P = ('Output a JSON array of 60 objects, each exactly '
     '{"id":N,"name":"user_N","active":true}. JSON only.')


def ask(think, temp, top_p):
    body = {"model": MODEL, "messages": [{"role": "user", "content": P}],
            "max_tokens": 2000, "temperature": temp}
    if top_p is not None:
        body["top_p"] = top_p
    body["chat_template_kwargs"] = {"thinking": bool(think)}
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    m = r["choices"][0]["message"]
    return (m.get("content") or ""), r["usage"]["completion_tokens"]


print(f"n={N} per cell; degenerate := completion_tokens < 50 or content.strip() in low-content set")
print(f"{'thinking':>9} {'sampling':>22} {'degenerate':>12} {'tok min':>8} {'tok max':>8}  seq")
rows = []
for think in (False, True):
    for label, temp, top_p in (("temp 0", 0.0, None), ("temp 1.0/top_p 0.95", 1.0, 0.95)):
        bad, toks, seq = 0, [], []
        for _ in range(N):
            try:
                text, ct = ask(think, temp, top_p)
                stripped = text.strip().lower().replace(" ", "")
                degenerate = ct < 50 or stripped in ('["json"]', '["json"]\n', 'json')
                toks.append(ct)
                bad += degenerate
                seq.append("X" if degenerate else ".")
            except Exception as exc:
                seq.append("E")
                print(f"    error: {exc}")
        pct = 100.0 * bad / max(1, N)
        print(f"{str(think):>9} {label:>22} {bad:>6}/{N} ({pct:4.0f}%) "
              f"{min(toks) if toks else 0:>8} {max(toks) if toks else 0:>8}  {''.join(seq)}")
        rows.append((think, label, bad, N))

print()
for think, label, bad, n in rows:
    print(f"  thinking={think!s:<5} {label:<22} -> {bad}/{n}")
