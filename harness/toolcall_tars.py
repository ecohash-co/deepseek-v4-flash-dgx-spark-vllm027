#!/usr/bin/env python3
"""Agent-shaped tool-calling gate. Run this BEFORE any flash stack change.

WHY THIS EXISTS
---------------
On 2026-08-11 vLLM 0.27.1 passed every gate we had -- needle recall at 377K tokens,
arithmetic, JSON mode, 12-way saturation, zero asserts -- and broke tool calling so badly
TARS "lost the plot instantly". The existing gates could not have caught it:

    toolcall.py            no temperature (server default), non-streaming, ONE tiny tool
    toolcall_multiturn.py  same, plus it reported a phantom issue-#21 on a spec-correct 400

TARS runs at temp 1.0 / top_p 0.95 (pinned 2026-08-10 as the repetition-loop mitigation),
streaming, with a fat tool set. MiaAI-Lab's docs/DSML_SYNTAX_TEMP_ASYMMETRY.md documents why
that combination matters: vLLM samples DSML *structural* tokens at the request temperature, so
at temp 1.0 the syntax itself can derail into malformed DSML. A greedy single-tool probe is
blind to it by construction.

WHAT IT CHECKS
--------------
The real failure signature, not just "did we get a tool call":
  1. structured tool_calls actually arrive (streaming, assembled from deltas)
  2. arguments parse as JSON and match the declared schema's required keys
  3. NO raw DSML leaks into content -- the 0.27 failure emitted invented tags like
     <|DSML|exec_command> and <|DSML|parameter name="exec"> as plain content
  4. it holds over n>=20 samples, because this failure is stochastic

CALIBRATION -- READ THIS BEFORE TRUSTING A GREEN RUN
----------------------------------------------------
Validate against the known-good 0.21 production stack FIRST. A gate that has never been shown
to pass a good stack proves nothing when it passes a new one. Expect ~100% on 0.21.

    python3 toolcall_tars.py                       # through LiteLLM (the path TARS uses)
    URL=http://192.168.1.12:8888/v1/chat/completions MODEL=deepseek-v4-flash-dspark \
        python3 toolcall_tars.py                   # direct to castor, bypassing LiteLLM

NOTE: LiteLLM's hosted_vllm transform calls _remove_strict_from_schema() and
_remove_additional_properties(), which RECURSIVELY DELETE those keys from every tool. So
`strict: true` cannot reach vLLM through LiteLLM -- relevant because 0.27 gates its DSML
grammar behind `tool_choice=="auto" and _any_tool_strict(tools)`. Direct-vs-LiteLLM runs are
therefore NOT equivalent on 0.27. They are equivalent on 0.21, which ignores strict entirely.
"""
import json, os, re, sys, urllib.request, urllib.error
from collections import Counter

URL = os.environ.get("URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")
KEY = os.environ.get("LITELLM_KEY", "EMPTY")
N = int(os.environ.get("N", "20"))
TEMP = float(os.environ.get("TEMP", "1.0"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
STRICT = os.environ.get("STRICT", "0") == "1"   # engages 0.27 DSML grammar (direct only)

# An agent-shaped toolset: several tools, nested/enum/array params -- not one toy function.
TOOLS = [
    {"type": "function", "function": {
        "name": "run_shell", "description": "Run a shell command on a node.",
        "parameters": {"type": "object", "properties": {
            "node": {"type": "string", "enum": ["node-a", "node-b", "node-c", "node-d"]},
            "command": {"type": "string"},
            "timeout_s": {"type": "integer"}},
            "required": ["node", "command"]}}},
    {"type": "function", "function": {
        "name": "query_logs", "description": "Query Loki for log lines.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "hours": {"type": "integer"},
            "limit": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_notes", "description": "Semantic search over a document corpus.",
        "parameters": {"type": "object", "properties": {
            "q": {"type": "string"},
            "collections": {"type": "array", "items": {"type": "string"}}},
            "required": ["q"]}}},
    {"type": "function", "function": {
        "name": "restart_service", "description": "Restart a docker service on a node.",
        "parameters": {"type": "object", "properties": {
            "node": {"type": "string"}, "service": {"type": "string"},
            "confirm": {"type": "boolean"}}, "required": ["node", "service"]}}},
    {"type": "function", "function": {
        "name": "get_metrics", "description": "Fetch a Prometheus metric.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string"}, "window": {"type": "string"}},
            "required": ["metric"]}}},
]
if STRICT:
    for t in TOOLS:
        t["function"]["strict"] = True

SYSTEM = ("You are an infrastructure operations agent. You have tools. When the user asks for "
          "something that requires data from the infrastructure, CALL A TOOL. Do not guess.")

PROMPTS = [
    ("run_shell", "Check how much disk space is left on node-a."),
    ("query_logs", "Did anything error in the last 2 hours? Check the logs."),
    ("search_notes", "What did we decide about the DGX Spark power settings?"),
    ("get_metrics", "What's the GPU temperature on node-b right now?"),
    ("restart_service", "Frigate is wedged on node-d, restart it."),
]

# The 0.27 failure signature: DSML markup surfacing as CONTENT rather than parsed tool_calls.
# The optional `/` is load-bearing: one observed signature was a run of stray CLOSING tags
# (`</｜DSML｜tool_calls>` x3). A first version of this regex omitted it and silently missed
# them -- caught only by the self-test below, which is why that self-test exists. Both the
# fullwidth ｜ (U+FF5C, what the checkpoint actually emits) and ASCII | are accepted.
DSML_LEAK = re.compile(r"[<\[]\s*/?\s*[|｜]\s*DSML", re.I)

# Known-bad strings from the 2026-08-11 incident, and known-good text that must NOT trip.
# Run `python3 toolcall_tars.py --self-test` before trusting a green result.
_MUST_DETECT = [
    '<｜DSML｜parameter name="exec">ls -la</｜DSML｜parameter>',
    '<｜DSML｜exec_command>docker ps</｜DSML｜exec_command>',
    '<｜DSML｜exec command="ls">',
    '</｜DSML｜tool_calls></｜DSML｜tool_calls></｜DSML｜tool_calls>',
    '<|DSML|tool_calls>',
]
_MUST_NOT = [
    "I'll check the disk space on node-a for you.",
    "Here's a shell snippet: `df -h | grep /dev`",
    "The DSML format is used internally.",
    "Use [|pipe|] notation in the docs.",
]

if "--self-test" in sys.argv:
    bad = [s for s in _MUST_DETECT if not DSML_LEAK.search(s)]
    fp = [s for s in _MUST_NOT if DSML_LEAK.search(s)]
    for s in bad:
        print(f"  MISS         {s[:70]}")
    for s in fp:
        print(f"  FALSE POS    {s[:70]}")
    if bad or fp:
        print(f"\n  ⛔ detector broken: {len(bad)} missed, {len(fp)} false positive(s)")
        sys.exit(1)
    print(f"  ✅ detector OK ({len(_MUST_DETECT)} detected, {len(_MUST_NOT)} correctly ignored)")
    sys.exit(0)


def once(idx):
    want, prompt = PROMPTS[idx % len(PROMPTS)]
    body = {"model": MODEL, "stream": True, "temperature": TEMP, "top_p": TOP_P,
            "max_tokens": 800, "tools": TOOLS, "tool_choice": "auto",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    content, calls = "", {}
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)["choices"][0]["delta"]
                except Exception:
                    continue
                content += d.get("content") or ""
                for tc in d.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["args"] += fn.get("arguments") or ""
    except urllib.error.HTTPError as e:
        return "http_error", f"{e.code} {e.read().decode(errors='replace')[:120]}"
    except Exception as e:
        return "failed", f"{type(e).__name__} {str(e)[:120]}"

    if DSML_LEAK.search(content):
        return "dsml_leak", content.strip().replace("\n", " ")[:150]
    if not calls:
        return "no_tool_call", content.strip().replace("\n", " ")[:150]
    for slot in calls.values():
        if not slot["name"]:
            return "unnamed_call", repr(slot["args"][:100])
        try:
            parsed = json.loads(slot["args"] or "{}")
        except json.JSONDecodeError:
            return "bad_json_args", f"{slot['name']}: {slot['args'][:110]!r}"
        if not isinstance(parsed, dict):
            return "bad_json_args", f"{slot['name']}: not an object -> {slot['args'][:90]!r}"
        spec = next((t["function"] for t in TOOLS if t["function"]["name"] == slot["name"]), None)
        if spec is None:
            return "unknown_tool", slot["name"]
        missing = [k for k in spec["parameters"].get("required", []) if k not in parsed]
        if missing:
            return "missing_required", f"{slot['name']} missing {missing}"
    return "ok", ",".join(s["name"] for s in calls.values())


print(f"url={URL}\nmodel={MODEL}  n={N}  temp={TEMP}  top_p={TOP_P}  stream=True  "
      f"tools={len(TOOLS)}  strict={STRICT}")
results, firsts = Counter(), {}
for i in range(N):
    verdict, detail = once(i)
    results[verdict] += 1
    firsts.setdefault(verdict, detail)
    sys.stdout.write("." if verdict == "ok" else "X")
    sys.stdout.flush()
print("\n")
for verdict, count in results.most_common():
    print(f"  {verdict:17s} {count:3d}/{N}   e.g. {firsts[verdict][:110]!r}")

ok = results["ok"]
rate = 100.0 * ok / N
print(f"\n  PASS RATE {ok}/{N} = {rate:.0f}%")
# 0.21 baseline is ~100%. Anything that drops well below it is a regression, and a single
# dsml_leak is disqualifying on its own -- that is the exact 08-11 production failure.
if results["dsml_leak"]:
    print("  ⛔ FAIL — raw DSML leaked into content. This IS the 2026-08-11 signature.")
    sys.exit(2)
if rate < 95:
    print("  ⛔ FAIL — below the 95% floor; do not promote this stack.")
    sys.exit(1)
print("  ✅ PASS")
