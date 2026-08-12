#!/usr/bin/env python3
"""Enable vLLM 0.27's native DSML structural-tag grammar for tool_choice="auto".

Usage:  python3 patch-dsml-grammar.py <structural_tag_registry.py>

WHY
---
vLLM 0.27.1 ships a complete DSML xgrammar structural tag for `deepseek_v4` — it is in
XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS and it works. It is gated off for exactly the case that
needs it:

    if tool_choice == "auto" and not _any_tool_strict(tools):
        return None

Agents send `tool_choice="auto"` and rarely mark tools `strict`, so the grammar never engages for
them. That matters because vLLM samples DSML *structural* tokens at the request temperature: the
tags framing a tool call are drawn from the same distribution as prose. We run temp 1.0 / top_p
0.95 deliberately (it is the mitigation for a separate repetition-loop problem), which puts agent
traffic exactly where structural sampling is most fragile.

Observed 2026-08-11 in production: the model emitted invented grammar — `<｜DSML｜exec_command>`,
parameter tags used as invokes, stray repeated closers — and the agent, receiving prose where it
expected a callable tool, failed instantly. Reproduced 2026-08-12 at roughly 1% per tool call.
That rate is invisible to benchmarks and brutal for agents: at 1%/call, a 50-call session has a
~40% chance of at least one malformed call.

The grammar makes illegal DSML tokens *unsamplable* at any temperature, and constrains arguments
to each tool's JSON schema. It is a structural fix, not a probabilistic one.

WHY NOT DO THIS CLIENT-SIDE
---------------------------
You cannot, if LiteLLM is in front of vLLM. Its `hosted_vllm` transform calls
`_remove_strict_from_schema()`, which recursively deletes every `strict` key from every tool
(and `_remove_additional_properties()` with it), so `strict: true` never reaches the server.
Verified in litellm/llms/hosted_vllm/chat/transformation.py. Only `hosted_vllm`,
`vertex_ai/gemini` and `watsonx` do this; the `openai/` provider passes it through.

BEHAVIOR
--------
Default ON. Two independent kill switches, neither requiring a rebuild:
    VLLM_DSML_GRAMMAR_ON_AUTO=0          this patch only; restores stock upstream behavior
    VLLM_ENFORCE_STRICT_TOOL_CALLING=0   upstream's own switch; disables ALL structural tags

Fails loudly if the anchor is missing or ambiguous, rather than silently shipping stock upstream.
"""
import sys

ANCHOR = '''    if tool_choice == "auto" and not _any_tool_strict(tools):
        return None
'''

REPLACEMENT = '''    if tool_choice == "auto" and not _any_tool_strict(tools):
        # PATCH(dsml-grammar-on-auto): upstream returns None here, disabling the DSML
        # structural-tag grammar for tool_choice="auto" unless some tool sets strict=true.
        # That is the case agents use, and it is where the grammar matters most --
        # DSML structural tokens are sampled at the request temperature, so at temp 1.0
        # the syntax itself can derail into invalid markup. Clients cannot opt in through
        # LiteLLM: its hosted_vllm transform strips `strict` from every tool.
        # Default ON; set VLLM_DSML_GRAMMAR_ON_AUTO=0 to restore upstream behavior, or
        # VLLM_ENFORCE_STRICT_TOOL_CALLING=0 to disable all structural tags.
        import os as _os

        if _os.environ.get("VLLM_DSML_GRAMMAR_ON_AUTO", "1") != "1":
            return None
'''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    src = open(path).read()

    if "PATCH(dsml-grammar-on-auto)" in src:
        print(f"already patched: {path}")
        return 0

    n = src.count(ANCHOR)
    if n != 1:
        print(f"FATAL: anchor found {n} times in {path} (expected exactly 1).")
        print("Upstream changed this gate. Re-read get_model_structural_tag() before forcing it.")
        return 1

    open(path, "w").write(src.replace(ANCHOR, REPLACEMENT, 1))
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
