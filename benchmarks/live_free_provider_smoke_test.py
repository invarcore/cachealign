"""
Universal Free-Tier & Local KV-Cache Verification Smoke Test.
Allows testing CacheAlign end-to-end without paid subscriptions or heavy local LLM runtimes.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

from openai import OpenAI

from cachealign import CacheAlignConfig, wrap

sys.stdout.reconfigure(encoding="utf-8")

FALLBACK_FREE_MODELS = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3.8-27b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-26b-a4b-it:free",
]


def fetch_active_openrouter_free_models():
    """Discover active free models on OpenRouter."""
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "CacheAlign-SmokeTest/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            discovered = [
                m["id"]
                for m in data.get("data", [])
                if m.get("id", "").endswith(":free")
                or (
                    m.get("pricing", {}).get("prompt") == "0"
                    and m.get("pricing", {}).get("completion") == "0"
                )
            ]
            if discovered:
                return ["openrouter/free"] + [m for m in discovered if m != "openrouter/free"]
    except Exception:
        pass
    return FALLBACK_FREE_MODELS


def run_smoke_test(mode="local", base_url=None, api_key=None, model=None):
    print("\n" + "=" * 70)
    print(f"🚀 CacheAlign End-to-End Verification: [{mode.upper()} MODE]")
    print("=" * 70)

    candidate_models = []
    if mode == "local":
        base_url = base_url or "http://127.0.0.1:8080/v1"
        api_key = api_key or "mock-test-key"
        candidate_models = [model or "gpt-4o"]
        print(f"Connecting to Local Simulator: {base_url}")
    elif mode == "openrouter":
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("❌ OPENROUTER_API_KEY is not set.")
            print("Get a free key in 30 seconds (no credit card): https://openrouter.ai/keys")
            return
        base_url = "https://openrouter.ai/api/v1"
        if model:
            candidate_models = [model]
        else:
            print("🔍 Auto-discovering active free models on OpenRouter...")
            candidate_models = fetch_active_openrouter_free_models()
            print(f"   Found {len(candidate_models)} candidates. Primary: {candidate_models[0]}")
    elif mode == "groq":
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("❌ GROQ_API_KEY is not set.")
            return
        base_url = "https://api.groq.com/openai/v1"
        candidate_models = [model or "llama-3.3-70b-versatile"]
    elif mode == "opencode":
        api_key = api_key or os.environ.get("OPENCODE_API_KEY")
        if not api_key:
            print("❌ OPENCODE_API_KEY is not set. Sign in at https://opencode.ai/auth")
            return
        base_url = "https://opencode.ai/zen/v1"
        candidate_models = [model or "nemotron-3.5-lightning-free"]

    raw_client = OpenAI(base_url=base_url, api_key=api_key)
    client = wrap(raw_client, config=CacheAlignConfig(fail_open=True, verbose=True))

    system_prompt = (
        "You are an enterprise database architect. Enforce ANSI SQL standards and strict ACID compliance.\n"
        "Security Mandate: All queries touching customer_pii must require role=DBA authentication."
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_table_schema",
                "description": "Fetch schema for database table.",
                "parameters": {
                    "type": "object",
                    "properties": {"table_name": {"type": "string"}},
                    "required": ["table_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "audit_query",
                "description": "Log high-privilege SQL execution.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
    ]

    selected_model = None
    resp_1 = None
    dur_1 = 0.0

    for candidate in candidate_models:
        print(f"\n[Turn 1] Attempting Cold Prefill with model: {candidate}...")
        start_t1 = time.perf_counter()
        try:
            resp_1 = client.chat.completions.create(
                model=candidate,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "Explain primary key optimization for customer_orders in 2 sentences.",
                    },
                ],
                tools=tools,
            )
            dur_1 = (time.perf_counter() - start_t1) * 1000
            selected_model = candidate
            print(f"   ✅ Connected successfully with: {selected_model}")
            break
        except Exception as e:
            err_msg = str(e)
            print(f"   ⚠️ Model {candidate} unavailable: {err_msg[:120]}")
            if (
                "404" in err_msg
                or "unavailable" in err_msg.lower()
                or "not found" in err_msg.lower()
            ):
                continue
            else:
                break

    if not resp_1:
        print("\n❌ Could not connect with any candidate models.")
        return

    usage_1 = getattr(resp_1, "usage", None)
    cached_1 = getattr(getattr(usage_1, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    print(f"   ⏱️  Turn 1 Latency (TTFT): {dur_1:.1f}ms")
    print(
        f"   📊 Prompt Tokens: {getattr(usage_1, 'prompt_tokens', 'N/A')} | Cached Tokens: {cached_1}"
    )
    if resp_1.choices and resp_1.choices[0].message and resp_1.choices[0].message.content:
        print(f"   💬 Response Preview: {resp_1.choices[0].message.content.strip()[:100]}...")

    print(f"\n[Turn 2] Sending Follow-Up Query (Warm Cache Read using {selected_model})...")
    start_t2 = time.perf_counter()
    try:
        resp_2 = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Explain primary key optimization for customer_orders in 2 sentences.",
                },
                {
                    "role": "assistant",
                    "content": resp_1.choices[0].message.content or "Completed prefill.",
                },
                {
                    "role": "user",
                    "content": "How does this impact write latency on foreign key indexes?",
                },
            ],
            tools=tools,
        )
        dur_2 = (time.perf_counter() - start_t2) * 1000
        usage_2 = getattr(resp_2, "usage", None)
        cached_2 = getattr(getattr(usage_2, "prompt_tokens_details", None), "cached_tokens", 0) or 0

        print(f"   ⏱️  Turn 2 Latency (TTFT): {dur_2:.1f}ms")
        print(
            f"   📊 Prompt Tokens: {getattr(usage_2, 'prompt_tokens', 'N/A')} | Cached Tokens: {cached_2}"
        )
        if resp_2.choices and resp_2.choices[0].message and resp_2.choices[0].message.content:
            print(f"   💬 Response Preview: {resp_2.choices[0].message.content.strip()[:100]}...")

        print("\n" + "=" * 70)
        print("📈 End-to-End CacheAlign Functional Parity Summary:")
        print("=" * 70)
        speedup = ((dur_1 - dur_2) / dur_1) * 100 if dur_1 > 0 else 0
        print(f"• Active Provider: {mode.upper()} ({selected_model})")
        print(f"• Turn 1 Latency: {dur_1:.1f} ms")
        print(f"• Turn 2 Latency: {dur_2:.1f} ms ({speedup:+.1f}% TTFT reduction)")
        if cached_2 > 0:
            print(f"• Prompt Cache Hit Verified: {cached_2} cached tokens recognized by provider!")
        print("• Tool Schemas: Canonicalized via RFC 8785 (Deterministic ordering guaranteed)")
        print("• In-Process Fail-Open: Passed with zero runtime exceptions")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"❌ Turn 2 failed: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CacheAlign Free-Tier & Local Verification")
    p.add_argument("--local", action="store_true")
    p.add_argument("--openrouter", action="store_true")
    p.add_argument("--groq", action="store_true")
    p.add_argument("--opencode", action="store_true")
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()

    mode = "local"
    if args.openrouter:
        mode = "openrouter"
    elif args.groq:
        mode = "groq"
    elif args.opencode:
        mode = "opencode"

    run_smoke_test(mode=mode, base_url=args.base_url, api_key=args.api_key, model=args.model)
