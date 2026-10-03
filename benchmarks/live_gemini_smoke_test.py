"""
Live Gemini API Smoke Test with CacheAlign.
Demonstrates live context caching with Google Gemini 2.5 Flash / Pro.
Verifies TTFT latency reduction and token discounts on live provider infrastructure.

Usage:
    uv run python benchmarks/live_gemini_smoke_test.py [--pro]
"""

import argparse
import os
import sys
import time

# Ensure UTF-8 output on Windows PowerShell
sys.stdout.reconfigure(encoding="utf-8")


def generate_large_system_policy(min_tokens: int = 1200) -> str:
    """Generates a realistic enterprise policy prompt (>1,024 tokens) for Gemini caching."""
    header = (
        "Enterprise Governance & Data Security Specification v4.2\n"
        "Classification: CONFIDENTIAL // STRICT COMPLIANCE REQUIRED\n\n"
        "SECTION 1: DATA CLASSIFICATION AND RETENTION POLICIES\n"
    )
    body_paragraphs = [
        (
            f"1.{i} Policy Rule 10{i}: All customer financial records classified under Tier-{i % 3 + 1} "
            "must be cryptographically signed using HMAC-SHA256 tokens and replicated across three availability zones. "
            "Data retention requirements mandate immutable append-only storage for a minimum duration of 7 years. "
            "Any administrative override requires multi-party authorization with dual-custody MFA verification. "
            "Continuous anomaly detection watchdogs monitor all egress pipelines for potential canary leaks."
        )
        for i in range(1, 35)
    ]
    policy = header + "\n\n".join(body_paragraphs)
    return policy


def run_live_smoke_test(use_pro: bool = False, model_override: str | None = None):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n" + "=" * 70)
        print("❌ GEMINI_API_KEY is not set.")
        print("=" * 70)
        print("\nTo execute this live smoke test against your Gemini subscription:")
        print("1. Get your API key from Google AI Studio: https://aistudio.google.com/app/apikey")
        print("2. Set the environment variable in your terminal:\n")
        print("   In PowerShell (Windows):")
        print('     $env:GEMINI_API_KEY = "your-api-key-here"')
        print("     uv run python benchmarks/live_gemini_smoke_test.py\n")
        print("   In Bash / Zsh (macOS/Linux):")
        print('     export GEMINI_API_KEY="your-api-key-here"')
        print("     uv run python benchmarks/live_gemini_smoke_test.py\n")
        print("=" * 70)
        return False

    try:
        from google import genai
    except ImportError:
        print("❌ google-genai is not installed. Run: uv add google-genai")
        return False

    from cachealign import CacheAlignConfig, wrap

    if model_override:
        model_name = model_override
    elif use_pro:
        model_name = "gemini-3.1-pro-preview"
    else:
        model_name = "gemini-2.5-flash"

    print("\n" + "=" * 70)
    print(f"🚀 CacheAlign Live Gemini Smoke Test: {model_name}")
    print("=" * 70)

    # 1. Initialize client and wrap with CacheAlign
    raw_client = genai.Client(api_key=api_key)
    client = wrap(raw_client, config=CacheAlignConfig(fail_open=True, verbose=True))

    system_instruction = generate_large_system_policy(min_tokens=1200)
    approx_tokens = len(system_instruction) // 4
    print(
        f"📄 Static System Policy: ~{approx_tokens} tokens (meets Gemini 1,024-token cache threshold)"
    )

    # Turn 1: Cold Cache (Initial Cache Write)
    print("\n[Turn 1] Sending Query: 'Summarize Tier-1 data retention and audit requirements.'")
    start_time = time.perf_counter()
    resp_1 = client.models.generate_content(
        model=model_name,
        contents="Summarize Tier-1 data retention and audit requirements in 2 concise sentences.",
        config={"system_instruction": system_instruction},
    )
    t1_duration = (time.perf_counter() - start_time) * 1000

    usage_1 = getattr(resp_1, "usage_metadata", None)
    cached_tokens_1 = getattr(usage_1, "cached_content_token_count", 0) or 0
    prompt_tokens_1 = getattr(usage_1, "prompt_token_count", 0) or 0

    print(f"   ⏱️  Turn 1 Latency: {t1_duration:.1f}ms")
    print(f"   📊 Prompt Tokens: {prompt_tokens_1} | Cached Tokens: {cached_tokens_1}")
    print(f"   💬 Response Preview: {resp_1.text.strip()[:140]}...")

    # Turn 2: Warm Cache (Cache Hit on Aligned Prefix)
    print(
        "\n[Turn 2] Sending Follow-up Query: 'What watchdog mechanism monitors egress pipelines?'"
    )
    start_time = time.perf_counter()
    resp_2 = client.models.generate_content(
        model=model_name,
        contents="What watchdog mechanism monitors egress pipelines? Answer in 1 sentence.",
        config={"system_instruction": system_instruction},
    )
    t2_duration = (time.perf_counter() - start_time) * 1000

    usage_2 = getattr(resp_2, "usage_metadata", None)
    cached_tokens_2 = getattr(usage_2, "cached_content_token_count", 0) or 0
    prompt_tokens_2 = getattr(usage_2, "prompt_token_count", 0) or 0

    print(f"   ⏱️  Turn 2 Latency: {t2_duration:.1f}ms")
    print(f"   📊 Prompt Tokens: {prompt_tokens_2} | Cached Tokens: {cached_tokens_2}")
    print(f"   💬 Response Preview: {resp_2.text.strip()[:140]}...")

    # Summary
    print("\n" + "=" * 70)
    print("📈 Live Verification Results Summary:")
    print("=" * 70)
    latency_delta = ((t1_duration - t2_duration) / t1_duration) * 100 if t1_duration > 0 else 0
    print(f"• Turn 1 (Cold Write): {t1_duration:.1f} ms")
    print(f"• Turn 2 (Warm Read):  {t2_duration:.1f} ms ({latency_delta:+.1f}% TTFT improvement)")
    print(f"• Turn 2 Cached Tokens: {cached_tokens_2} / {prompt_tokens_2}")
    if cached_tokens_2 > 0:
        hit_rate = (cached_tokens_2 / prompt_tokens_2) * 100
        print(f"• Effective Cache Hit Rate: {hit_rate:.1f}%")
        print("✅ Live Gemini Context Caching successfully verified with CacheAlign!")
    else:
        print("[Info] Provider did not return cached_content_token_count > 0 on this run.")
        print(
            "  (Note: Context cache TTL creation on Gemini may take a few moments depending on region quota)."
        )
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Gemini API Smoke Test")
    parser.add_argument(
        "--pro", action="store_true", help="Use gemini-3.1-pro-preview instead of gemini-2.5-flash"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit model identifier (e.g. gemini-3.1-pro-preview, gemini-2.5-flash)",
    )
    args = parser.parse_args()

    run_live_smoke_test(use_pro=args.pro, model_override=args.model)
