#!/usr/bin/env python3
# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
cache-probe (cachealign-probe) Reference Runner.
Executes byte-identical warm/TTL/route-affinity prompt-cache probes across
OpenAI-compatible and Anthropic-compatible routes and appends an immutable
JSONL drift log for longitudinal tracking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


def _nfc(val: str) -> str:
    return unicodedata.normalize("NFC", val)


def build_aligned_prefix(target_tokens: int = 1200) -> str:
    """
    Generates a deterministic, high-entropy token block (>1024 tokens) so
    provider prefix caches engage reliably without synthetic repetition compression.
    """
    blocks: list[str] = []
    count = max(1, target_tokens // 16)
    for idx in range(count):
        raw_seed = f"invarcore-cachealign-block-{idx}"
        digest = hashlib.sha256(raw_seed.encode("utf-8")).hexdigest()[:24]
        blocks.append(f"[BLK:{idx:04d}|DIGEST:{digest}|SPEC:MCP_2026_07_28_INVARIANT_CHECK]")
    return "\n".join(blocks)


@dataclass
class ProbeObservation:
    timestamp: str
    route_name: str
    api_format: str  # "openai" | "anthropic"
    model: str
    probe_phase: str  # "cold_seed" | "warm_10s" | "affinity_2" | "affinity_3" | "ttl_boundary"
    delay_seconds: float
    http_status: int
    latency_ms: float
    prompt_tokens: int
    cache_write_tokens: int | None
    cache_read_tokens: int | None
    usage_field_present: bool
    prefix_sha256: str
    invariants_failed: list[str]


def send_probe_request(
    base_url: str,
    api_key: str,
    api_format: str,
    model: str,
    prefix_text: str,
    ttl_hint: str | None = None,
    timeout: float = 45.0,
) -> dict[str, Any]:
    t0 = time.monotonic()
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "invarcore-cache-probe/0.1.0",
    }
    norm_format = _nfc(api_format).strip().lower()
    if norm_format == "anthropic":
        url = f"{base_url.rstrip('/')}/v1/messages"
        headers["x-api-key"] = api_key.strip()
        headers["anthropic-version"] = "2023-06-01"
        cache_ctrl: dict[str, Any] = {"type": "ephemeral"}
        if ttl_hint:
            cache_ctrl["ttl"] = ttl_hint
        payload = {
            "model": model,
            "max_tokens": 16,
            "temperature": 0,
            "system": [
                {
                    "type": "text",
                    "text": prefix_text,
                    "cache_control": cache_ctrl,
                }
            ],
            "messages": [{"role": "user", "content": "Reply with OK only."}],
        }
    else:
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key.strip()}"
        payload = {
            "model": model,
            "max_tokens": 16,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prefix_text},
                {"role": "user", "content": "Reply with OK only."},
            ],
        }
        if ttl_hint:
            payload["prompt_cache_retention"] = ttl_hint

    req_data = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        status = int(e.code)
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"error": str(e)}
    except Exception as e:
        status = 0
        body = {"error": str(e)}

    latency_ms = round((time.monotonic() - t0) * 1000.0, 2)
    return {"status": status, "latency_ms": latency_ms, "body": body}


def parse_usage_fields(api_format: str, body: dict[str, Any]) -> dict[str, Any]:
    usage = body.get("usage") or {}
    norm_format = _nfc(api_format).strip().lower()
    if norm_format == "anthropic":
        c_write = usage.get("cache_creation_input_tokens")
        c_read = usage.get("cache_read_input_tokens")
        inp = usage.get("input_tokens", 0)
        present = ("cache_read_input_tokens" in usage) or ("cache_creation_input_tokens" in usage)
        return {
            "prompt_tokens": inp + (c_write or 0) + (c_read or 0),
            "cache_write_tokens": c_write,
            "cache_read_tokens": c_read,
            "usage_field_present": present,
        }
    else:
        details = usage.get("prompt_tokens_details") or {}
        c_read = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens"))
        c_write = details.get("cache_creation_tokens", usage.get("prompt_cache_miss_tokens"))
        present = ("cached_tokens" in details) or ("prompt_cache_hit_tokens" in usage)
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "cache_write_tokens": c_write,
            "cache_read_tokens": c_read,
            "usage_field_present": present,
        }


def run_probe_sequence(
    route_name: str,
    base_url: str,
    api_key: str,
    api_format: str,
    model: str,
    target_tokens: int = 1200,
    ttl_check_seconds: int | None = None,
    mode: str = "full",
    send_fn: Callable[..., dict[str, Any]] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> list[ProbeObservation]:
    if not api_key or not api_key.strip():
        raise ValueError("api_key must be non-empty and non-whitespace.")
    prefix_text = build_aligned_prefix(target_tokens)
    prefix_hash = hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()[:16]
    observations: list[ProbeObservation] = []

    norm_mode = _nfc(mode).strip().lower()
    if norm_mode == "daily-smoke":
        phases: list[tuple[str, float]] = [
            ("cold_seed", 0.0),
            ("warm_10s", 10.0),
        ]
    elif norm_mode == "full":
        phases = [
            ("cold_seed", 0.0),
            ("warm_10s", 10.0),
            ("affinity_2", 2.0),
            ("affinity_3", 2.0),
        ]
        if ttl_check_seconds:
            phases.append(("ttl_boundary", float(ttl_check_seconds)))
    else:
        raise ValueError(f"Unknown mode '{mode}': expected 'full' or 'daily-smoke'")

    requester = send_fn or send_probe_request
    sleeper = sleep_fn or time.sleep

    for phase_name, delay_s in phases:
        if delay_s > 0:
            sleeper(delay_s)
        res = requester(
            base_url=base_url,
            api_key=api_key.strip(),
            api_format=api_format,
            model=model,
            prefix_text=prefix_text,
        )
        parsed = parse_usage_fields(api_format, res.get("body", {}))
        failures: list[str] = []
        if not parsed["usage_field_present"]:
            failures.append("P4_MISSING_CACHE_USAGE_FIELD")
        c_read = parsed["cache_read_tokens"] or 0
        if phase_name == "warm_10s" and c_read == 0:
            failures.append("P1_ZERO_WARM_CACHE_ENGAGEMENT")
        elif phase_name in ("affinity_2", "affinity_3") and c_read == 0:
            failures.append("P5_ROUTE_AFFINITY_COLD_DROP")
        elif phase_name == "ttl_boundary" and c_read == 0:
            failures.append("P2_EARLY_TTL_EVICTION")

        obs = ProbeObservation(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            route_name=route_name,
            api_format=api_format,
            model=model,
            probe_phase=phase_name,
            delay_seconds=delay_s,
            http_status=res.get("status", 0),
            latency_ms=res.get("latency_ms", 0.0),
            prompt_tokens=parsed["prompt_tokens"],
            cache_write_tokens=parsed["cache_write_tokens"],
            cache_read_tokens=parsed["cache_read_tokens"],
            usage_field_present=parsed["usage_field_present"],
            prefix_sha256=prefix_hash,
            invariants_failed=failures,
        )
        observations.append(obs)
    return observations


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="cache-probe v1 scheduled prompt-cache regression monitor"
    )
    parser.add_argument("--route-name", default="openai-direct", help="Logical route name")
    parser.add_argument(
        "--base-url", default="https://api.openai.com", help="Provider or gateway base URL"
    )
    parser.add_argument(
        "--api-format", choices=["openai", "anthropic"], default="openai", help="API format"
    )
    parser.add_argument("--model", default="gpt-5.6", help="Target model identifier")
    parser.add_argument(
        "--tokens", type=int, default=1200, help="Deterministic prefix token target"
    )
    parser.add_argument(
        "--ttl-seconds",
        type=int,
        default=None,
        help="Optional idle TTL boundary probe in seconds",
    )
    parser.add_argument(
        "--output-jsonl",
        default="cache_probe_history.jsonl",
        help="Append-only JSONL history path",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "daily-smoke"],
        default="full",
        help="Probe sequence execution mode (full or daily-smoke)",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key or not api_key.strip():
        raise SystemExit("Error: set LLM_API_KEY environment variable before running live probes.")

    obs_list = run_probe_sequence(
        route_name=args.route_name,
        base_url=args.base_url,
        api_key=api_key,
        api_format=args.api_format,
        model=args.model,
        target_tokens=args.tokens,
        ttl_check_seconds=args.ttl_seconds,
        mode=args.mode,
    )

    with open(args.output_jsonl, "a", encoding="utf-8") as f:
        for obs in obs_list:
            f.write(
                json.dumps(
                    asdict(obs), sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                + "\n"
            )

    print(f"\n=== cache-probe Run Summary ({args.route_name} / {args.model}) ===")
    for obs in obs_list:
        status_str = (
            "PASS" if not obs.invariants_failed else f"FAIL ({','.join(obs.invariants_failed)})"
        )
        print(
            f"  [{obs.probe_phase:<12}] HTTP {obs.http_status} | {obs.latency_ms:>7.1f} ms | "
            f"read={obs.cache_read_tokens!s:<6} write={obs.cache_write_tokens!s:<6} | {status_str}"
        )

    with open(args.output_jsonl, "rb") as rf:
        ledger_sha256 = hashlib.sha256(rf.read()).hexdigest()
    print(f"[LEDGER_ANCHOR] file={args.output_jsonl} sha256={ledger_sha256}")


if __name__ == "__main__":
    main()
