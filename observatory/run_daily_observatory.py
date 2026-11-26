#!/usr/bin/env python3
# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
Longitudinal Multi-Provider Cache Observatory Runner.
Executes scheduled daily smoke probes across Anthropic, OpenAI, AWS Bedrock,
and NVIDIA NIM when provider credentials are present, skips unconfigured
providers cleanly with exit code 0, and appends observation vectors to
observatory/cache_probe_history.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cachealign.cache_probe_runner import ProbeObservation, run_probe_sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cache_observatory")

OBSERVATORY_DIR = Path(__file__).resolve().parent
DEFAULT_HISTORY_FILE = OBSERVATORY_DIR / "cache_probe_history.jsonl"


PROVIDER_MATRIX = [
    {
        "provider_id": "anthropic",
        "route_name": "anthropic-direct",
        "api_key_envs": ["ANTHROPIC_API_KEY"],
        "base_url_env": "ANTHROPIC_BASE_URL",
        "default_base_url": "https://api.anthropic.com",
        "api_format": "anthropic",
        "model": "claude-sonnet-4-6",
        "tokens": 1500,
    },
    {
        "provider_id": "openai",
        "route_name": "openai-direct",
        "api_key_envs": ["OPENAI_API_KEY"],
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://api.openai.com",
        "api_format": "openai",
        "model": "gpt-4o-mini",
        "tokens": 1500,
    },
    {
        "provider_id": "bedrock",
        "route_name": "aws-bedrock-gateway",
        "api_key_envs": ["AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_API_KEY"],
        "base_url_env": "BEDROCK_BASE_URL",
        "default_base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "api_format": "anthropic",
        "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "tokens": 1500,
    },
    {
        "provider_id": "nvidia_nim",
        "route_name": "nvidia-nim-direct",
        "api_key_envs": ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"],
        "base_url_env": "NVIDIA_NIM_BASE_URL",
        "default_base_url": "https://integrate.api.nvidia.com",
        "api_format": "openai",
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "tokens": 1500,
    },
]


def resolve_api_key(env_names: list[str]) -> str | None:
    for name in env_names:
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    return None


def run_provider_smoke_probes(
    mode: str = "daily-smoke",
    output_jsonl: Path | str = DEFAULT_HISTORY_FILE,
    provider_filter: str | None = None,
    mock_runner: Any = None,
) -> list[ProbeObservation]:
    out_path = Path(output_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_observations: list[ProbeObservation] = []
    executed_providers: list[str] = []
    skipped_providers: list[str] = []

    for prov in PROVIDER_MATRIX:
        p_id = prov["provider_id"]
        if provider_filter and provider_filter.lower() not in (p_id, prov["route_name"]):
            continue

        api_key = resolve_api_key(prov["api_key_envs"])
        if not api_key:
            logger.info(
                "[SKIP] Provider '%s' skipped: none of %s configured in environment.",
                p_id,
                prov["api_key_envs"],
            )
            skipped_providers.append(p_id)
            continue

        base_url = os.environ.get(prov["base_url_env"], prov["default_base_url"])
        logger.info(
            "[*] Executing %s smoke probe for '%s' (%s / %s)...",
            mode,
            p_id,
            base_url,
            prov["model"],
        )

        runner_fn = mock_runner or run_probe_sequence
        try:
            obs_list = runner_fn(
                route_name=prov["route_name"],
                base_url=base_url,
                api_key=api_key,
                api_format=prov["api_format"],
                model=prov["model"],
                target_tokens=prov["tokens"],
                mode=mode,
            )
            all_observations.extend(obs_list)
            executed_providers.append(p_id)

            with open(out_path, "a", encoding="utf-8") as f:
                for obs in obs_list:
                    f.write(
                        json.dumps(
                            asdict(obs), sort_keys=True, separators=(",", ":"), ensure_ascii=False
                        )
                        + "\n"
                    )
            logger.info("[+] Successfully anchored %d observations for '%s'.", len(obs_list), p_id)
        except Exception as exc:
            logger.error("[-] Error during probe execution for '%s': %s", p_id, exc)

    logger.info("=== Daily Observatory Smoke Probes Complete ===")
    logger.info("  Executed providers: %s", executed_providers or "none")
    logger.info("  Skipped providers (clean 0 exit): %s", skipped_providers)
    logger.info("  Total new observations: %d", len(all_observations))
    return all_observations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run longitudinal daily smoke probes across LLM providers."
    )
    parser.add_argument(
        "--mode",
        choices=["daily-smoke", "full"],
        default="daily-smoke",
        help="Probe sequence mode",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_HISTORY_FILE),
        help="Target history JSONL file",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Optional single provider to evaluate (anthropic, openai, bedrock, nvidia_nim)",
    )
    args = parser.parse_args(argv)

    run_provider_smoke_probes(
        mode=args.mode,
        output_jsonl=args.output_jsonl,
        provider_filter=args.provider,
    )
    # Always exit 0 when skipping unconfigured providers
    return 0


if __name__ == "__main__":
    sys.exit(main())
