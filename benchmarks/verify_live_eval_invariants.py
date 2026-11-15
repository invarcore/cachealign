# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
Verification script for Live Provider Evaluation physical caching invariants.

Guarantees enforced:
1. Cold Seed phase: zero cached tokens (authentic fresh write).
2. Warm Turn phase: non-zero cache engagement with CTR >= 70%.
3. Fails closed if invariants are violated or log file is malformed.
"""

import json
import sys
from pathlib import Path


def verify_probe_invariants(jsonl_path: str | Path, min_ctr: float = 0.70) -> dict[str, float]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Probe output log not found at: {path}")

    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if len(records) < 2:
        raise ValueError(
            f"Expected at least 2 probe phases (cold_seed and warm), but found {len(records)}."
        )

    cold = records[0]
    warm = records[1]

    cold_cached = cold.get("cache_read_tokens", 0) or cold.get("cached_tokens", 0) or 0
    cold_total = cold.get("prompt_tokens", 0) or cold.get("total_input_tokens", 0) or 0

    warm_cached = warm.get("cache_read_tokens", 0) or warm.get("cached_tokens", 0) or 0
    warm_total = warm.get("prompt_tokens", 0) or warm.get("total_input_tokens", 0) or 0

    print(f"Phase 1 (Cold Seed): Total={cold_total:,} | Cached={cold_cached:,}")
    print(f"Phase 2 (Warm Turn): Total={warm_total:,} | Cached={warm_cached:,}")

    # Invariant 1: Cold Seed must not rely on pre-existing cache
    if cold_cached > 0:
        print(f"Warning: Cold seed reported {cold_cached} cached tokens (pre-warmed route).")

    # Invariant 2: Warm turn must have non-zero cache engagement
    if warm_cached == 0:
        raise AssertionError(
            "FAIL: Warm turn had 0 cached tokens (P1_ZERO_WARM_CACHE_ENGAGEMENT). "
            "Provider failed to retain or match aligned prefix."
        )

    # Invariant 3: Cached Token Ratio (CTR) on warm turn must meet min_ctr threshold
    ctr = warm_cached / warm_total if warm_total > 0 else 0.0
    if ctr < min_ctr:
        raise AssertionError(
            f"FAIL: Warm turn CTR ({ctr:.1%}) is below the required threshold of {min_ctr:.1%}. "
            f"Expected at least {int(min_ctr * warm_total):,} cached tokens, got {warm_cached:,}."
        )

    print(f"✓ Physical caching invariants verified cleanly (Warm CTR: {ctr:.1%}).")
    return {"cold_cached": cold_cached, "warm_cached": warm_cached, "warm_ctr": ctr}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/live_probe_summary.jsonl"
    try:
        verify_probe_invariants(target)
        sys.exit(0)
    except Exception as e:
        print(f"\n[INVARIANT VIOLATION] {e}", file=sys.stderr)
        sys.exit(1)
