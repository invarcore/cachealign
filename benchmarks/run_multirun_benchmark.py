# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
Candidate C15 / Day-30 Multi-Run Benchmark Gate:
Automated Multi-Iteration Statistical Verification Harness for CacheAlign.

Executes N independent runs (default N >= 30) across Anthropic, OpenAI, and vLLM
trace workloads, computing sample mean, sample standard deviation, and 95% Student's t
confidence intervals for:
- Cached Token Ratio (CTR %)
- Net Cost Reduction vs Raw Baseline (NCR %)
- Net Cost Reduction vs Naive Provider Caching (%)
- Cache Invalidation Penalty (%)
- p50 and p95 Time-To-First-Token (TTFT ms)

Graduates "self-reported single-run" labels to statistically verified multi-run data.

Usage:
    uv run python benchmarks/run_multirun_benchmark.py [--iterations 30] [--workload all] [--provider all]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.run_3arm_benchmark import (  # noqa: E402
    PRICING_PROFILES,
    HardenedKVCacheSimulator,
    HardenedLiveExecutor,
    PricingProfile,
    TurnMetric,
    get_workload_definitions,
    jitter_tool_list,
)
from cachealign.adapters.anthropic import AnthropicAdapter  # noqa: E402
from cachealign.adapters.openai import OpenAIAdapter  # noqa: E402
from cachealign.config import CacheAlignConfig  # noqa: E402
from cachealign.integrations.litellm import CacheAlignLiteLLMHandler  # noqa: E402

console = Console()

# Two-tailed Student's t critical values for alpha = 0.05 (95% Confidence Interval)
# Indexed by degrees of freedom nu = N - 1
STUDENT_T_CRITICAL_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,  # Standard for N = 30
    30: 2.042,
}


def get_student_t_crit(nu: int) -> float:
    """Returns the two-tailed Student's t critical value for df = nu at alpha = 0.05."""
    if nu in STUDENT_T_CRITICAL_95:
        return STUDENT_T_CRITICAL_95[nu]
    if nu <= 0:
        return 1.96
    # Hill's asymptotic expansion approximation for large degrees of freedom
    return 1.95996 + (2.37227 / nu) + (2.8225 / (nu**2))


def compute_distribution_stats(values: list[float]) -> dict[str, float]:
    """Computes mean, sample stddev, SE, and 95% Student's t confidence interval."""
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "se": 0.0,
            "ci95_lower": 0.0,
            "ci95_upper": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }
    mean = sum(values) / n
    if n == 1:
        val = round(mean, 4)
        return {
            "n": 1,
            "mean": val,
            "std": 0.0,
            "se": 0.0,
            "ci95_lower": val,
            "ci95_upper": val,
            "min": val,
            "max": val,
            "median": val,
        }

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(max(0.0, variance))
    se = std / math.sqrt(n)
    nu = n - 1
    t_crit = get_student_t_crit(nu)
    margin = t_crit * se

    sorted_vals = sorted(values)
    median = (
        sorted_vals[n // 2] if n % 2 != 0 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    )

    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "se": round(se, 4),
        "ci95_lower": round(mean - margin, 4),
        "ci95_upper": round(mean + margin, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "median": round(median, 4),
    }


@dataclass
class SingleIterationResult:
    iteration: int
    provider: str
    workload: str
    arm_id: str
    arm_label: str
    total_input_tokens: int
    total_cached_tokens: int
    cached_token_ratio_pct: float
    total_cost_usd: float
    savings_vs_raw_pct: float
    savings_vs_naive_pct: float
    invalidation_penalty_pct: float
    p50_ttft_ms: float
    p95_ttft_ms: float


def run_single_iteration(
    iteration_idx: int,
    provider_name: str,
    pricing: PricingProfile,
    workload_key: str,
    workload_defs: dict[str, Any],
    turns: int = 10,
    logs_file: Path | None = None,
    backend: str = "simulation",
    live_executor: HardenedLiveExecutor | None = None,
    preloaded_arms: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, SingleIterationResult]:
    """Runs a single iteration of all 4 arms for a given workload and pricing profile or live backend."""
    executor: Any = (
        live_executor
        if backend == "live" and live_executor is not None
        else HardenedKVCacheSimulator(min_token_threshold=1024, pricing=pricing)
    )

    w_info = workload_defs[workload_key]
    arms = [
        ("Arm 1", "raw_baseline"),
        ("Arm 2", "naive_provider"),
        ("Arm 3a", "cachealign_direct"),
        ("Arm 3b", "cachealign_litellm"),
    ]

    anthropic_adapter = AnthropicAdapter(config=CacheAlignConfig(min_token_threshold=1024))
    openai_adapter = OpenAIAdapter(config=CacheAlignConfig(min_token_threshold=1024))
    litellm_handler = CacheAlignLiteLLMHandler(
        config=CacheAlignConfig(min_token_threshold=1024), verbose=False
    )

    arm_turn_records: dict[str, list[dict[str, Any]]] = {}
    iter_salt = hashlib.sha256(f"{time.time_ns()}:{iteration_idx}".encode()).hexdigest()[:10]

    for _arm_label, arm_id in arms:
        if preloaded_arms and arm_id in preloaded_arms:
            arm_turn_records[arm_id] = preloaded_arms[arm_id]
            continue

        executor.reset()
        messages: list[dict[str, Any]] = []
        turn_records: list[dict[str, Any]] = []
        sys_base = (
            f"[MultiRunLive:{iter_salt}:{arm_id}]\n{w_info['system_base']}"
            if backend == "live"
            else w_info["system_base"]
        )

        for turn in range(1, turns + 1):
            timestamp_header = f"Current Time: 2026-11-26 14:{turn:02d}:00 UTC\nSession-ID: sess_iter_{iteration_idx}_{turn}"
            user_content = (
                f"Turn {turn}: Process analytics pipeline and verify compliance constraints."
            )

            if workload_key == "rotating_rag":
                user_content += (
                    f"\n[Evidence Chunk {turn}A: Verified RLS context digest abc{turn}456]"
                    f"\n[Evidence Chunk {turn}B: Policy compliance verified at epoch {turn}000]"
                )
            elif workload_key == "coding_loop":
                user_content += (
                    f"\n[Build #{turn} exit status: 0, warnings: 0, test_cases_passed: {turn * 12}]"
                )

            messages.append({"role": "user", "content": user_content})

            if arm_id == "raw_baseline":
                sys_content = f"{timestamp_header}\n{sys_base}"
                tools_payload = (
                    jitter_tool_list(w_info["tools"])
                    if w_info["has_tool_jitter"]
                    else list(w_info["tools"])
                )
                has_cache_control = False
                turn_messages = messages

            elif arm_id == "naive_provider":
                sys_content = f"{timestamp_header}\n{sys_base}"
                tools_payload = (
                    jitter_tool_list(w_info["tools"])
                    if w_info["has_tool_jitter"]
                    else list(w_info["tools"])
                )
                has_cache_control = True
                turn_messages = messages

            elif arm_id == "cachealign_direct":
                raw_sys = f"{timestamp_header}\n{sys_base}"
                raw_tools = (
                    jitter_tool_list(w_info["tools"])
                    if w_info["has_tool_jitter"]
                    else list(w_info["tools"])
                )

                if provider_name == "openai" or (
                    backend == "live"
                    and getattr(executor, "provider", None) in ("nvidia", "nvidia_nim")
                ):
                    opt_payload = openai_adapter.optimize_request(
                        {
                            "model": getattr(executor, "model", "gpt-4o"),
                            "system": raw_sys,
                            "tools": [{"type": "function", "function": t} for t in raw_tools],
                            "messages": messages,
                        }
                    )
                    messages_opt = opt_payload.optimized_kwargs.get("messages", [])
                    sys_content = next(
                        (m["content"] for m in messages_opt if m.get("role") == "system"), ""
                    )
                    tools_payload = opt_payload.optimized_kwargs.get("tools")
                    has_cache_control = True
                    turn_messages = [m for m in messages_opt if m.get("role") != "system"]
                else:
                    opt_payload = anthropic_adapter.optimize_request(
                        {
                            "model": "claude-sonnet-4-6",
                            "system": raw_sys,
                            "tools": raw_tools,
                            "messages": messages,
                        }
                    )
                    sys_content = opt_payload.optimized_kwargs.get("system")
                    tools_payload = opt_payload.optimized_kwargs.get("tools")
                    has_cache_control = True
                    turn_messages = opt_payload.optimized_kwargs.get("messages", messages)

            elif arm_id == "cachealign_litellm":
                raw_sys = f"{timestamp_header}\n{sys_base}"
                raw_tools = (
                    jitter_tool_list(w_info["tools"])
                    if w_info["has_tool_jitter"]
                    else list(w_info["tools"])
                )

                litellm_data = {
                    "model": getattr(executor, "model", "claude-sonnet-4-6"),
                    "messages": [{"role": "system", "content": raw_sys}, *messages],
                    "tools": [{"function": t} for t in raw_tools],
                }
                opt_litellm = litellm_handler.async_pre_call_hook(
                    {}, None, litellm_data, "completion"
                )
                sys_content = next(
                    (m["content"] for m in opt_litellm["messages"] if m["role"] == "system"), ""
                )
                tools_payload = [t["function"] for t in opt_litellm["tools"]]
                has_cache_control = True
                turn_messages = [m for m in opt_litellm["messages"] if m["role"] != "system"]

            res = executor.process_turn(
                arm=arm_id,
                system=sys_content,
                tools=tools_payload,
                messages=turn_messages,
                has_cache_control=has_cache_control,
            )

            metric = TurnMetric(
                workload=workload_key,
                arm=arm_id,
                turn=turn,
                total_input_tokens=res["total_input_tokens"],
                cached_tokens=res["cached_tokens"],
                cache_write_tokens=res["cache_write_tokens"],
                uncached_tokens=res["uncached_tokens"],
                cost_usd=round(res["cost_usd"], 6),
                ttft_ms=res["ttft_ms"],
                is_hit=res["is_hit"],
                prefix_hash=res["prefix_hash"],
            )
            turn_records.append(asdict(metric))

            if logs_file:
                record_entry = asdict(metric)
                record_entry["iteration"] = iteration_idx
                record_entry["provider"] = provider_name
                with open(logs_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record_entry) + "\n")

            messages.append(
                {
                    "role": "assistant",
                    "content": f"Acknowledged turn {turn} completed successfully.",
                }
            )

        arm_turn_records[arm_id] = turn_records

    return _compute_iteration_arm_outputs(
        iteration_idx=iteration_idx,
        provider_name=provider_name,
        workload_key=workload_key,
        arm_turn_records=arm_turn_records,
    )


def _compute_iteration_arm_outputs(
    iteration_idx: int,
    provider_name: str,
    workload_key: str,
    arm_turn_records: dict[str, list[dict[str, Any]]],
) -> dict[str, SingleIterationResult]:
    """Compute per-arm SingleIterationResult metrics and relative savings from turn records."""
    arms = [
        ("Arm 1", "raw_baseline"),
        ("Arm 2", "naive_provider"),
        ("Arm 3a", "cachealign_direct"),
        ("Arm 3b", "cachealign_litellm"),
    ]
    arm_outputs: dict[str, SingleIterationResult] = {}
    costs: dict[str, float] = {}

    for arm_label, arm_id in arms:
        turn_records = arm_turn_records[arm_id]
        total_in = sum(int(m["total_input_tokens"]) for m in turn_records)
        total_cached = sum(int(m["cached_tokens"]) for m in turn_records)
        total_cost = sum(float(m["cost_usd"]) for m in turn_records)
        ctr = (total_cached / max(1, total_in)) * 100.0

        ttfts = sorted(float(m["ttft_ms"]) for m in turn_records)
        p50_ttft = ttfts[len(ttfts) // 2]
        p95_ttft = ttfts[int(len(ttfts) * 0.95)]

        costs[arm_id] = total_cost
        arm_outputs[arm_id] = SingleIterationResult(
            iteration=iteration_idx,
            provider=provider_name,
            workload=workload_key,
            arm_id=arm_id,
            arm_label=arm_label,
            total_input_tokens=total_in,
            total_cached_tokens=total_cached,
            cached_token_ratio_pct=round(ctr, 2),
            total_cost_usd=round(total_cost, 4),
            savings_vs_raw_pct=0.0,
            savings_vs_naive_pct=0.0,
            invalidation_penalty_pct=0.0,
            p50_ttft_ms=p50_ttft,
            p95_ttft_ms=p95_ttft,
        )

    raw_cost = costs["raw_baseline"]
    naive_cost = costs["naive_provider"]
    invalidation_penalty_pct = round(((naive_cost - raw_cost) / max(1e-6, raw_cost)) * 100.0, 2)

    for arm_id, res in arm_outputs.items():
        cost = costs[arm_id]
        res.savings_vs_raw_pct = round(((raw_cost - cost) / max(1e-6, raw_cost)) * 100.0, 2)
        res.savings_vs_naive_pct = round(((naive_cost - cost) / max(1e-6, naive_cost)) * 100.0, 2)
        res.invalidation_penalty_pct = (
            invalidation_penalty_pct if arm_id == "naive_provider" else 0.0
        )

    return arm_outputs


class InsufficientCreditsError(RuntimeError):
    """Raised when an API provider reports zero remaining paid credits pre-flight."""


def check_openrouter_credits(api_key: str, timeout: float = 10.0) -> float:
    """Query OpenRouter /api/v1/credits and return remaining USD credit balance."""
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data") or {}
    total_credits = float(data.get("total_credits", 0.0) or 0.0)
    total_usage = float(data.get("total_usage", 0.0) or 0.0)
    return round(total_credits - total_usage, 6)


def _is_quota_or_payment_error(exc: Exception) -> bool:
    """Return True if an exception represents HTTP 402 payment required or 429 quota/rate limit."""
    if getattr(exc, "status_code", None) in (402, 429):
        return True
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in (
            "402",
            "payment required",
            "insufficient credit",
            "insufficient_quota",
            "more credits",
            "rate_limit",
            "429",
        )
    )


def _load_completed_iterations(
    logs_file: Path,
    turns: int,
) -> tuple[dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    """Load completed per-arm runs (of length == turns) from an existing JSONL log."""
    expected_arms = ("raw_baseline", "naive_provider", "cachealign_direct", "cachealign_litellm")
    grouped: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = {}
    if not logs_file.exists():
        return {}, []

    for line in logs_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            str(row.get("provider", "")),
            str(row.get("workload", "")),
            int(row.get("iteration", 0)),
        )
        arm = str(row.get("arm", ""))
        if arm in expected_arms and key[2] > 0:
            grouped.setdefault(key, {a: [] for a in expected_arms})[arm].append(row)

    resumed_arms: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = {}
    retained_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        arm_map = grouped[key]
        complete_arms = {a: arm_map[a] for a in expected_arms if len(arm_map[a]) == turns}
        if complete_arms:
            resumed_arms[key] = complete_arms
            for a in expected_arms:
                if a in complete_arms:
                    retained_rows.extend(complete_arms[a])
    return resumed_arms, retained_rows


def _write_summary_artifact(
    *,
    raw_results_matrix: list[SingleIterationResult],
    selected_providers: list[str],
    selected_workloads: list[str],
    backend: str,
    live_executor: HardenedLiveExecutor | None,
    requested_iterations: int,
    turns: int,
    duration_sec: float,
    results_file: Path,
    status: str = "complete",
    stop_reason: str | None = None,
    render_table: bool = True,
) -> dict[str, Any]:
    """Aggregate completed iteration results, optionally render table, and write summary JSON."""
    aggregated_summary: list[dict[str, Any]] = []
    arms = [
        ("Arm 1", "raw_baseline"),
        ("Arm 2", "naive_provider"),
        ("Arm 3a", "cachealign_direct"),
        ("Arm 3b", "cachealign_litellm"),
    ]

    completed_counts: list[int] = []
    for p_name in selected_providers:
        for wk in selected_workloads:
            for arm_label, arm_id in arms:
                slice_records = [
                    r
                    for r in raw_results_matrix
                    if r.provider == p_name and r.workload == wk and r.arm_id == arm_id
                ]
                completed_counts.append(len(slice_records))
                row_data = {
                    "provider": p_name,
                    "provider_name": (
                        f"Live {p_name} ({live_executor.model})"
                        if backend == "live" and live_executor
                        else PRICING_PROFILES[p_name].name
                    ),
                    "workload": wk,
                    "arm_id": arm_id,
                    "arm_label": arm_label,
                    "n_iterations": len(slice_records),
                    "cached_token_ratio": compute_distribution_stats(
                        [r.cached_token_ratio_pct for r in slice_records]
                    ),
                    "total_cost_usd": compute_distribution_stats(
                        [r.total_cost_usd for r in slice_records]
                    ),
                    "savings_vs_raw_pct": compute_distribution_stats(
                        [r.savings_vs_raw_pct for r in slice_records]
                    ),
                    "savings_vs_naive_pct": compute_distribution_stats(
                        [r.savings_vs_naive_pct for r in slice_records]
                    ),
                    "invalidation_penalty_pct": compute_distribution_stats(
                        [r.invalidation_penalty_pct for r in slice_records]
                    ),
                    "p50_ttft_ms": compute_distribution_stats(
                        [r.p50_ttft_ms for r in slice_records]
                    ),
                }
                aggregated_summary.append(row_data)

    completed_n = min(completed_counts) if completed_counts else 0

    if render_table:
        table = Table(
            title=f"CacheAlign Multi-Run Statistical Benchmark Matrix (N={completed_n} Iterations, 95% Student's t CI)"
        )
        table.add_column("Provider", style="cyan")
        table.add_column("Workload", style="yellow")
        table.add_column("Arm", style="bold")
        table.add_column("CTR (%) [Mean ± 95% CI]", justify="right", style="magenta")
        table.add_column("Cost Savings vs Raw (%)", justify="right", style="green")
        table.add_column("Invalidation Penalty", justify="right", style="red")
        table.add_column("p50 TTFT (ms) [Mean ± CI]", justify="right")

        for row in aggregated_summary:
            ctr = row["cached_token_ratio"]
            ctr_str = f"{ctr['mean']:.1f}% ± {(ctr['ci95_upper'] - ctr['mean']):.2f}%"
            sav_raw = row["savings_vs_raw_pct"]
            sav_str = (
                f"{sav_raw['mean']:.1f}% ± {(sav_raw['ci95_upper'] - sav_raw['mean']):.2f}%"
                if sav_raw["mean"] > 0
                else "-"
            )
            pen = row["invalidation_penalty_pct"]
            pen_str = f"+{pen['mean']:.1f}%" if pen["mean"] > 0 else "-"
            ttft = row["p50_ttft_ms"]
            ttft_str = f"{ttft['mean']:.1f} ± {(ttft['ci95_upper'] - ttft['mean']):.1f}ms"
            table.add_row(
                row["provider"],
                row["workload"],
                row["arm_label"],
                ctr_str,
                sav_str,
                pen_str,
                ttft_str,
            )
        console.print(table)

    final_payload: dict[str, Any] = {
        "benchmark": (
            "Candidate C15 Multi-Run Live API Benchmark"
            if backend == "live"
            else "Candidate C15 Multi-Run Simulation Benchmark"
        ),
        "execution_mode": "live_api" if backend == "live" else "deterministic_simulation",
        "is_live_api": backend == "live",
        "status": status,
        "stop_reason": stop_reason,
        "simulator": None if backend == "live" else "HardenedKVCacheSimulator",
        "live_model": live_executor.model if backend == "live" and live_executor else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iterations_n": completed_n,
        "completed_iterations_n": completed_n,
        "requested_iterations_n": requested_iterations,
        "turns_per_run": turns,
        "duration_seconds": duration_sec,
        "confidence_level": 0.95,
        "providers_evaluated": selected_providers,
        "workloads_evaluated": selected_workloads,
        "summary": aggregated_summary,
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    return final_payload


def run_multirun_benchmark(
    iterations: int = 30,
    workload_key: str = "all",
    provider_key: str = "all",
    turns: int = 10,
    output_dir: str = "benchmarks/results",
    backend: str = "simulation",
    live_provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    resume: bool = False,
    checkpoint_every: int = 5,
) -> dict[str, Any]:
    """
    Executes N automated benchmark iterations across selected workloads and provider pricing models
    (or live endpoints when backend='live'), aggregating distributions and 95% confidence intervals.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_prefix = "multirun_live" if backend == "live" else "multirun_benchmark"
    logs_file = out_path / f"{file_prefix}_logs.jsonl"
    results_file = out_path / f"{file_prefix}_summary.json"

    resumed_arms: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = {}
    if resume and logs_file.exists():
        resumed_arms, retained_rows = _load_completed_iterations(logs_file, turns=turns)
        with open(logs_file, "w", encoding="utf-8") as f:
            for row in retained_rows:
                f.write(json.dumps(row) + "\n")
    else:
        # Reset log file
        with open(logs_file, "w", encoding="utf-8") as f:
            pass

    console.print(
        "\n[bold cyan]════════════════════════════════════════════════════════════════════════════════[/bold cyan]"
    )
    console.print(
        "[bold white]  Candidate C15: Automated Multi-Run Benchmark Gate (N >= 30)  [/bold white]"
    )
    console.print(
        f"[bold cyan]  Backend: {backend.upper()}  |  Iterations (N): {iterations}  |  Turns: {turns}  |  Providers: {provider_key.upper()}[/bold cyan]"
    )
    console.print(
        "[bold cyan]════════════════════════════════════════════════════════════════════════════════[/bold cyan]\n"
    )

    workload_defs = get_workload_definitions()
    selected_workloads = list(workload_defs.keys() if workload_key == "all" else [workload_key])

    # Override Anthropic pricing profile label so retired models are never referenced
    PRICING_PROFILES[
        "anthropic"
    ].name = "Anthropic Claude Sonnet 4.6 ($3.00 in, $3.75 write, $0.30 read, $15.00 out)"

    live_executor: HardenedLiveExecutor | None = None
    if backend == "live":
        prov = (
            live_provider or (None if provider_key == "all" else provider_key) or "nvidia_nim"
        ).lower()
        key = api_key
        if not key:
            if prov in ("nvidia", "nvidia_nim"):
                key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
                if not key:
                    env_fcc = os.path.expanduser("~/.fcc/.env")
                    if os.path.exists(env_fcc):
                        with open(env_fcc, encoding="utf-8") as f:
                            for line in f:
                                if line.startswith("NVIDIA_NIM_API_KEY="):
                                    key = line.strip().split("=", 1)[1].strip("\"'")
                                    break
            elif prov == "openrouter":
                key = os.getenv("OPENROUTER_API_KEY")
            elif prov == "anthropic":
                key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(f"Live multi-run execution requires an API key for provider '{prov}'.")
        live_executor = HardenedLiveExecutor(api_key=key, provider=prov, model=model)
        if prov == "openrouter" and not str(live_executor.model).endswith(":free"):
            remaining_usd = check_openrouter_credits(key)
            if remaining_usd <= 0.0:
                raise InsufficientCreditsError(
                    f"OpenRouter account has ${remaining_usd:.4f} remaining paid credits for model "
                    f"'{live_executor.model}'. Top up credits or pass a ':free' model."
                )
        selected_providers = [prov]
        console.print(
            f"[bold magenta][LIVE API MODE] Executing {iterations} independent runs against "
            f"{prov} ({live_executor.model})...[/bold magenta]\n"
        )
    else:
        if provider_key == "all":
            selected_providers = ["anthropic", "openai", "vllm"]
        else:
            selected_providers = [provider_key]

        console.print(
            "[bold yellow][SIMULATION MODE] Executing deterministic trace replay via HardenedKVCacheSimulator "
            "(zero network calls; synthetic hardware timing).[/bold yellow]\n"
        )

    raw_results_matrix: list[SingleIterationResult] = []
    expected_arms = ("raw_baseline", "naive_provider", "cachealign_direct", "cachealign_litellm")
    t_start = time.perf_counter()
    stop_reason: str | None = None

    for p_name in selected_providers:
        pricing = PRICING_PROFILES.get(p_name, PRICING_PROFILES["openai"])
        display_name = (
            f"Live {p_name} ({live_executor.model})"
            if backend == "live" and live_executor
            else pricing.name
        )
        console.print(
            f"[bold magenta]▶ Evaluating Provider Model: {display_name}...[/bold magenta]"
        )

        for wk in selected_workloads:
            w_name = workload_defs[wk]["name"]
            console.print(f"  • Running workload '{w_name}' over {iterations} iterations...")

            for i in range(1, iterations + 1):
                resumed_key = (p_name, wk, i)
                preloaded = resumed_arms.get(resumed_key)
                if preloaded and all(a in preloaded for a in expected_arms):
                    iter_res = _compute_iteration_arm_outputs(
                        iteration_idx=i,
                        provider_name=p_name,
                        workload_key=wk,
                        arm_turn_records=preloaded,
                    )
                else:
                    try:
                        iter_res = run_single_iteration(
                            iteration_idx=i,
                            provider_name=p_name,
                            pricing=pricing,
                            workload_key=wk,
                            workload_defs=workload_defs,
                            turns=turns,
                            logs_file=logs_file,
                            backend=backend,
                            live_executor=live_executor,
                            preloaded_arms=preloaded,
                        )
                    except Exception as exc:
                        if _is_quota_or_payment_error(exc):
                            stop_reason = f"{type(exc).__name__}: {exc}"
                            console.print(
                                f"\n[bold yellow]⚠ Stopping gracefully at iteration {i}/{iterations} "
                                f"due to provider payment/quota limit: {stop_reason}[/bold yellow]"
                            )
                            break
                        raise
                    if backend == "live":
                        print(f"    [live] completed iteration {i}/{iterations}", flush=True)
                raw_results_matrix.extend(iter_res.values())

                if (
                    checkpoint_every > 0
                    and i % checkpoint_every == 0
                    and i < iterations
                    and len(selected_workloads) == 1
                ):
                    _write_summary_artifact(
                        raw_results_matrix=raw_results_matrix,
                        selected_providers=selected_providers,
                        selected_workloads=selected_workloads,
                        backend=backend,
                        live_executor=live_executor,
                        requested_iterations=iterations,
                        turns=turns,
                        duration_sec=round(time.perf_counter() - t_start, 2),
                        results_file=results_file,
                        status="checkpoint",
                        stop_reason=None,
                        render_table=False,
                    )
            if stop_reason:
                break
        if stop_reason:
            break

    total_duration_sec = round(time.perf_counter() - t_start, 2)
    final_status = "partial_stopped" if stop_reason else "complete"
    final_payload = _write_summary_artifact(
        raw_results_matrix=raw_results_matrix,
        selected_providers=selected_providers,
        selected_workloads=selected_workloads,
        backend=backend,
        live_executor=live_executor,
        requested_iterations=iterations,
        turns=turns,
        duration_sec=total_duration_sec,
        results_file=results_file,
        status=final_status,
        stop_reason=stop_reason,
        render_table=True,
    )

    console.print("\n[bold green]✓ Multi-Run Statistical Benchmark Complete.[/bold green]")
    console.print(f"  • Statistical Summary: [bold]{results_file}[/bold]")
    console.print(f"  • Raw Run Logs: [bold]{logs_file}[/bold]\n")

    return final_payload


def verify_multirun_truth_gate(
    summary_path: str | Path,
    min_iterations: int = 30,
    min_ctr: float = 0.60,
    require_live: bool = False,
) -> bool:
    """Enforces Day-30 Truth Gate invariants on multi-run benchmark results."""
    p = Path(summary_path)
    if not p.exists():
        raise FileNotFoundError(f"Summary file not found: {p}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    if require_live and not data.get("is_live_api", False):
        raise AssertionError(
            "FAIL: Summary reports deterministic_simulation (is_live_api=False) while require_live=True."
        )

    n_runs = data.get("iterations_n", 0)
    if n_runs < min_iterations:
        raise AssertionError(
            f"FAIL: Iteration count ({n_runs}) is below required minimum of {min_iterations}."
        )

    summary = data.get("summary", [])
    if not summary:
        raise AssertionError("FAIL: Summary matrix is empty.")

    for row in summary:
        arm = row["arm_id"]
        ctr = row["cached_token_ratio"]

        if arm == "raw_baseline":
            if ctr["mean"] != 0.0 or ctr["max"] != 0.0:
                raise AssertionError(
                    f"FAIL: Raw Baseline reported non-zero cache hits (CTR={ctr['mean']}%)."
                )

        if arm in ("cachealign_direct", "cachealign_litellm"):
            if ctr["mean"] < min_ctr * 100.0:
                raise AssertionError(
                    f"FAIL: {arm} mean CTR ({ctr['mean']}%) is below {min_ctr * 100.0}% truth threshold on {row['workload']}."
                )
            if ctr["ci95_lower"] < (min_ctr - 0.05) * 100.0:
                raise AssertionError(
                    f"FAIL: {arm} lower 95% CI ({ctr['ci95_lower']}%) is below {(min_ctr - 0.05) * 100.0}% threshold."
                )

    print(
        f"[PASS] Multi-Run Truth Gate verified across N={n_runs} iterations (All Invariants Satisfied)."
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Candidate C15: Multi-Run Statistical Benchmark Gate"
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=30,
        help="Number of benchmark iterations (default: 30)",
    )
    parser.add_argument(
        "--workload", choices=["all", "data_agent", "rotating_rag", "coding_loop"], default="all"
    )
    parser.add_argument("--provider", choices=["all", "anthropic", "openai", "vllm"], default="all")
    parser.add_argument("--turns", type=int, default=10, help="Turns per workload run")
    parser.add_argument(
        "--output-dir", default="benchmarks/results", help="Directory for artifacts"
    )
    parser.add_argument(
        "--backend", choices=["simulation", "live"], default="simulation", help="Execution backend"
    )
    parser.add_argument(
        "--live-provider",
        choices=["anthropic", "openrouter", "nvidia", "nvidia_nim"],
        default=None,
        help="Live provider when --backend live is set",
    )
    parser.add_argument("--model", default=None, help="Explicit model name for live backend")
    parser.add_argument("--api-key", default=None, help="Explicit API key for live backend")
    parser.add_argument(
        "--min-ctr",
        type=float,
        default=0.60,
        help="Minimum cached token ratio threshold (e.g. 0.35 for 2-turn live runs)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed iterations/arms already recorded in the output JSONL log",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Flush an interim summary JSON checkpoint every k iterations (0 disables)",
    )

    args = parser.parse_args()
    results = run_multirun_benchmark(
        iterations=args.iterations,
        workload_key=args.workload,
        provider_key=args.provider,
        turns=args.turns,
        output_dir=args.output_dir,
        backend=args.backend,
        live_provider=args.live_provider,
        api_key=args.api_key,
        model=args.model,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )

    out_prefix = "multirun_live" if args.backend == "live" else "multirun_benchmark"
    verify_multirun_truth_gate(
        Path(args.output_dir) / f"{out_prefix}_summary.json",
        min_iterations=args.iterations,
        min_ctr=args.min_ctr,
        require_live=(args.backend == "live"),
    )
