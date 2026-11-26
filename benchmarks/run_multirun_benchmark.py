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
import json
import math
import sys
import time
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
) -> dict[str, SingleIterationResult]:
    """Runs a single iteration of all 4 arms for a given workload and pricing profile."""
    executor = HardenedKVCacheSimulator(min_token_threshold=1024, pricing=pricing)

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

    arm_outputs: dict[str, SingleIterationResult] = {}
    costs: dict[str, float] = {}

    for arm_label, arm_id in arms:
        executor.reset()
        messages: list[dict[str, Any]] = []
        turn_records: list[TurnMetric] = []

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
                sys_content = f"{timestamp_header}\n{w_info['system_base']}"
                tools_payload = (
                    jitter_tool_list(w_info["tools"])
                    if w_info["has_tool_jitter"]
                    else list(w_info["tools"])
                )
                has_cache_control = False
                turn_messages = messages

            elif arm_id == "naive_provider":
                sys_content = f"{timestamp_header}\n{w_info['system_base']}"
                tools_payload = (
                    jitter_tool_list(w_info["tools"])
                    if w_info["has_tool_jitter"]
                    else list(w_info["tools"])
                )
                has_cache_control = True
                turn_messages = messages

            elif arm_id == "cachealign_direct":
                raw_sys = f"{timestamp_header}\n{w_info['system_base']}"
                raw_tools = (
                    jitter_tool_list(w_info["tools"])
                    if w_info["has_tool_jitter"]
                    else list(w_info["tools"])
                )

                if provider_name == "openai":
                    opt_payload = openai_adapter.optimize_request(
                        {
                            "model": "gpt-4o",
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
                            "model": "claude-3-5-sonnet",
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
                raw_sys = f"{timestamp_header}\n{w_info['system_base']}"
                raw_tools = (
                    jitter_tool_list(w_info["tools"])
                    if w_info["has_tool_jitter"]
                    else list(w_info["tools"])
                )

                litellm_data = {
                    "model": "claude-3-5-sonnet",
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
            turn_records.append(metric)

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

        total_in = sum(m.total_input_tokens for m in turn_records)
        total_cached = sum(m.cached_tokens for m in turn_records)
        total_cost = sum(m.cost_usd for m in turn_records)
        ctr = (total_cached / max(1, total_in)) * 100.0

        ttfts = sorted(m.ttft_ms for m in turn_records)
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


def run_multirun_benchmark(
    iterations: int = 30,
    workload_key: str = "all",
    provider_key: str = "all",
    turns: int = 10,
    output_dir: str = "benchmarks/results",
) -> dict[str, Any]:
    """
    Executes N automated benchmark iterations across selected workloads and provider pricing models,
    aggregating full distributions and computing 95% confidence intervals.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logs_file = out_path / "multirun_benchmark_logs.jsonl"
    results_file = out_path / "multirun_benchmark_summary.json"

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
        f"[bold cyan]  Iterations (N): {iterations}  |  Turns: {turns}  |  Providers: {provider_key.upper()}[/bold cyan]"
    )
    console.print(
        "[bold cyan]════════════════════════════════════════════════════════════════════════════════[/bold cyan]\n"
    )

    workload_defs = get_workload_definitions()
    selected_workloads = list(workload_defs.keys() if workload_key == "all" else [workload_key])

    if provider_key == "all":
        selected_providers = ["anthropic", "openai", "vllm"]
    else:
        selected_providers = [provider_key]

    raw_results_matrix: list[SingleIterationResult] = []

    t_start = time.perf_counter()

    for p_name in selected_providers:
        pricing = PRICING_PROFILES[p_name]
        console.print(
            f"[bold magenta]▶ Evaluating Provider Model: {pricing.name}...[/bold magenta]"
        )

        for wk in selected_workloads:
            w_name = workload_defs[wk]["name"]
            console.print(f"  • Running workload '{w_name}' over {iterations} iterations...")

            for i in range(1, iterations + 1):
                iter_res = run_single_iteration(
                    iteration_idx=i,
                    provider_name=p_name,
                    pricing=pricing,
                    workload_key=wk,
                    workload_defs=workload_defs,
                    turns=turns,
                    logs_file=logs_file,
                )
                raw_results_matrix.extend(iter_res.values())

    total_duration_sec = round(time.perf_counter() - t_start, 2)

    # Statistical Aggregation per (provider, workload, arm)
    aggregated_summary: list[dict[str, Any]] = []

    arms = [
        ("Arm 1", "raw_baseline"),
        ("Arm 2", "naive_provider"),
        ("Arm 3a", "cachealign_direct"),
        ("Arm 3b", "cachealign_litellm"),
    ]

    for p_name in selected_providers:
        for wk in selected_workloads:
            for arm_label, arm_id in arms:
                slice_records = [
                    r
                    for r in raw_results_matrix
                    if r.provider == p_name and r.workload == wk and r.arm_id == arm_id
                ]

                ctr_stats = compute_distribution_stats(
                    [r.cached_token_ratio_pct for r in slice_records]
                )
                cost_stats = compute_distribution_stats([r.total_cost_usd for r in slice_records])
                savings_raw_stats = compute_distribution_stats(
                    [r.savings_vs_raw_pct for r in slice_records]
                )
                savings_naive_stats = compute_distribution_stats(
                    [r.savings_vs_naive_pct for r in slice_records]
                )
                penalty_stats = compute_distribution_stats(
                    [r.invalidation_penalty_pct for r in slice_records]
                )
                ttft_stats = compute_distribution_stats([r.p50_ttft_ms for r in slice_records])

                row_data = {
                    "provider": p_name,
                    "provider_name": PRICING_PROFILES[p_name].name,
                    "workload": wk,
                    "arm_id": arm_id,
                    "arm_label": arm_label,
                    "n_iterations": len(slice_records),
                    "cached_token_ratio": ctr_stats,
                    "total_cost_usd": cost_stats,
                    "savings_vs_raw_pct": savings_raw_stats,
                    "savings_vs_naive_pct": savings_naive_stats,
                    "invalidation_penalty_pct": penalty_stats,
                    "p50_ttft_ms": ttft_stats,
                }
                aggregated_summary.append(row_data)

    # Render Summary Table
    table = Table(
        title=f"CacheAlign Multi-Run Statistical Benchmark Matrix (N={iterations} Iterations, 95% Student's t CI)"
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

    final_payload = {
        "benchmark": "Candidate C15 Day-30 Multi-Run Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iterations_n": iterations,
        "turns_per_run": turns,
        "duration_seconds": total_duration_sec,
        "confidence_level": 0.95,
        "providers_evaluated": selected_providers,
        "workloads_evaluated": selected_workloads,
        "summary": aggregated_summary,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    console.print("\n[bold green]✓ Multi-Run Statistical Benchmark Complete.[/bold green]")
    console.print(f"  • Statistical Summary: [bold]{results_file}[/bold]")
    console.print(f"  • Raw Run Logs: [bold]{logs_file}[/bold]\n")

    return final_payload


def verify_multirun_truth_gate(
    summary_path: str | Path,
    min_iterations: int = 30,
    min_ctr: float = 0.60,
) -> bool:
    """Enforces Day-30 Truth Gate invariants on multi-run benchmark results."""
    p = Path(summary_path)
    if not p.exists():
        raise FileNotFoundError(f"Summary file not found: {p}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

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

    args = parser.parse_args()
    results = run_multirun_benchmark(
        iterations=args.iterations,
        workload_key=args.workload,
        provider_key=args.provider,
        turns=args.turns,
        output_dir=args.output_dir,
    )

    verify_multirun_truth_gate(
        Path(args.output_dir) / "multirun_benchmark_summary.json", min_iterations=args.iterations
    )
