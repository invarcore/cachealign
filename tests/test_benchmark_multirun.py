# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
Unit tests for Multi-Run Statistical Benchmark Harness and Gate.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.run_multirun_benchmark import (  # noqa: E402
    compute_distribution_stats,
    get_student_t_crit,
    run_multirun_benchmark,
    verify_multirun_truth_gate,
)


def test_student_t_critical_values():
    assert get_student_t_crit(29) == 2.045
    assert get_student_t_crit(1) == 12.706
    assert get_student_t_crit(100) < 2.00
    assert get_student_t_crit(100) > 1.96


def test_compute_distribution_stats_empty():
    stats = compute_distribution_stats([])
    assert stats["n"] == 0
    assert stats["mean"] == 0.0


def test_compute_distribution_stats_single():
    stats = compute_distribution_stats([42.0])
    assert stats["n"] == 1
    assert stats["mean"] == 42.0
    assert stats["std"] == 0.0
    assert stats["ci95_lower"] == 42.0
    assert stats["ci95_upper"] == 42.0


def test_compute_distribution_stats_sample():
    # Sample with known distribution: [10, 12, 14, 16, 18]
    data = [10.0, 12.0, 14.0, 16.0, 18.0]
    stats = compute_distribution_stats(data)
    assert stats["n"] == 5
    assert stats["mean"] == 14.0
    assert stats["min"] == 10.0
    assert stats["max"] == 18.0
    assert stats["median"] == 14.0
    assert stats["std"] > 3.1
    # 95% CI should bound the mean symmetrically
    assert stats["ci95_lower"] < 14.0
    assert stats["ci95_upper"] > 14.0
    assert round(stats["ci95_upper"] - stats["mean"], 4) == round(
        stats["mean"] - stats["ci95_lower"], 4
    )


def test_run_multirun_benchmark_smoke(tmp_path: Path):
    output_dir = tmp_path / "results"
    summary = run_multirun_benchmark(
        iterations=3,
        workload_key="coding_loop",
        provider_key="anthropic",
        turns=3,
        output_dir=str(output_dir),
    )

    assert summary["iterations_n"] == 3
    assert len(summary["summary"]) == 4  # 4 arms for coding_loop

    summary_file = output_dir / "multirun_benchmark_summary.json"
    logs_file = output_dir / "multirun_benchmark_logs.jsonl"
    assert summary_file.exists()
    assert logs_file.exists()

    # Verify truth gate passes with min_iterations=3 for smoke test
    assert verify_multirun_truth_gate(summary_file, min_iterations=3)
