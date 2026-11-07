# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
Unit and Integration Tests for 3-Arm Benchmark Harness.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.run_3arm_benchmark import (  # noqa: E402
    HardenedKVCacheSimulator,
    get_workload_definitions,
    jitter_tool_list,
    run_benchmark,
)


def test_workload_definitions():
    defs = get_workload_definitions()
    assert "data_agent" in defs
    assert "rotating_rag" in defs
    assert "coding_loop" in defs

    assert len(defs["data_agent"]["tools"]) == 15
    assert len(defs["rotating_rag"]["tools"]) == 8
    assert len(defs["coding_loop"]["tools"]) == 6


def test_tool_jitter_preserves_content():
    tools = [
        {
            "name": "test_tool",
            "description": "desc",
            "input_schema": {"type": "object", "properties": {"a": 1, "b": 2}},
        },
    ]
    jittered = jitter_tool_list(tools)
    assert len(jittered) == 1
    assert jittered[0]["name"] == "test_tool"
    assert "properties" in jittered[0]["input_schema"]


def test_simulator_physics():
    sim = HardenedKVCacheSimulator(min_token_threshold=1024)

    # 1. Below threshold -> No cache hit even with breakpoint
    res_sub = sim.process_turn(
        arm="naive_provider",
        system="Short prompt",
        tools=[],
        messages=[{"role": "user", "content": "hi"}],
        has_cache_control=True,
    )
    assert res_sub["cached_tokens"] == 0

    # 2. Above threshold -> Turn 1 writes, Turn 2 with identical prefix hits!
    long_sys = "Invariant system content " * 150  # ~3000 chars -> ~780 tokens
    long_tools = [{"name": f"t_{i}", "description": "desc" * 50} for i in range(10)]  # large

    res_t1 = sim.process_turn(
        arm="cachealign_direct",
        system=long_sys,
        tools=long_tools,
        messages=[{"role": "user", "content": "Query 1"}],
        has_cache_control=True,
    )
    assert res_t1["cache_write_tokens"] > 1024
    assert res_t1["cached_tokens"] == 0
    assert res_t1["is_hit"] is False

    # Turn 2: Exact same prefix -> HIT
    res_t2 = sim.process_turn(
        arm="cachealign_direct",
        system=long_sys,
        tools=long_tools,
        messages=[{"role": "user", "content": "Query 2"}],
        has_cache_control=True,
    )
    assert res_t2["cached_tokens"] == res_t1["cache_write_tokens"]
    assert res_t2["cache_write_tokens"] == 0
    assert res_t2["is_hit"] is True
    assert res_t2["ttft_ms"] < 65.0


def test_run_benchmark_fast_smoke(tmp_path):
    output_dir = tmp_path / "results"
    summary = run_benchmark(
        backend="simulation",
        workload_key="coding_loop",
        turns=3,
        output_dir=str(output_dir),
    )

    assert summary["backend"] == "simulation"
    assert summary["turns"] == 3
    assert len(summary["summary"]) == 4  # 4 arms

    logs_file = output_dir / "3arm_benchmark_logs.jsonl"
    results_file = output_dir / "3arm_benchmark_results.json"
    assert logs_file.exists()
    assert results_file.exists()

    # Raw baseline vs naive provider vs cachealign
    arms_by_id = {row["arm_id"]: row for row in summary["summary"]}
    assert "raw_baseline" in arms_by_id
    assert "naive_provider" in arms_by_id
    assert "cachealign_direct" in arms_by_id
    assert "cachealign_litellm" in arms_by_id

    # CacheAlign achieves high CTR on coding loop
    assert arms_by_id["cachealign_direct"]["cached_token_ratio_pct"] > 50.0
    assert arms_by_id["cachealign_litellm"]["cached_token_ratio_pct"] > 50.0
