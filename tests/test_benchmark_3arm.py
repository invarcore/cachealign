# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
Unit and Integration Tests for 3-Arm Benchmark Harness.
"""

import json
import sys
from pathlib import Path

import pytest

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


def test_verify_probe_invariants_pass_and_fail(tmp_path):
    from benchmarks.verify_live_eval_invariants import verify_probe_invariants

    # 1. Valid passing log
    pass_file = tmp_path / "pass.jsonl"
    with open(pass_file, "w", encoding="utf-8") as f:
        f.write(
            json.dumps({"probe_phase": "cold_seed", "prompt_tokens": 1500, "cache_read_tokens": 0})
            + "\n"
        )
        f.write(
            json.dumps(
                {"probe_phase": "warm_10s", "prompt_tokens": 1500, "cache_read_tokens": 1200}
            )
            + "\n"
        )

    res = verify_probe_invariants(pass_file, min_ctr=0.70)
    assert res["warm_ctr"] == 1200 / 1500  # 80%

    # 2. Failing: 0 cached tokens on warm turn
    fail_zero = tmp_path / "fail_zero.jsonl"
    with open(fail_zero, "w", encoding="utf-8") as f:
        f.write(
            json.dumps({"probe_phase": "cold_seed", "prompt_tokens": 1500, "cache_read_tokens": 0})
            + "\n"
        )
        f.write(
            json.dumps({"probe_phase": "warm_10s", "prompt_tokens": 1500, "cache_read_tokens": 0})
            + "\n"
        )

    with pytest.raises(AssertionError, match="P1_ZERO_WARM_CACHE_ENGAGEMENT"):
        verify_probe_invariants(fail_zero)

    # 3. Failing: low CTR (< 70%)
    fail_low = tmp_path / "fail_low.jsonl"
    with open(fail_low, "w", encoding="utf-8") as f:
        f.write(
            json.dumps({"probe_phase": "cold_seed", "prompt_tokens": 1000, "cache_read_tokens": 0})
            + "\n"
        )
        f.write(
            json.dumps({"probe_phase": "warm_10s", "prompt_tokens": 1000, "cache_read_tokens": 500})
            + "\n"
        )

    with pytest.raises(AssertionError, match="below the required threshold"):
        verify_probe_invariants(fail_low, min_ctr=0.70)


def test_live_executor_disjoint_token_partition_and_non_anthropic_cost():
    from types import SimpleNamespace

    from benchmarks.run_3arm_benchmark import HardenedLiveExecutor

    executor = HardenedLiveExecutor(api_key="nvapi-test-key", provider="nvidia_nim")
    mock_usage = SimpleNamespace(
        prompt_tokens=2000,
        completion_tokens=100,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )

    class _FakeStreamClient:
        class _Completions:
            @staticmethod
            def create(**kwargs):
                if kwargs.get("stream"):
                    return iter(
                        [
                            SimpleNamespace(
                                id="chunk_1",
                                choices=[SimpleNamespace(delta=SimpleNamespace(content="ok", tool_calls=None))],
                                usage=None,
                            ),
                            SimpleNamespace(id="chunk_2", choices=[], usage=mock_usage),
                        ]
                    )
                return SimpleNamespace(id="resp_1", usage=mock_usage)

        def __init__(self):
            self.completions = self._Completions()

    executor.client = SimpleNamespace(chat=_FakeStreamClient())
    res = executor.process_turn(
        arm="raw_baseline",
        system="Static instructions",
        tools=[],
        messages=[{"role": "user", "content": "hi"}],
        has_cache_control=False,
    )
    assert res["total_input_tokens"] == 2000
    assert res["uncached_tokens"] + res["cache_write_tokens"] + res["cached_tokens"] == 2000
    assert res["cache_write_tokens"] == 0
    assert res["cost_usd"] == 0.0
    assert res["response_id"] == "chunk_2"
