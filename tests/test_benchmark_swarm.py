# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
Unit tests for Multi-Agent Swarm KV-Cache Concurrency Simulation.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.eval_concurrent_agent_swarm import (  # noqa: E402
    PagedAttentionSimulator,
    run_swarm_evaluation,
)


def test_paged_attention_simulator_prefix_sharing():
    sim = PagedAttentionSimulator(page_size=4)

    # Ingest first sequence: ["a", "b", "c", "d", "e", "f", "g", "h"] -> 2 blocks
    tot_1, new_1, shared_1, _ = sim.ingest_token_stream(["a", "b", "c", "d", "e", "f", "g", "h"])
    assert tot_1 == 2
    assert new_1 == 2
    assert shared_1 == 0

    # Ingest second sequence sharing the first block ["a", "b", "c", "d", "x", "y", "z", "w"]
    tot_2, new_2, shared_2, _ = sim.ingest_token_stream(["a", "b", "c", "d", "x", "y", "z", "w"])
    assert tot_2 == 2
    assert new_2 == 1  # only ["x", "y", "z", "w"] is new
    assert shared_2 == 1  # ["a", "b", "c", "d"] is shared!


def test_run_swarm_evaluation_invariants(tmp_path: Path):
    results = run_swarm_evaluation(output_dir=str(tmp_path))

    assert results["subagents_evaluated"] == 5
    assert results["cachealign"]["vram_reduction_pct"] > 50.0
    assert results["cachealign"]["subagents_ttft_speedup"] > 3.0

    # Verify that Orchestrator writes root and subsequent agents share >70%
    agent_results = results["cachealign"]["results"]
    assert agent_results[0]["shared_blocks"] == 0  # cold root
    for a in agent_results[1:]:
        assert a["sharing_ratio"] >= 70.0
