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


def test_run_multirun_live_backend_mocked_executor(monkeypatch, tmp_path: Path):
    """Verify backend='live' records live metadata and passes verify_multirun_truth_gate(require_live=True)."""
    from types import SimpleNamespace

    from benchmarks import run_multirun_benchmark as rmb
    from benchmarks.run_3arm_benchmark import HardenedLiveExecutor

    class _FakeLiveExecutor(HardenedLiveExecutor):
        def __init__(self, api_key: str, provider: str = "nvidia_nim", model: str | None = None):
            super().__init__(api_key=api_key, provider=provider, model=model)
            self._call_idx = 0
            test_obj = self

            class _Completions:
                @staticmethod
                def create(**kwargs):
                    test_obj._call_idx += 1
                    msgs = kwargs.get("messages") or []
                    is_cachealign_warm = (
                        "<cachealign_ephemeral_context>" in str(kwargs) and len(msgs) > 1
                    )
                    cached = 4000 if is_cachealign_warm else 0
                    usage = SimpleNamespace(
                        prompt_tokens=5000,
                        completion_tokens=40,
                        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
                    )
                    if kwargs.get("stream"):
                        return iter(
                            [
                                SimpleNamespace(
                                    id=f"c_{test_obj._call_idx}",
                                    choices=[
                                        SimpleNamespace(
                                            delta=SimpleNamespace(content="ok", tool_calls=None)
                                        )
                                    ],
                                    usage=None,
                                ),
                                SimpleNamespace(
                                    id=f"resp_{test_obj._call_idx}",
                                    choices=[],
                                    usage=usage,
                                ),
                            ]
                        )
                    return SimpleNamespace(id=f"resp_{test_obj._call_idx}", usage=usage)

            self.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    monkeypatch.setattr(rmb, "HardenedLiveExecutor", _FakeLiveExecutor)
    monkeypatch.setattr(rmb.time, "sleep", lambda _s: None)

    output_dir = tmp_path / "live_results"
    summary = run_multirun_benchmark(
        iterations=2,
        workload_key="data_agent",
        turns=2,
        output_dir=str(output_dir),
        backend="live",
        live_provider="nvidia_nim",
        api_key="nvapi-unit-test",
    )

    assert summary["is_live_api"] is True
    assert summary["execution_mode"] == "live_api"
    assert summary["iterations_n"] == 2

    live_summary = output_dir / "multirun_live_summary.json"
    live_logs = output_dir / "multirun_live_logs.jsonl"
    assert live_summary.exists()
    assert live_logs.exists()
    assert verify_multirun_truth_gate(
        live_summary, min_iterations=2, min_ctr=0.35, require_live=True
    )

    # Verify resume=True skips completed iterations 1..2 and executes only iteration 3
    calls_before_resume = len(live_logs.read_text(encoding="utf-8").splitlines())
    resumed_summary = run_multirun_benchmark(
        iterations=3,
        workload_key="data_agent",
        turns=2,
        output_dir=str(output_dir),
        backend="live",
        live_provider="nvidia_nim",
        api_key="nvapi-unit-test",
        resume=True,
    )
    calls_after_resume = len(live_logs.read_text(encoding="utf-8").splitlines())
    assert calls_before_resume == 16
    assert calls_after_resume == 24
    assert resumed_summary["iterations_n"] == 3
    assert verify_multirun_truth_gate(
        live_summary, min_iterations=3, min_ctr=0.35, require_live=True
    )


def test_run_multirun_arm_resume_checkpoint_and_402_stop(monkeypatch, tmp_path: Path):
    """Verify arm-level resume skips finished arms, writes checkpoints, checks OpenRouter credits, and handles 402."""
    import json
    from types import SimpleNamespace

    import pytest

    from benchmarks import run_multirun_benchmark as rmb
    from benchmarks.run_3arm_benchmark import HardenedLiveExecutor

    monkeypatch.setattr(rmb.time, "sleep", lambda _s: None)

    # 1. Pre-flight OpenRouter zero-credit guard rejects paid models immediately
    monkeypatch.setattr(rmb, "check_openrouter_credits", lambda _k: 0.0)
    with pytest.raises(rmb.InsufficientCreditsError, match=r"0\.0000 remaining paid credits"):
        rmb.run_multirun_benchmark(
            iterations=2,
            workload_key="data_agent",
            turns=2,
            output_dir=str(tmp_path / "or_blocked"),
            backend="live",
            live_provider="openrouter",
            model="anthropic/claude-haiku-4.5",
            api_key="sk-or-test",
        )

    # 2. Seed partial iteration 2 (only 2/4 arms completed) + 1 incomplete turn row to test arm-level resume
    resume_dir = tmp_path / "arm_resume"
    resume_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for it in (1, 2):
        arms_to_seed = (
            ("raw_baseline", "naive_provider", "cachealign_direct", "cachealign_litellm")
            if it == 1
            else ("raw_baseline", "naive_provider")
        )
        for arm in arms_to_seed:
            for turn in (1, 2):
                cached = 4000 if (arm.startswith("cachealign") and turn == 2) else 0
                seed_rows.append(
                    {
                        "workload": "data_agent",
                        "arm": arm,
                        "turn": turn,
                        "total_input_tokens": 5000,
                        "cached_tokens": cached,
                        "cache_write_tokens": 0,
                        "uncached_tokens": 5000 - cached,
                        "cost_usd": 0.0,
                        "ttft_ms": 500.0 + it * 10 + turn,
                        "is_hit": cached > 0,
                        "prefix_hash": "abc",
                        "iteration": it,
                        "provider": "nvidia_nim",
                    }
                )
    # Add 1 orphan incomplete turn row for iter 2 cachealign_direct turn 1 (should be dropped cleanly)
    seed_rows.append({**seed_rows[-1], "arm": "cachealign_direct", "turn": 1, "iteration": 2})
    log_path = resume_dir / "multirun_live_logs.jsonl"
    log_path.write_text("\n".join(json.dumps(r) for r in seed_rows) + "\n", encoding="utf-8")

    exec_calls: list[str] = []
    checkpoints: list[int] = []

    class _QuotaExecutor(HardenedLiveExecutor):
        def __init__(self, api_key: str, provider: str = "nvidia_nim", model: str | None = None):
            super().__init__(api_key=api_key, provider=provider, model=model)
            self._idx = 0
            parent = self

            class _Comp:
                @staticmethod
                def create(**kwargs):
                    parent._idx += 1
                    exec_calls.append("call")
                    if parent._idx > 4:
                        err = RuntimeError("HTTP Error 402: Payment Required")
                        err.status_code = 402
                        raise err
                    usage = SimpleNamespace(
                        prompt_tokens=5000,
                        completion_tokens=10,
                        prompt_tokens_details=SimpleNamespace(cached_tokens=4000),
                    )
                    if kwargs.get("stream"):
                        return iter(
                            [
                                SimpleNamespace(
                                    id="c",
                                    choices=[
                                        SimpleNamespace(
                                            delta=SimpleNamespace(content="ok", tool_calls=None)
                                        )
                                    ],
                                    usage=None,
                                ),
                                SimpleNamespace(id="r", choices=[], usage=usage),
                            ]
                        )
                    return SimpleNamespace(id="r", usage=usage)

            self.client = SimpleNamespace(chat=SimpleNamespace(completions=_Comp()))

    orig_write = rmb._write_summary_artifact

    def _spy_write(**kwargs):
        if kwargs.get("status") == "checkpoint":
            checkpoints.append(len(kwargs["raw_results_matrix"]))
        return orig_write(**kwargs)

    monkeypatch.setattr(rmb, "HardenedLiveExecutor", _QuotaExecutor)
    monkeypatch.setattr(rmb, "_write_summary_artifact", _spy_write)

    res = rmb.run_multirun_benchmark(
        iterations=4,
        workload_key="data_agent",
        turns=2,
        output_dir=str(resume_dir),
        backend="live",
        live_provider="nvidia_nim",
        api_key="nvapi-unit",
        resume=True,
        checkpoint_every=2,
    )

    # Iter 1 reused completely, iter 2 ran only remaining 2 arms (4 calls) before iter 3 hit 402.
    # process_turn() first attempts a streaming call; on 402 RuntimeError it falls into a non-streaming
    # fallback which raises again — so the first turn of iter 3 makes 2 create() calls (calls 5+6).
    assert len(exec_calls) == 6
    assert checkpoints == [8]
    assert res["status"] == "partial_stopped"
    assert res["completed_iterations_n"] == 2
    assert res["iterations_n"] == 2
    assert "402" in (res["stop_reason"] or "")
    # Verify truth gate enforces min_iterations=4 against partial 2-iteration result
    with pytest.raises(AssertionError, match="below required minimum"):
        verify_multirun_truth_gate(
            resume_dir / "multirun_live_summary.json",
            min_iterations=4,
            min_ctr=0.35,
            require_live=True,
        )
