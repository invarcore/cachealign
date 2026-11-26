# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""Unit tests for cache_probe_runner and daily-smoke probe mode."""

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cachealign.cache_probe_runner import (
    build_aligned_prefix,
    main,
    parse_usage_fields,
    run_probe_sequence,
    send_probe_request,
)
from cachealign.cli.main import cli


def test_build_aligned_prefix_deterministic():
    prefix1 = build_aligned_prefix(1200)
    prefix2 = build_aligned_prefix(1200)
    assert prefix1 == prefix2
    assert "[BLK:0000|DIGEST:" in prefix1
    assert len(prefix1.splitlines()) == 1200 // 16


def test_send_probe_request_anthropic_and_openai():
    # Mock Anthropic HTTP success
    mock_body = {
        "id": "msg_123",
        "usage": {
            "input_tokens": 50,
            "cache_creation_input_tokens": 1200,
            "cache_read_input_tokens": 0,
        },
    }
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(mock_body).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = send_probe_request(
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            api_format="anthropic",
            model="claude-sonnet-4-6",
            prefix_text="sample",
            ttl_hint="1h",
        )
        assert res["status"] == 200
        assert res["body"]["usage"]["cache_creation_input_tokens"] == 1200

    # Mock OpenAI HTTP success
    mock_body_openai = {
        "id": "chatcmpl_123",
        "usage": {
            "prompt_tokens": 1250,
            "prompt_tokens_details": {"cached_tokens": 1200},
        },
    }
    mock_resp_openai = MagicMock()
    mock_resp_openai.status = 200
    mock_resp_openai.read.return_value = json.dumps(mock_body_openai).encode("utf-8")
    mock_resp_openai.__enter__.return_value = mock_resp_openai

    with patch("urllib.request.urlopen", return_value=mock_resp_openai):
        res2 = send_probe_request(
            base_url="https://api.openai.com",
            api_key="sk-openai-test",
            api_format="openai",
            model="gpt-4o",
            prefix_text="sample",
            ttl_hint="24h",
        )
        assert res2["status"] == 200
        assert res2["body"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 1200


def test_send_probe_request_errors():
    # HTTPError
    err = urllib.error.HTTPError(
        url="http://test",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=io.BytesIO(b'{"error": "invalid_api_key"}'),
    )
    with patch("urllib.request.urlopen", side_effect=err):
        res = send_probe_request("https://api.openai.com", "key", "openai", "gpt-4o", "text")
        assert res["status"] == 401
        assert res["body"] == {"error": "invalid_api_key"}

    # General Exception
    with patch("urllib.request.urlopen", side_effect=ConnectionResetError("Reset")):
        res2 = send_probe_request("https://api.openai.com", "key", "openai", "gpt-4o", "text")
        assert res2["status"] == 0
        assert "Reset" in res2["body"]["error"]


def test_parse_usage_fields_variants():
    # Anthropic
    anth_res = parse_usage_fields(
        "anthropic",
        {
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 500,
            }
        },
    )
    assert anth_res["prompt_tokens"] == 1600
    assert anth_res["cache_write_tokens"] == 1000
    assert anth_res["cache_read_tokens"] == 500
    assert anth_res["usage_field_present"] is True

    # OpenAI details format
    openai_res = parse_usage_fields(
        "openai",
        {
            "usage": {
                "prompt_tokens": 1200,
                "prompt_tokens_details": {"cached_tokens": 1000},
            }
        },
    )
    assert openai_res["prompt_tokens"] == 1200
    assert openai_res["cache_read_tokens"] == 1000
    assert openai_res["usage_field_present"] is True

    # OpenAI fallback prompt_cache_hit_tokens
    openai_fallback = parse_usage_fields(
        "openai",
        {
            "usage": {
                "prompt_tokens": 1200,
                "prompt_cache_hit_tokens": 800,
                "prompt_cache_miss_tokens": 400,
            }
        },
    )
    assert openai_fallback["cache_read_tokens"] == 800
    assert openai_fallback["cache_write_tokens"] == 400

    # Missing fields
    missing_res = parse_usage_fields("openai", {})
    assert missing_res["usage_field_present"] is False


def test_run_probe_sequence_daily_smoke_mode():
    delays: list[float] = []

    def mock_send(base_url, api_key, api_format, model, prefix_text):
        call_idx = getattr(mock_send, "calls", 0)
        mock_send.calls = call_idx + 1
        return {
            "status": 200,
            "latency_ms": 15.0,
            "body": {
                "usage": {
                    "input_tokens": 50,
                    "cache_creation_input_tokens": 1200 if call_idx == 0 else 0,
                    "cache_read_input_tokens": 0 if call_idx == 0 else 1200,
                }
            },
        }

    obs = run_probe_sequence(
        route_name="test-route",
        base_url="https://api.anthropic.com",
        api_key="valid-key",
        api_format="anthropic",
        model="claude-sonnet-4-6",
        target_tokens=1200,
        mode="daily-smoke",
        send_fn=mock_send,
        sleep_fn=delays.append,
    )

    # daily-smoke MUST only run cold_seed + warm_10s
    assert len(obs) == 2
    assert obs[0].probe_phase == "cold_seed"
    assert obs[0].delay_seconds == 0.0
    assert obs[1].probe_phase == "warm_10s"
    assert obs[1].delay_seconds == 10.0
    assert delays == [10.0]
    assert all(len(o.invariants_failed) == 0 for o in obs)


def test_run_probe_sequence_full_mode_and_invariants():
    calls = []

    def mock_send_with_drop(base_url, api_key, api_format, model, prefix_text):
        phase_idx = len(calls)
        calls.append(phase_idx)
        # Drop cache on warm_10s, affinity, and ttl_boundary to trigger invariant failures
        read_tokens = 0 if phase_idx in (1, 2, 4) else 1000
        return {
            "status": 200,
            "latency_ms": 10.0,
            "body": {
                "usage": {
                    "input_tokens": 50,
                    "cache_creation_input_tokens": 1000,
                    "cache_read_input_tokens": read_tokens,
                }
            },
        }

    obs = run_probe_sequence(
        route_name="full-route",
        base_url="https://api.anthropic.com",
        api_key="valid-key",
        api_format="anthropic",
        model="claude-sonnet-4-6",
        target_tokens=1200,
        ttl_check_seconds=65,
        mode="full",
        send_fn=mock_send_with_drop,
        sleep_fn=lambda _: None,
    )

    # full mode with ttl_check_seconds runs all 5 phases
    assert len(obs) == 5
    assert [o.probe_phase for o in obs] == [
        "cold_seed",
        "warm_10s",
        "affinity_2",
        "affinity_3",
        "ttl_boundary",
    ]
    assert "P1_ZERO_WARM_CACHE_ENGAGEMENT" in obs[1].invariants_failed
    assert "P5_ROUTE_AFFINITY_COLD_DROP" in obs[2].invariants_failed
    assert "P2_EARLY_TTL_EVICTION" in obs[4].invariants_failed


def test_run_probe_sequence_validations():
    with pytest.raises(ValueError, match="api_key must be non-empty"):
        run_probe_sequence(
            route_name="r",
            base_url="u",
            api_key="   ",
            api_format="anthropic",
            model="m",
        )

    with pytest.raises(ValueError, match="Unknown mode 'invalid'"):
        run_probe_sequence(
            route_name="r",
            base_url="u",
            api_key="valid-key",
            api_format="anthropic",
            model="m",
            mode="invalid",
        )


def test_main_cli_execution_daily_smoke(tmp_path, monkeypatch):
    out_file = tmp_path / "smoke_ledger.jsonl"
    monkeypatch.setenv("LLM_API_KEY", "test-live-key")

    mock_send = MagicMock(
        return_value={
            "status": 200,
            "latency_ms": 25.0,
            "body": {
                "usage": {
                    "input_tokens": 100,
                    "cache_creation_input_tokens": 1200,
                    "cache_read_input_tokens": 1200,
                }
            },
        }
    )

    with patch("cachealign.cache_probe_runner.send_probe_request", mock_send):
        with patch("cachealign.cache_probe_runner.time.sleep", lambda _: None):
            main(
                [
                    "--route-name",
                    "smoke-test",
                    "--api-format",
                    "anthropic",
                    "--model",
                    "claude-sonnet-4-6",
                    "--output-jsonl",
                    str(out_file),
                    "--mode",
                    "daily-smoke",
                ]
            )

    assert out_file.exists()
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    assert lines[0]["probe_phase"] == "cold_seed"
    assert lines[1]["probe_phase"] == "warm_10s"


def test_main_cli_missing_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="Error: set LLM_API_KEY"):
        main([])


def test_cachealign_probe_cli_daily_smoke_mode(tmp_path):
    # Verify click CLI `cachealign probe -m daily-smoke`
    runner = CliRunner()
    result = runner.invoke(cli, ["probe", "--provider", "anthropic", "--mode", "daily-smoke"])
    assert result.exit_code == 0
    assert "ALL PROBES PASSED" in result.output
    assert "threshold_boundary" in result.output
    assert "accounting_integrity" in result.output
    # Vectors 2 and 4 should not be in daily-smoke output
    assert "retention_and_latency" not in result.output
    assert "route_parity" not in result.output

    # With output file
    rep_file = tmp_path / "probe_daily_report.json"
    result_out = runner.invoke(
        cli,
        [
            "probe",
            "--provider",
            "anthropic",
            "--mode",
            "daily-smoke",
            "--output",
            str(rep_file),
        ],
    )
    assert result_out.exit_code == 0
    assert rep_file.exists()
    assert "[LEDGER_ANCHOR]" in result_out.output
