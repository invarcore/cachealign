# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
Unit and Integration Tests for Cache-Probe Regression Monitor.
"""

from click.testing import CliRunner

from cachealign.cli.main import cli
from cachealign.cli.probe import CacheProbeEngine, execute_probe_cli


def test_cache_probe_engine_anthropic():
    engine = CacheProbeEngine(provider="anthropic", route="direct")
    report = engine.run_all_probes()

    assert report.provider == "anthropic"
    assert report.route == "direct"
    assert report.overall_status == "PASS"
    assert len(report.probes) == 4

    vectors = {p.vector: p for p in report.probes}
    assert vectors["threshold_boundary"].status == "PASS"
    assert vectors["threshold_boundary"].metrics["min_expected_threshold"] == 1024

    assert vectors["retention_and_latency"].status == "PASS"
    assert vectors["retention_and_latency"].metrics["prefix_byte_identical"] is True

    assert vectors["accounting_integrity"].status == "PASS"
    assert vectors["accounting_integrity"].metrics["reconciliation_delta"] == 0

    assert vectors["route_parity"].status == "PASS"
    assert vectors["route_parity"].metrics["hashes_match"] is True


def test_cache_probe_engine_openai():
    engine = CacheProbeEngine(provider="openai", route="direct")
    report = engine.run_all_probes()

    assert report.provider == "openai"
    assert report.overall_status == "PASS"
    vectors = {p.vector: p for p in report.probes}
    assert vectors["threshold_boundary"].status == "PASS"
    assert vectors["accounting_integrity"].status == "PASS"


def test_cache_probe_engine_gemini():
    engine = CacheProbeEngine(provider="gemini", route="direct")
    report = engine.run_all_probes()

    assert report.provider == "gemini"
    assert report.overall_status == "PASS"
    vectors = {p.vector: p for p in report.probes}
    assert vectors["threshold_boundary"].status == "PASS"
    assert vectors["threshold_boundary"].metrics["min_expected_threshold"] == 32768


def test_cache_probe_cli_execution():
    runner = CliRunner()
    result = runner.invoke(cli, ["probe", "--provider", "anthropic"])
    assert result.exit_code == 0
    assert "ALL PROBES PASSED" in result.output
    assert "threshold_boundary" in result.output


def test_execute_probe_cli_with_output_file(tmp_path):
    report_file = tmp_path / "probe_report.json"
    code = execute_probe_cli(provider="anthropic", route="direct", output_file=str(report_file))
    assert code == 0
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert '"overall_status": "PASS"' in content
