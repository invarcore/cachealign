# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
Closed-Tokenizer API Guardrail Verification.
Proves that CacheAlign performs RFC 8785 deterministic JSON key sorting and
volatile tail migration without injecting synthetic block-boundary padding tokens
on closed-tokenizer APIs (Anthropic, OpenAI, AWS Bedrock), preserving exact byte lengths
and preventing off-by-one PagedAttention block evictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from cachealign.adapters.anthropic import AnthropicAdapter
from cachealign.adapters.openai import OpenAIAdapter
from cachealign.normalizers.schema import canonicalize_tool_schemas
from cachealign.wrappers.client import wrap
from observatory.run_daily_observatory import (
    main as observatory_main,
)
from observatory.run_daily_observatory import (
    resolve_api_key,
    run_provider_smoke_probes,
)


def test_anthropic_closed_tokenizer_zero_synthetic_padding():
    """Verify AnthropicAdapter does not inflate or pad static system prompts."""
    adapter = AnthropicAdapter(min_token_threshold=0)
    static_prompt = (
        "You are an enterprise AI security analyst. "
        "Your task is to review RFC 8693 token exchange flows and report delegation vulnerabilities. "
        "Ensure all assertions adhere to OAuth 2.0 specifications."
    )
    original_bytes = len(static_prompt.encode("utf-8"))

    raw_kwargs = {
        "model": "claude-3-5-sonnet-20241022",
        "system": static_prompt,
        "messages": [{"role": "user", "content": "Audit this request."}],
    }

    res = adapter.optimize_request(raw_kwargs)
    opt_system = res.optimized_kwargs["system"]

    # When converted to structured cache_control blocks, the text inside the block must be byte-identical
    assert isinstance(opt_system, list)
    assert len(opt_system) == 1
    text_content = opt_system[0]["text"]
    assert text_content == static_prompt
    assert len(text_content.encode("utf-8")) == original_bytes
    # No synthetic block-boundary whitespace or repetitive padding tokens
    assert not text_content.endswith("   ")
    assert "PAD" not in text_content
    assert opt_system[0]["cache_control"] == {"type": "ephemeral"}


def test_openai_closed_tokenizer_zero_synthetic_padding():
    """Verify OpenAIAdapter preserves static prefix without synthetic padding."""
    adapter = OpenAIAdapter()
    static_prompt = "You are a specialized code grounding auditor verifying AST symbols."
    original_bytes = len(static_prompt.encode("utf-8"))

    raw_kwargs = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": static_prompt},
            {"role": "user", "content": "Analyze lib/rand.c"},
        ],
    }

    res = adapter.optimize_request(raw_kwargs)
    messages = res.optimized_kwargs["messages"]
    system_msg = messages[0]
    assert system_msg["role"] == "system"
    assert system_msg["content"] == static_prompt
    assert len(system_msg["content"].encode("utf-8")) == original_bytes


def test_rfc8785_canonical_tool_sorting_no_padding():
    """Verify RFC 8785 schema canonicalization sorts keys deterministically without padding."""
    tools = [
        {
            "name": "lookup_symbol",
            "description": "Lookup AST symbol",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol_name": {"type": "string"},
                    "file_path": {"type": "string"},
                    "line_hint": {"type": "integer"},
                },
                "required": ["file_path", "symbol_name"],
            },
        }
    ]

    canonical = canonicalize_tool_schemas(tools)
    serialized = json.dumps(canonical)
    # Ensure no padding or trailing dummy fields
    assert "padding" not in serialized
    assert "__pad__" not in serialized
    # Verify deterministic sorting: properties should be ordered: file_path, line_hint, symbol_name
    props = list(canonical[0]["input_schema"]["properties"].keys())
    assert props == ["file_path", "line_hint", "symbol_name"]


def test_volatile_tail_migration_preserves_static_prefix():
    """Verify volatile headers are migrated to tail and static prefix is left clean."""
    adapter = AnthropicAdapter(min_token_threshold=0)
    system_with_volatile = (
        "Timestamp: 2026-11-17T21:00:00Z\n"
        "Session-ID: sess_998877\n"
        "You are an infallible cryptographic verifier."
    )
    expected_static = "You are an infallible cryptographic verifier."

    raw_kwargs = {
        "model": "claude-3-5-sonnet-20241022",
        "system": system_with_volatile,
        "messages": [{"role": "user", "content": "Verify signature."}],
    }

    res = adapter.optimize_request(raw_kwargs)
    opt_system = res.optimized_kwargs["system"]
    assert isinstance(opt_system, list)
    assert opt_system[0]["text"] == expected_static
    assert len(opt_system[0]["text"].encode("utf-8")) == len(expected_static.encode("utf-8"))

    # Volatile variables migrated to user turn inside XML container
    user_content = res.optimized_kwargs["messages"][-1]["content"]
    assert "<cachealign_ephemeral_context>" in user_content
    assert "Timestamp: 2026-11-17T21:00:00Z" in user_content


def test_wrap_client_closed_tokenizer_guarantee():
    """Verify cachealign.wrap() intercepts calls and preserves zero-padding invariant."""
    from types import SimpleNamespace

    mock_usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    mock_raw_client = MagicMock()
    mock_raw_client.messages.create.return_value = SimpleNamespace(usage=mock_usage)

    wrapped_client = wrap(mock_raw_client)
    static_system = "Audit system instructions with zero padding."

    wrapped_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=static_system,
        messages=[{"role": "user", "content": "Execute."}],
    )

    call_kwargs = mock_raw_client.messages.create.call_args.kwargs
    sent_system = call_kwargs["system"]
    assert isinstance(sent_system, list)
    assert sent_system[0]["text"] == static_system
    assert len(sent_system[0]["text"].encode("utf-8")) == len(static_system.encode("utf-8"))


def test_observatory_helpers_and_skip_logic(tmp_path: Path):
    """Verify observatory skips unconfigured providers cleanly with exit code 0."""
    # Test resolve_api_key
    assert resolve_api_key(["NON_EXISTENT_KEY_12345"]) is None

    # Test run_provider_smoke_probes with no environment variables configured
    test_jsonl = tmp_path / "test_history.jsonl"
    observations = run_provider_smoke_probes(
        mode="daily-smoke",
        output_jsonl=test_jsonl,
    )
    # Should cleanly skip without error
    assert isinstance(observations, list)

    # Test main CLI entrypoint returns 0
    rc = observatory_main(["--mode", "daily-smoke", "--output-jsonl", str(test_jsonl)])
    assert rc == 0


def test_observatory_mock_execution(tmp_path: Path, monkeypatch):
    """Verify observatory execution with mock runner and mocked credentials."""
    from cachealign.cache_probe_runner import ProbeObservation

    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-anthropic-key-12345")
    test_jsonl = tmp_path / "obs_history.jsonl"

    def mock_runner(**kwargs):
        return [
            ProbeObservation(
                timestamp="2026-11-17T21:00:00Z",
                route_name=kwargs.get("route_name", "test"),
                api_format=kwargs.get("api_format", "openai"),
                model=kwargs.get("model", "test-model"),
                probe_phase="cold_seed",
                delay_seconds=0.0,
                http_status=200,
                latency_ms=120.0,
                prompt_tokens=1500,
                cache_write_tokens=1500,
                cache_read_tokens=0,
                usage_field_present=True,
                prefix_sha256="abc123",
                invariants_failed=[],
            )
        ]

    obs = run_provider_smoke_probes(
        mode="daily-smoke",
        output_jsonl=test_jsonl,
        provider_filter="anthropic",
        mock_runner=mock_runner,
    )
    assert len(obs) == 1
    assert test_jsonl.exists()
