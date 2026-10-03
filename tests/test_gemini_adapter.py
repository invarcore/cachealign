"""
Unit tests for Google Gemini Adapter and Client Wrapper.
Tests system instruction partitioning, tool schema canonicalization,
CachedContent TTL registry, streaming, and sync/async client wrapping.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cachealign.adapters.gemini import GeminiAdapter
from cachealign.wrappers.client import wrap


def test_gemini_adapter_optimization():
    adapter = GeminiAdapter()

    kwargs = {
        "model": "gemini-2.5-flash",
        "contents": "Explain quantum computing.",
        "config": {
            "system_instruction": "You are a physics professor.\nCurrent Time: 2026-10-03 12:00:00",
            "tools": [
                {
                    "name": "search_arxiv",
                    "description": "Search papers",
                    "parameters": {"type": "object"},
                },
                {
                    "name": "calculate",
                    "description": "Calculate math",
                    "parameters": {"type": "object"},
                },
            ],
        },
    }

    res = adapter.optimize_request(kwargs)
    opt = res.optimized_kwargs

    # Check system instruction partitioning
    clean_sys = opt["config"]["system_instruction"]
    assert "Current Time:" not in clean_sys
    assert "You are a physics professor." in clean_sys

    # Check tools canonicalized alphabetically
    tools = opt["config"]["tools"]
    assert tools[0]["name"] == "calculate"
    assert tools[1]["name"] == "search_arxiv"

    # Check volatile timestamp migrated to contents tail
    contents = opt["contents"]
    assert "Current Time: 2026-10-03 12:00:00" in contents
    assert "<cachealign_ephemeral_context>" in contents


def test_gemini_adapter_parse_usage():
    adapter = GeminiAdapter()

    mock_resp = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=10000,
            candidates_token_count=500,
            total_token_count=10500,
            cached_content_token_count=8000,
        )
    )

    stats = adapter.parse_usage(mock_resp, "gemini-2.5-flash")
    assert stats.cached_tokens == 8000
    assert stats.cache_hit_rate == 80.0
    assert stats.estimated_cost_saved_usd > 0.0

    # Test Pro model pricing
    stats_pro = adapter.parse_usage(mock_resp, "gemini-2.5-pro")
    assert stats_pro.estimated_cost_saved_usd > stats.estimated_cost_saved_usd


def test_gemini_cache_registry_and_ttl():
    adapter = GeminiAdapter()
    fingerprint = adapter.compute_prefix_fingerprint(
        system_instruction="Static base prompt",
        tools=[{"name": "tool_a"}],
    )

    # Register active CachedContent resource
    adapter.register_cached_content(
        fingerprint=fingerprint,
        name="cachedContents/alpha123",
        model="gemini-2.5-flash",
        token_count=5000,
        ttl_seconds=3600,
    )

    # Verify retrieval
    cached_name = adapter.get_valid_cached_content(fingerprint)
    assert cached_name == "cachedContents/alpha123"

    # Verify optimization binds cached_content
    kwargs = {
        "model": "gemini-2.5-flash",
        "contents": "User question",
        "config": {
            "system_instruction": "Static base prompt",
            "tools": [{"name": "tool_a"}],
        },
    }
    res = adapter.optimize_request(kwargs)
    assert res.optimized_kwargs["config"]["cached_content"] == "cachedContents/alpha123"
    assert res.breakpoints_injected == 1


def test_wrap_gemini_client():
    mock_client = MagicMock()
    # Ensure it doesn't match Anthropic
    del mock_client.messages

    mock_resp = SimpleNamespace(
        text="Quantum computing utilizes qubits.",
        usage_metadata=SimpleNamespace(
            prompt_token_count=5000,
            candidates_token_count=200,
            total_token_count=5200,
            cached_content_token_count=4000,
        ),
    )
    mock_client.models.generate_content.return_value = mock_resp

    wrapped = wrap(mock_client, verbose=False)

    resp = wrapped.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello",
        config={"system_instruction": "System\nCurrent Time: 2026-10-03"},
    )

    assert resp.text == "Quantum computing utilizes qubits."
    assert wrapped.telemetry.session.turns_count == 1
    assert wrapped.telemetry.session.total_cached_tokens == 4000
    assert mock_client.models.generate_content.called


@pytest.mark.anyio
async def test_wrap_async_gemini_client():
    mock_client = MagicMock()
    del mock_client.messages

    mock_resp = SimpleNamespace(
        text="Async quantum response.",
        usage_metadata=SimpleNamespace(
            prompt_token_count=4000,
            candidates_token_count=100,
            total_token_count=4100,
            cached_content_token_count=3500,
        ),
    )
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    wrapped = wrap(mock_client, verbose=False)

    resp = await wrapped.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello async",
    )

    assert resp.text == "Async quantum response."
    assert wrapped.telemetry.session.turns_count == 1
    assert wrapped.telemetry.session.total_cached_tokens == 3500


def test_wrap_gemini_streaming():
    mock_client = MagicMock()
    del mock_client.messages

    chunks = [
        SimpleNamespace(text="Chunk 1", usage_metadata=None),
        SimpleNamespace(
            text="Chunk 2",
            usage_metadata=SimpleNamespace(
                prompt_token_count=3000,
                candidates_token_count=50,
                total_token_count=3050,
                cached_content_token_count=2500,
            ),
        ),
    ]
    mock_client.models.generate_content_stream.return_value = iter(chunks)

    wrapped = wrap(mock_client, verbose=False)

    stream = wrapped.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents="Stream prompt",
    )

    accumulated = []
    for chunk in stream:
        accumulated.append(chunk.text)

    assert accumulated == ["Chunk 1", "Chunk 2"]
    assert wrapped.telemetry.session.turns_count == 1
    assert wrapped.telemetry.session.total_cached_tokens == 2500
