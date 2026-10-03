"""
Unit tests for WP-1.1 Core Hardening & Security Safeguards.
Tests schema recursion bombs, instruction stripping protection,
tenant isolation salts, fail-open resilience, and message stream context managers.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cachealign.adapters.anthropic import AnthropicAdapter
from cachealign.adapters.openai import OpenAIAdapter
from cachealign.normalizers.partitioner import (
    extract_volatile_elements,
    partition_messages_and_system,
)
from cachealign.normalizers.schema import canonicalize_object
from cachealign.wrappers.client import wrap


def test_schema_recursion_guard():
    """Verify circular references and excessive nesting trigger ValueError without crashing."""
    # 1. Circular reference test
    circular_dict = {}
    circular_dict["self"] = circular_dict
    with pytest.raises(ValueError, match="Circular reference detected"):
        canonicalize_object(circular_dict)

    # 2. Deeply nested dictionary test (> 20 levels)
    deep_dict = {}
    curr = deep_dict
    for _ in range(25):
        curr["nested"] = {}
        curr = curr["nested"]

    with pytest.raises(ValueError, match="Schema recursion limit exceeded"):
        canonicalize_object(deep_dict)


def test_instruction_stripping_protection():
    """Verify natural language instructions with temporal keywords are NOT stripped."""
    system_prompt = (
        "You are an enterprise auditing assistant.\n"
        "Do not disclose the timestamp of this audit under any circumstance.\n"
        "Ensure all actions taken now are compliant with policy 402.\n"
        "Timestamp: 2026-10-03 14:00:00\n"
        "Session-ID: sess_prod_8923\n"
        "Always verify that timestamp format is ISO-8601."
    )

    clean_text, extracted = extract_volatile_elements(system_prompt)

    # Instructional lines must remain completely untouched
    assert "Do not disclose the timestamp of this audit under any circumstance." in clean_text
    assert "Ensure all actions taken now are compliant with policy 402." in clean_text
    assert "Always verify that timestamp format is ISO-8601." in clean_text

    # Volatile header lines must be extracted
    assert "Timestamp: 2026-10-03 14:00:00" not in clean_text
    assert "Session-ID: sess_prod_8923" not in clean_text
    assert len(extracted) == 2


def test_tenant_salt_isolation():
    """Verify tenant_id injects deterministic cryptographic salt to prevent CacheProbe timing attacks."""
    adapter_tenant_a = AnthropicAdapter(tenant_id="tenant_alpha")
    adapter_tenant_b = AnthropicAdapter(tenant_id="tenant_beta")

    kwargs = {
        "model": "claude-3-5-sonnet",
        "system": "Base system instructions.",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    res_a = adapter_tenant_a.optimize_request(kwargs)
    res_b = adapter_tenant_b.optimize_request(kwargs)

    sys_a = res_a.optimized_kwargs["system"][0]["text"]
    sys_b = res_b.optimized_kwargs["system"][0]["text"]

    assert "[CacheTenancy:" in sys_a
    assert "[CacheTenancy:" in sys_b
    assert sys_a != sys_b  # Different tenants must produce different prefix hashes

    # Verify OpenAI adapter also injects tenant salt
    openai_adapter = OpenAIAdapter(tenant_id="tenant_alpha")
    res_o = openai_adapter.optimize_request(kwargs)
    assert "[CacheTenancy:" in res_o.optimized_kwargs["messages"][0]["content"]


def test_tool_result_tail_migration_safety():
    """Verify tail migration does NOT corrupt tool response message structures."""
    # Test OpenAI tool message (role: tool)
    system = "System prompt.\nCurrent Time: 2026-10-03 12:00:00"
    messages = [
        {"role": "user", "content": "Check weather in Tokyo"},
        {"role": "assistant", "content": "Calling tool..."},
        {"role": "tool", "content": '{"temperature": 22, "condition": "sunny"}'},
    ]

    _clean_sys, new_msgs, _extracted = partition_messages_and_system(system, messages)

    # Tool message content must remain pristine JSON string
    assert new_msgs[2]["content"] == '{"temperature": 22, "condition": "sunny"}'
    # Migrated context must be attached to the preceding user turn
    assert "<cachealign_ephemeral_context>" in new_msgs[0]["content"]


def test_fail_open_reliability():
    """Verify that an unexpected optimization error falls open gracefully without crashing."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=100, output_tokens=50)
    )

    wrapped = wrap(mock_client, fail_open=True, verbose=False)

    # Pass an unoptimizable object that could cause an error if not guarded
    # (Here we mock optimize_request to raise an exception)
    wrapped.messages._adapter.optimize_request = MagicMock(
        side_effect=RuntimeError("Unexpected bug")
    )

    # The create call must NOT raise RuntimeError; it must fall open and call raw client
    resp = wrapped.messages.create(
        model="claude-3-5-sonnet",
        system="Hello",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert resp.usage.input_tokens == 100
    assert mock_client.messages.create.called


def test_anthropic_message_stream_manager():
    """Verify Anthropic client.messages.stream context manager pattern."""

    class MockStream:
        def __init__(self):
            self.text_stream = ["Hello", " world", "!"]

        def get_final_message(self):
            return SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=1000,
                    output_tokens=20,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=800,
                )
            )

    class MockStreamManager:
        def __init__(self):
            self.stream = MockStream()

        def __enter__(self):
            return self.stream

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = MockStreamManager()

    wrapped = wrap(mock_client, verbose=False)

    with wrapped.messages.stream(
        model="claude-3-5-sonnet",
        system="System prompt",
        messages=[{"role": "user", "content": "Hi"}],
    ) as stream:
        chunks = list(stream.text_stream)

    assert chunks == ["Hello", " world", "!"]
    # Telemetry should be captured upon __exit__
    assert wrapped.telemetry.session.turns_count == 1
    assert wrapped.telemetry.session.total_cached_tokens == 800
