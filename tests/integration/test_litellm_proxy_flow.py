# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
Integration tests for LiteLLM Proxy CustomLogger interception flow.
Verifies that CacheAlignLiteLLMHandler stabilizes multi-turn prompt prefixes,
canonicalizes shuffled tool schemas, migrates volatile metadata, and captures FinOps metrics.
"""

from types import SimpleNamespace
from typing import Any

from cachealign.integrations.litellm import CacheAlignLiteLLMHandler


def test_litellm_proxy_multi_turn_prefix_stability():
    """
    Simulates a 2-turn agent session mediated by a LiteLLM Proxy.
    Verifies that despite tool shuffling and dynamic timestamps,
    the resulting system prompt and tool list are byte-identical across turns.
    """
    handler = CacheAlignLiteLLMHandler(verbose=False)

    # Turn 1: Tool ordering [tool_z, tool_a, tool_m] + timestamp at token 0
    turn_1_payload: dict[str, Any] = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "system",
                "content": "Current Time: 2026-11-26 10:00:00\nYou are an enterprise data analyst.",
            },
            {"role": "user", "content": "Analyze Q3 pipeline."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "tool_z",
                    "description": "Z tool",
                    "parameters": {"properties": {"b": {"type": "string"}, "a": {"type": "int"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "description": "A tool",
                    "parameters": {"properties": {"x": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_m",
                    "description": "M tool",
                    "parameters": {"properties": {"y": {"type": "string"}}},
                },
            },
        ],
    }

    # Turn 2: Tools shuffled [tool_m, tool_z, tool_a] + advanced timestamp
    turn_2_payload: dict[str, Any] = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "system",
                "content": "Current Time: 2026-11-26 10:00:15\nYou are an enterprise data analyst.",
            },
            {"role": "user", "content": "Analyze Q3 pipeline."},
            {"role": "assistant", "content": "Calling tool_a..."},
            {"role": "user", "content": "Tool output: success."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "tool_m",
                    "description": "M tool",
                    "parameters": {"properties": {"y": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_z",
                    "description": "Z tool",
                    "parameters": {"properties": {"a": {"type": "int"}, "b": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "description": "A tool",
                    "parameters": {"properties": {"x": {"type": "string"}}},
                },
            },
        ],
    }

    # Execute synchronous pre_call_hook as LiteLLM Proxy does
    out_1 = handler.pre_call_hook({}, None, turn_1_payload, "completion")
    out_2 = handler.pre_call_hook({}, None, turn_2_payload, "completion")

    # Invariant 1: System prompt prefix is 100% byte-identical across turns
    sys_1 = out_1["messages"][0]["content"]
    sys_2 = out_2["messages"][0]["content"]
    assert sys_1 == sys_2
    assert "Current Time:" not in sys_1
    assert "You are an enterprise data analyst." in sys_1

    # Invariant 2: Dynamic timestamps were migrated to the ephemeral tail
    assert "Current Time: 2026-11-26 10:00:00" in out_1["messages"][1]["content"]
    assert "Current Time: 2026-11-26 10:00:15" in out_2["messages"][-1]["content"]

    # Invariant 3: Tools are canonically sorted in identical order [tool_a, tool_m, tool_z]
    tool_names_1 = [t["function"]["name"] for t in out_1["tools"]]
    tool_names_2 = [t["function"]["name"] for t in out_2["tools"]]
    assert tool_names_1 == ["tool_a", "tool_m", "tool_z"]
    assert tool_names_2 == ["tool_a", "tool_m", "tool_z"]
    assert out_1["tools"] == out_2["tools"]


def test_litellm_proxy_sync_and_async_telemetry():
    """Verifies that both sync and async log_success_event record FinOps stats cleanly."""
    handler = CacheAlignLiteLLMHandler(verbose=False)

    fake_response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=5000,
            completion_tokens=200,
            prompt_tokens_details=SimpleNamespace(cached_tokens=4000),
        )
    )

    # Sync log_success_event
    handler.log_success_event({"model": "gpt-4o"}, fake_response, 0, 1)
    assert handler.reporter.session.turns_count == 1
    assert handler.reporter.session.total_cached_tokens == 4000
    assert handler.reporter.session.overall_hit_rate == 76.92  # 4000 / 5200 = 76.92%

    # Async log_success_event
    handler.async_log_success_event({"model": "gpt-4o"}, fake_response, 0, 1)
    assert handler.reporter.session.turns_count == 2
    assert handler.reporter.session.total_cached_tokens == 8000
