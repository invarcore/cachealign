from types import SimpleNamespace

from cachealign.adapters.anthropic import AnthropicAdapter


def test_anthropic_adapter_system_and_tools_optimization():
    adapter = AnthropicAdapter()
    kwargs = {
        "model": "claude-3-5-sonnet-20241022",
        "system": "System instructions for enterprise agent.\nCurrent Time: 2026-10-03 10:00:00",
        "tools": [
            {"name": "b_tool", "description": "tool b", "input_schema": {"type": "object"}},
            {"name": "a_tool", "description": "tool a", "input_schema": {"type": "object"}},
        ],
        "messages": [{"role": "user", "content": "Help me run task."}],
    }

    res = adapter.optimize_request(kwargs)
    opt = res.optimized_kwargs

    # Check system prompt conversion to structured block with cache_control
    assert isinstance(opt["system"], list)
    assert opt["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Current Time:" not in opt["system"][0]["text"]

    # Check tools sorted alphabetically
    assert opt["tools"][0]["name"] == "a_tool"
    assert opt["tools"][1]["name"] == "b_tool"
    # Check cache_control on last tool
    assert opt["tools"][1]["cache_control"] == {"type": "ephemeral"}

    # Check volatile timestamp migrated to user message
    assert "Current Time:" in opt["messages"][-1]["content"]


def test_anthropic_adapter_parse_usage():
    adapter = AnthropicAdapter()
    mock_response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=500,
            output_tokens=150,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=10000,
        )
    )

    stats = adapter.parse_usage(mock_response, "claude-3-5-sonnet")
    assert stats.cached_tokens == 10000
    assert stats.cache_hit_rate > 90.0
    assert stats.estimated_cost_saved_usd > 0.02
