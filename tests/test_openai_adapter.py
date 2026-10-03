from types import SimpleNamespace

from cachealign.adapters.openai import OpenAIAdapter


def test_openai_adapter_partition_and_tools():
    adapter = OpenAIAdapter()
    kwargs = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a bot.\nCurrent Time: 2026-10-03 14:00:00\nExecute commands.",
            },
            {"role": "user", "content": "Status report?"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "z_func", "parameters": {"y": 1, "x": 2}}},
            {"type": "function", "function": {"name": "a_func", "parameters": {"b": 1, "a": 2}}},
        ],
    }

    res = adapter.optimize_request(kwargs)
    opt = res.optimized_kwargs

    # Check tools sorted alphabetically by function name
    assert opt["tools"][0]["function"]["name"] == "a_func"
    assert opt["tools"][1]["function"]["name"] == "z_func"
    assert list(opt["tools"][0]["function"]["parameters"].keys()) == ["a", "b"]

    # Check system prompt cleaned
    assert "Current Time:" not in opt["messages"][0]["content"]
    assert "You are a bot." in opt["messages"][0]["content"]

    # Check tail migration
    assert "Current Time:" in opt["messages"][1]["content"]


def test_openai_adapter_parse_usage():
    adapter = OpenAIAdapter()
    mock_response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=8000,
            completion_tokens=200,
            total_tokens=8200,
            prompt_tokens_details=SimpleNamespace(cached_tokens=6000),
        )
    )

    stats = adapter.parse_usage(mock_response, "gpt-4o")
    assert stats.cached_tokens == 6000
    assert stats.cache_hit_rate == 75.0
    assert stats.estimated_cost_saved_usd > 0.005
