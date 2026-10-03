"""
Unit tests for Framework Integrations (LangChain / LangGraph & LiteLLM).
"""

from types import SimpleNamespace

from cachealign.integrations.langchain import CacheAlignCallbackHandler
from cachealign.integrations.litellm import CacheAlignLiteLLMHandler


def test_langchain_callback_optimize_messages():
    handler = CacheAlignCallbackHandler(verbose=False)

    messages = [
        {"role": "system", "content": "You are a research bot.\nCurrent Time: 2026-10-03"},
        {"role": "user", "content": "Help me search."},
    ]
    tools = [
        {"name": "z_tool", "description": "z"},
        {"name": "a_tool", "description": "a"},
    ]

    opt_msgs, opt_tools = handler.optimize_chat_messages(messages, tools)

    assert "Current Time:" not in opt_msgs[0]["content"]
    assert "Current Time:" in opt_msgs[1]["content"]
    assert opt_tools[0]["name"] == "a_tool"
    assert opt_tools[1]["name"] == "z_tool"


def test_langchain_callback_on_llm_end():
    handler = CacheAlignCallbackHandler(verbose=False)

    response = SimpleNamespace(
        llm_output={
            "token_usage": {
                "prompt_tokens": 4000,
                "completion_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 3000},
            },
            "model_name": "gpt-4o",
        }
    )

    handler.on_llm_end(response)
    assert handler.reporter.session.turns_count == 1
    assert handler.reporter.session.total_cached_tokens == 3000
    assert handler.reporter.session.overall_hit_rate == 73.17  # 3000 / 4100 = 73.17%


def test_litellm_pre_call_hook():
    handler = CacheAlignLiteLLMHandler(verbose=False)

    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "System prompt.\nSession-ID: sess_123"},
            {"role": "user", "content": "Query"},
        ],
        "tools": [
            {"function": {"name": "tool_b"}},
            {"function": {"name": "tool_a"}},
        ],
    }

    opt_data = handler.async_pre_call_hook({}, None, data, "completion")

    assert "Session-ID:" not in opt_data["messages"][0]["content"]
    assert "Session-ID:" in opt_data["messages"][1]["content"]
    assert opt_data["tools"][0]["function"]["name"] == "tool_a"
    assert opt_data["tools"][1]["function"]["name"] == "tool_b"


def test_litellm_log_success_event():
    handler = CacheAlignLiteLLMHandler(verbose=False)

    mock_resp = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=8000,
            completion_tokens=200,
            prompt_tokens_details=SimpleNamespace(cached_tokens=6000),
        )
    )

    handler.async_log_success_event({"model": "gpt-4o"}, mock_resp, 0, 1)

    assert handler.reporter.session.turns_count == 1
    assert handler.reporter.session.total_cached_tokens == 6000
