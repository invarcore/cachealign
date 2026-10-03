from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cachealign import wrap


@pytest.mark.anyio
async def test_wrap_async_anthropic_client():
    mock_messages = SimpleNamespace()
    mock_response = SimpleNamespace(
        id="async_msg_123",
        usage=SimpleNamespace(
            input_tokens=200,
            output_tokens=60,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=4000,
        ),
    )
    mock_messages.create = AsyncMock(return_value=mock_response)

    mock_client = SimpleNamespace(messages=mock_messages)
    wrapped = wrap(mock_client, verbose=False)

    resp = await wrapped.messages.create(
        model="claude-3-5-sonnet-20241022",
        system="System prompt\nCurrent Time: 2026-10-03 12:00:00",
        messages=[{"role": "user", "content": "Async prompt"}],
    )

    assert resp.id == "async_msg_123"
    assert mock_messages.create.called
    call_kwargs = mock_messages.create.call_args[1]

    # Verify that wrap intercepted and optimized kwargs
    assert isinstance(call_kwargs["system"], list)
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert wrapped.telemetry.session.total_cached_tokens == 4000


@pytest.mark.anyio
async def test_wrap_async_openai_client():
    mock_completions = SimpleNamespace()
    mock_response = SimpleNamespace(
        id="async_chat_123",
        usage=SimpleNamespace(
            prompt_tokens=5000,
            completion_tokens=150,
            prompt_tokens_details=SimpleNamespace(cached_tokens=4500),
        ),
    )
    mock_completions.create = AsyncMock(return_value=mock_response)

    mock_chat = SimpleNamespace(completions=mock_completions)
    mock_client = SimpleNamespace(chat=mock_chat)

    wrapped = wrap(mock_client, verbose=False)

    resp = await wrapped.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a bot.\nCurrent Time: 2026-10-03"},
            {"role": "user", "content": "Async question"},
        ],
    )

    assert resp.id == "async_chat_123"
    assert mock_completions.create.called
    call_kwargs = mock_completions.create.call_args[1]

    assert "Current Time:" not in call_kwargs["messages"][0]["content"]
    assert "Current Time:" in call_kwargs["messages"][1]["content"]
    assert wrapped.telemetry.session.total_cached_tokens == 4500
