from types import SimpleNamespace
from unittest.mock import MagicMock

from cachealign import wrap


def test_wrap_anthropic_client():
    mock_messages = MagicMock()
    mock_response = SimpleNamespace(
        id="msg_123",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=2000,
        ),
    )
    mock_messages.create.return_value = mock_response

    mock_client = SimpleNamespace(messages=mock_messages)
    wrapped = wrap(mock_client, verbose=False)

    resp = wrapped.messages.create(
        model="claude-3-5-sonnet-20241022",
        system="System prompt\nCurrent Time: 2026-10-03",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert resp.id == "msg_123"
    assert mock_messages.create.called
    call_kwargs = mock_messages.create.call_args[1]

    # Verify that wrap intercepted and optimized kwargs
    assert isinstance(call_kwargs["system"], list)
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert wrapped.telemetry.session.total_cached_tokens == 2000


def test_wrap_openai_client():
    mock_completions = MagicMock()
    mock_response = SimpleNamespace(
        id="chat_123",
        usage=SimpleNamespace(
            prompt_tokens=4000,
            completion_tokens=100,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3000),
        ),
    )
    mock_completions.create.return_value = mock_response

    mock_chat = SimpleNamespace(completions=mock_completions)
    mock_client = SimpleNamespace(chat=mock_chat)

    wrapped = wrap(mock_client, verbose=False)

    resp = wrapped.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a bot.\nCurrent Time: 2026-10-03"},
            {"role": "user", "content": "Hi"},
        ],
    )

    assert resp.id == "chat_123"
    assert mock_completions.create.called
    call_kwargs = mock_completions.create.call_args[1]

    # Verify that volatile timestamp was migrated out of system prompt
    assert "Current Time:" not in call_kwargs["messages"][0]["content"]
    assert "Current Time:" in call_kwargs["messages"][1]["content"]
    assert wrapped.telemetry.session.total_cached_tokens == 3000
