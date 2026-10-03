from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cachealign import wrap


def test_sync_streaming_anthropic():
    # Simulate Anthropic streaming chunks: message_start -> delta -> message_delta
    chunks = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=100,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=3000,
                )
            ),
        ),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="Hello ")),
        SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=25)),
    ]

    mock_messages = SimpleNamespace()
    mock_messages.create = MagicMock(return_value=iter(chunks))

    mock_client = SimpleNamespace(messages=mock_messages)
    wrapped = wrap(mock_client, verbose=False)

    stream = wrapped.messages.create(
        model="claude-3-5-sonnet-20241022",
        system="System prompt",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    consumed = list(stream)
    assert len(consumed) == 3
    # Check that usage was extracted upon stream completion
    assert wrapped.telemetry.session.total_cached_tokens == 3000
    assert wrapped.telemetry.session.turns_count == 1


@pytest.mark.anyio
async def test_async_streaming_openai():
    # Simulate OpenAI streaming chunks with usage in the final chunk
    async def mock_async_stream():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))])
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=4000,
                completion_tokens=50,
                prompt_tokens_details=SimpleNamespace(cached_tokens=3500),
            ),
        )

    mock_completions = SimpleNamespace()
    mock_completions.create = AsyncMock(return_value=mock_async_stream())

    mock_chat = SimpleNamespace(completions=mock_completions)
    mock_client = SimpleNamespace(chat=mock_chat)

    wrapped = wrap(mock_client, verbose=False)

    stream = await wrapped.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Hello"}], stream=True
    )

    consumed = []
    async for chunk in stream:
        consumed.append(chunk)

    assert len(consumed) == 2
    assert wrapped.telemetry.session.total_cached_tokens == 3500
