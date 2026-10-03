from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cachealign import optimize, session, wrap


def test_session_context_manager():
    with session(verbose=False) as sess:
        mock_messages = SimpleNamespace()
        mock_messages.create = MagicMock(
            return_value=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_input_tokens=2000,
                    cache_creation_input_tokens=0,
                )
            )
        )
        client = wrap(SimpleNamespace(messages=mock_messages), session=sess)
        client.messages.create(
            model="claude-3-5-sonnet", messages=[{"role": "user", "content": "Hi"}]
        )

        assert sess.telemetry.total_cached_tokens == 2000
        assert sess.telemetry.turns_count == 1


@optimize(verbose=False)
def sync_agent_task():
    return "done"


@pytest.mark.anyio
@optimize(verbose=False)
async def async_agent_task():
    return "async_done"


def test_optimize_decorator_sync():
    res = sync_agent_task()
    assert res == "done"


@pytest.mark.anyio
async def test_optimize_decorator_async():
    res = await async_agent_task()
    assert res == "async_done"
