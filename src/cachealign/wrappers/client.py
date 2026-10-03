"""
In-Process SDK Client Wrapper.
Provides `cachealign.wrap(client)` to transparently optimize synchronous and
asynchronous LLM client calls across Anthropic and OpenAI.
"""

import inspect
from typing import Any

from cachealign.adapters.anthropic import AnthropicAdapter
from cachealign.adapters.openai import OpenAIAdapter
from cachealign.telemetry.reporter import FinOpsReporter
from cachealign.wrappers.session import CacheAlignSession
from cachealign.wrappers.streaming import WrappedAsyncStream, WrappedSyncStream


class WrappedMessagesResource:
    """Wraps Anthropic client.messages (Sync & Async) to optimize calls."""

    def __init__(self, original_messages: Any, reporter: FinOpsReporter):
        self._original = original_messages
        self._reporter = reporter
        self._adapter = AnthropicAdapter()
        self._is_async = inspect.iscoroutinefunction(getattr(original_messages, "create", None))

    def create(self, *args: Any, **kwargs: Any) -> Any:
        opt_res = self._adapter.optimize_request(kwargs)
        is_streaming = kwargs.get("stream", False)
        model = kwargs.get("model", "claude-3-5-sonnet")

        if self._is_async:

            async def _async_create():
                raw_resp = await self._original.create(*args, **opt_res.optimized_kwargs)
                if is_streaming:
                    return WrappedAsyncStream(
                        raw_resp, self._adapter, self._reporter, model, "anthropic"
                    )
                stats = self._adapter.parse_usage(raw_resp, model)
                self._reporter.report_turn(stats, model)
                return raw_resp

            return _async_create()
        else:
            raw_resp = self._original.create(*args, **opt_res.optimized_kwargs)
            if is_streaming:
                return WrappedSyncStream(
                    raw_resp, self._adapter, self._reporter, model, "anthropic"
                )
            stats = self._adapter.parse_usage(raw_resp, model)
            self._reporter.report_turn(stats, model)
            return raw_resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class WrappedChatCompletionsResource:
    """Wraps OpenAI client.chat.completions (Sync & Async) to optimize calls."""

    def __init__(self, original_completions: Any, reporter: FinOpsReporter):
        self._original = original_completions
        self._reporter = reporter
        self._adapter = OpenAIAdapter()
        self._is_async = inspect.iscoroutinefunction(getattr(original_completions, "create", None))

    def create(self, *args: Any, **kwargs: Any) -> Any:
        # Transparently request stream options to obtain usage stats on streaming
        is_streaming = kwargs.get("stream", False)
        if is_streaming and "stream_options" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["stream_options"] = {"include_usage": True}

        opt_res = self._adapter.optimize_request(kwargs)
        model = kwargs.get("model", "gpt-4o")

        if self._is_async:

            async def _async_create():
                raw_resp = await self._original.create(*args, **opt_res.optimized_kwargs)
                if is_streaming:
                    return WrappedAsyncStream(
                        raw_resp, self._adapter, self._reporter, model, "openai"
                    )
                stats = self._adapter.parse_usage(raw_resp, model)
                self._reporter.report_turn(stats, model)
                return raw_resp

            return _async_create()
        else:
            raw_resp = self._original.create(*args, **opt_res.optimized_kwargs)
            if is_streaming:
                return WrappedSyncStream(raw_resp, self._adapter, self._reporter, model, "openai")
            stats = self._adapter.parse_usage(raw_resp, model)
            self._reporter.report_turn(stats, model)
            return raw_resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class WrappedChatResource:
    """Wraps OpenAI client.chat resource."""

    def __init__(self, original_chat: Any, reporter: FinOpsReporter):
        self._original = original_chat
        self.completions = WrappedChatCompletionsResource(original_chat.completions, reporter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class ClientWrapper:
    """Generic wrapper forwarding calls while intercepting target endpoint methods."""

    def __init__(
        self,
        client: Any,
        reporter: FinOpsReporter | None = None,
        verbose: bool = True,
    ):
        self._client = client
        self._reporter = reporter if reporter is not None else FinOpsReporter(verbose=verbose)

        # Detect Anthropic client (Sync or Async)
        if hasattr(client, "messages"):
            self.messages = WrappedMessagesResource(client.messages, self._reporter)

        # Detect OpenAI client (Sync or Async)
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            self.chat = WrappedChatResource(client.chat, self._reporter)

    @property
    def telemetry(self) -> FinOpsReporter:
        return self._reporter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap(
    client: Any,
    session: CacheAlignSession | None = None,
    verbose: bool = True,
) -> Any:
    """
    Wraps an Anthropic or OpenAI client instance (Sync or Async) with CacheAlign optimization.

    Usage:
        import anthropic
        from cachealign import wrap

        client = wrap(anthropic.Anthropic())
        response = client.messages.create(...)

        # Or with async client:
        async_client = wrap(anthropic.AsyncAnthropic())
        response = await async_client.messages.create(...)
    """
    reporter = session.reporter if session else None
    return ClientWrapper(client, reporter=reporter, verbose=verbose)
