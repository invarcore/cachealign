"""
Streaming SSE wrappers for CacheAlign.
Intercepts chunk streams in real-time with sub-millisecond pass-through,
extracting token usage metadata upon stream completion.
Supports Anthropic MessageStreamManager, OpenAI stream iterators, and Google Gemini streams.
"""

import inspect
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

from cachealign.adapters.base import ProviderAdapter
from cachealign.telemetry.reporter import FinOpsReporter


class StreamingUsageCollector:
    """Accumulates usage statistics from provider streaming chunks."""

    def __init__(self, provider: str):
        self.provider = provider
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0
        self.cached_tokens = 0
        self.has_usage = False

    def inspect_chunk(self, chunk: Any) -> None:
        if self.provider == "anthropic":
            # Anthropic streaming events
            event_type = getattr(chunk, "type", None)
            if event_type == "message_start":
                msg = getattr(chunk, "message", None)
                if msg and hasattr(msg, "usage"):
                    usage = msg.usage
                    self.input_tokens = getattr(usage, "input_tokens", 0)
                    self.cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0)
                    self.cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0)
                    self.has_usage = True
            elif event_type == "message_delta":
                usage = getattr(chunk, "usage", None)
                if usage:
                    self.output_tokens = getattr(usage, "output_tokens", 0)
                    self.has_usage = True

        elif self.provider == "openai":
            # OpenAI streaming chunk usage
            usage = getattr(chunk, "usage", None)
            if usage:
                self.input_tokens = getattr(usage, "prompt_tokens", 0)
                self.output_tokens = getattr(usage, "completion_tokens", 0)
                prompt_details = getattr(usage, "prompt_tokens_details", None)
                if prompt_details:
                    self.cached_tokens = getattr(prompt_details, "cached_tokens", 0)
                self.has_usage = True

        elif self.provider == "gemini":
            # Google Gemini chunk usage
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                self.input_tokens = getattr(usage, "prompt_token_count", 0)
                self.output_tokens = getattr(usage, "candidates_token_count", 0)
                self.cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
                self.has_usage = True

    def build_mock_response(self) -> Any:
        if self.provider == "anthropic":
            return SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens,
                    cache_creation_input_tokens=self.cache_creation_tokens,
                    cache_read_input_tokens=self.cache_read_tokens,
                )
            )
        elif self.provider == "gemini":
            return SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=self.input_tokens,
                    candidates_token_count=self.output_tokens,
                    total_token_count=self.input_tokens + self.output_tokens,
                    cached_content_token_count=self.cached_tokens,
                )
            )
        else:
            return SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=self.input_tokens,
                    completion_tokens=self.output_tokens,
                    total_tokens=self.input_tokens + self.output_tokens,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=self.cached_tokens),
                )
            )


class WrappedSyncStream:
    """Wraps synchronous streaming iterators."""

    def __init__(
        self,
        stream: Iterator[Any],
        adapter: ProviderAdapter,
        reporter: FinOpsReporter,
        model: str,
        provider: str,
    ):
        self._stream = stream
        self._adapter = adapter
        self._reporter = reporter
        self._model = model
        self._collector = StreamingUsageCollector(provider)

    def __iter__(self) -> Iterator[Any]:
        try:
            for chunk in self._stream:
                self._collector.inspect_chunk(chunk)
                yield chunk
        finally:
            if self._collector.has_usage:
                mock_resp = self._collector.build_mock_response()
                stats = self._adapter.parse_usage(mock_resp, self._model)
                self._reporter.report_turn(stats, self._model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class WrappedAsyncStream:
    """Wraps asynchronous streaming iterators."""

    def __init__(
        self,
        stream: AsyncIterator[Any],
        adapter: ProviderAdapter,
        reporter: FinOpsReporter,
        model: str,
        provider: str,
    ):
        self._stream = stream
        self._adapter = adapter
        self._reporter = reporter
        self._model = model
        self._collector = StreamingUsageCollector(provider)

    async def __aiter__(self) -> AsyncIterator[Any]:
        try:
            async for chunk in self._stream:
                self._collector.inspect_chunk(chunk)
                yield chunk
        finally:
            if self._collector.has_usage:
                mock_resp = self._collector.build_mock_response()
                stats = self._adapter.parse_usage(mock_resp, self._model)
                self._reporter.report_turn(stats, self._model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class WrappedMessageStreamManager:
    """
    Wraps Anthropic client.messages.stream(...) MessageStreamManager context manager.
    Preserves .text_stream, .get_final_message(), and automatically extracts token
    usage telemetry upon exiting the context block.
    """

    def __init__(
        self,
        manager: Any,
        adapter: ProviderAdapter,
        reporter: FinOpsReporter,
        model: str,
    ):
        self._manager = manager
        self._adapter = adapter
        self._reporter = reporter
        self._model = model
        self._stream: Any = None

    def __enter__(self) -> Any:
        self._stream = self._manager.__enter__()
        return self._stream

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        try:
            res = self._manager.__exit__(exc_type, exc_val, exc_tb)
            if self._stream and hasattr(self._stream, "get_final_message"):
                try:
                    final_msg = self._stream.get_final_message()
                    stats = self._adapter.parse_usage(final_msg, self._model)
                    self._reporter.report_turn(stats, self._model)
                except Exception:
                    pass
            return res
        except Exception:
            return self._manager.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> Any:
        self._stream = await self._manager.__aenter__()
        return self._stream

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        try:
            res = await self._manager.__aexit__(exc_type, exc_val, exc_tb)
            if self._stream and hasattr(self._stream, "get_final_message"):
                try:
                    fn = self._stream.get_final_message
                    final_msg = await fn() if inspect.iscoroutinefunction(fn) else fn()
                    stats = self._adapter.parse_usage(final_msg, self._model)
                    self._reporter.report_turn(stats, self._model)
                except Exception:
                    pass
            return res
        except Exception:
            return await self._manager.__aexit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)
