"""
In-Process SDK Client Wrapper.
Provides `cachealign.wrap(client)` to transparently optimize synchronous and
asynchronous LLM client calls across Anthropic and OpenAI.
Features enterprise fail-open reliability, multi-tenant cryptographic isolation,
and full Anthropic messages.stream context manager support.
"""

import inspect
import logging
from typing import Any

from cachealign.adapters.anthropic import AnthropicAdapter
from cachealign.adapters.openai import OpenAIAdapter
from cachealign.config import CacheAlignConfig
from cachealign.telemetry.reporter import FinOpsReporter
from cachealign.wrappers.session import CacheAlignSession
from cachealign.wrappers.streaming import (
    WrappedAsyncStream,
    WrappedMessageStreamManager,
    WrappedSyncStream,
)

logger = logging.getLogger("cachealign")


class WrappedMessagesResource:
    """Wraps Anthropic client.messages (Sync & Async) to optimize calls."""

    def __init__(self, original_messages: Any, reporter: FinOpsReporter, config: CacheAlignConfig):
        self._original = original_messages
        self._reporter = reporter
        self._config = config
        self._adapter = AnthropicAdapter(config=config)
        self._is_async = inspect.iscoroutinefunction(getattr(original_messages, "create", None))

    def _safe_optimize(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Safely optimizes request parameters with fail-open guarantee."""
        try:
            res = self._adapter.optimize_request(kwargs)
            return res.optimized_kwargs
        except Exception as e:
            if self._config.fail_open:
                logger.warning(
                    "CacheAlign optimization encountered an exception; "
                    "falling open to unoptimized request: %s",
                    e,
                    exc_info=True,
                )
                return kwargs
            raise

    def create(self, *args: Any, **kwargs: Any) -> Any:
        optimized_kwargs = self._safe_optimize(kwargs)
        is_streaming = kwargs.get("stream", False)
        model = kwargs.get("model", "claude-3-5-sonnet")

        if self._is_async:

            async def _async_create():
                raw_resp = await self._original.create(*args, **optimized_kwargs)
                if is_streaming:
                    return WrappedAsyncStream(
                        raw_resp, self._adapter, self._reporter, model, "anthropic"
                    )
                stats = self._adapter.parse_usage(raw_resp, model)
                self._reporter.report_turn(stats, model)
                return raw_resp

            return _async_create()
        else:
            raw_resp = self._original.create(*args, **optimized_kwargs)
            if is_streaming:
                return WrappedSyncStream(
                    raw_resp, self._adapter, self._reporter, model, "anthropic"
                )
            stats = self._adapter.parse_usage(raw_resp, model)
            self._reporter.report_turn(stats, model)
            return raw_resp

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """
        Intercepts Anthropic client.messages.stream(...) returning a WrappedMessageStreamManager.
        Preserves context manager syntax and auto-captures usage metrics upon completion.
        """
        optimized_kwargs = self._safe_optimize(kwargs)
        model = kwargs.get("model", "claude-3-5-sonnet")

        raw_manager = self._original.stream(*args, **optimized_kwargs)
        return WrappedMessageStreamManager(raw_manager, self._adapter, self._reporter, model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class WrappedChatCompletionsResource:
    """Wraps OpenAI client.chat.completions (Sync & Async) to optimize calls."""

    def __init__(
        self, original_completions: Any, reporter: FinOpsReporter, config: CacheAlignConfig
    ):
        self._original = original_completions
        self._reporter = reporter
        self._config = config
        self._adapter = OpenAIAdapter(config=config)
        self._is_async = inspect.iscoroutinefunction(getattr(original_completions, "create", None))

    def _safe_optimize(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Safely optimizes request parameters with fail-open guarantee."""
        try:
            res = self._adapter.optimize_request(kwargs)
            return res.optimized_kwargs
        except Exception as e:
            if self._config.fail_open:
                logger.warning(
                    "CacheAlign optimization encountered an exception; "
                    "falling open to unoptimized request: %s",
                    e,
                    exc_info=True,
                )
                return kwargs
            raise

    def create(self, *args: Any, **kwargs: Any) -> Any:
        # Transparently request stream options to obtain usage stats on streaming
        is_streaming = kwargs.get("stream", False)
        if is_streaming and "stream_options" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["stream_options"] = {"include_usage": True}

        optimized_kwargs = self._safe_optimize(kwargs)
        model = kwargs.get("model", "gpt-4o")

        if self._is_async:

            async def _async_create():
                raw_resp = await self._original.create(*args, **optimized_kwargs)
                if is_streaming:
                    return WrappedAsyncStream(
                        raw_resp, self._adapter, self._reporter, model, "openai"
                    )
                stats = self._adapter.parse_usage(raw_resp, model)
                self._reporter.report_turn(stats, model)
                return raw_resp

            return _async_create()
        else:
            raw_resp = self._original.create(*args, **optimized_kwargs)
            if is_streaming:
                return WrappedSyncStream(raw_resp, self._adapter, self._reporter, model, "openai")
            stats = self._adapter.parse_usage(raw_resp, model)
            self._reporter.report_turn(stats, model)
            return raw_resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class WrappedChatResource:
    """Wraps OpenAI client.chat resource."""

    def __init__(self, original_chat: Any, reporter: FinOpsReporter, config: CacheAlignConfig):
        self._original = original_chat
        self.completions = WrappedChatCompletionsResource(
            original_chat.completions, reporter, config
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class ClientWrapper:
    """Generic wrapper forwarding calls while intercepting target endpoint methods."""

    def __init__(
        self,
        client: Any,
        reporter: FinOpsReporter | None = None,
        config: CacheAlignConfig | None = None,
        verbose: bool = True,
        tenant_id: str | None = None,
        fail_open: bool = True,
    ):
        self._client = client
        if config is None:
            config = CacheAlignConfig(
                verbose=verbose,
                tenant_id=tenant_id,
                fail_open=fail_open,
            )
        self._config = config
        self._reporter = (
            reporter if reporter is not None else FinOpsReporter(verbose=config.verbose)
        )

        # Detect Anthropic client (Sync or Async)
        if hasattr(client, "messages"):
            self.messages = WrappedMessagesResource(client.messages, self._reporter, self._config)

        # Detect OpenAI client (Sync or Async)
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            self.chat = WrappedChatResource(client.chat, self._reporter, self._config)

    @property
    def telemetry(self) -> FinOpsReporter:
        return self._reporter

    @property
    def config(self) -> CacheAlignConfig:
        return self._config

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap(
    client: Any,
    session: CacheAlignSession | None = None,
    config: CacheAlignConfig | None = None,
    verbose: bool = True,
    tenant_id: str | None = None,
    fail_open: bool = True,
) -> Any:
    """
    Wraps an Anthropic or OpenAI client instance (Sync or Async) with CacheAlign optimization.

    Usage:
        import anthropic
        from cachealign import wrap

        client = wrap(anthropic.Anthropic())
        response = client.messages.create(...)

        # Or with streaming context manager:
        with client.messages.stream(...) as stream:
            for text in stream.text_stream:
                print(text, end="")
    """
    reporter = session.reporter if session else None
    return ClientWrapper(
        client,
        reporter=reporter,
        config=config,
        verbose=verbose,
        tenant_id=tenant_id,
        fail_open=fail_open,
    )
