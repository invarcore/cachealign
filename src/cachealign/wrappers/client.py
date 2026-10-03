"""
In-Process SDK Client Wrapper.
Provides `cachealign.wrap(client)` to transparently optimize LLM client calls
without changing existing agent or application code.
"""

from typing import Any

from cachealign.adapters.anthropic import AnthropicAdapter
from cachealign.adapters.openai import OpenAIAdapter
from cachealign.telemetry.reporter import FinOpsReporter


class WrappedMessagesResource:
    """Wraps Anthropic client.messages to optimize .create() calls."""

    def __init__(self, original_messages: Any, reporter: FinOpsReporter):
        self._original = original_messages
        self._reporter = reporter
        self._adapter = AnthropicAdapter()

    def create(self, *args: Any, **kwargs: Any) -> Any:
        opt_res = self._adapter.optimize_request(kwargs)
        response = self._original.create(*args, **opt_res.optimized_kwargs)

        model = kwargs.get("model", "claude-3-5-sonnet")
        stats = self._adapter.parse_usage(response, model)
        self._reporter.report_turn(stats, model)

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class WrappedChatCompletionsResource:
    """Wraps OpenAI client.chat.completions to optimize .create() calls."""

    def __init__(self, original_completions: Any, reporter: FinOpsReporter):
        self._original = original_completions
        self._reporter = reporter
        self._adapter = OpenAIAdapter()

    def create(self, *args: Any, **kwargs: Any) -> Any:
        opt_res = self._adapter.optimize_request(kwargs)
        response = self._original.create(*args, **opt_res.optimized_kwargs)

        model = kwargs.get("model", "gpt-4o")
        stats = self._adapter.parse_usage(response, model)
        self._reporter.report_turn(stats, model)

        return response

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

    def __init__(self, client: Any, verbose: bool = True):
        self._client = client
        self._reporter = FinOpsReporter(verbose=verbose)

        # Detect Anthropic client
        if hasattr(client, "messages"):
            self.messages = WrappedMessagesResource(client.messages, self._reporter)

        # Detect OpenAI client
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            self.chat = WrappedChatResource(client.chat, self._reporter)

    @property
    def telemetry(self) -> FinOpsReporter:
        return self._reporter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap(client: Any, verbose: bool = True) -> Any:
    """
    Wraps an Anthropic or OpenAI client instance with CacheAlign optimization.

    Usage:
        import anthropic
        from cachealign import wrap

        client = wrap(anthropic.Anthropic())
        response = client.messages.create(...)
    """
    return ClientWrapper(client, verbose=verbose)
