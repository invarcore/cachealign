"""
LangChain and LangGraph Integration for CacheAlign.
Provides CacheAlignCallbackHandler for automated prompt cache optimization,
volatile variable extraction, and FinOps telemetry tracking across LangGraph agents.
"""

from typing import Any

from cachealign.adapters.anthropic import AnthropicAdapter
from cachealign.adapters.base import UsageStats
from cachealign.adapters.openai import OpenAIAdapter
from cachealign.config import CacheAlignConfig
from cachealign.normalizers.partitioner import partition_messages_and_system
from cachealign.normalizers.schema import canonicalize_tool_schemas
from cachealign.telemetry.reporter import FinOpsReporter

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:

    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Fallback base class when langchain-core is not installed."""

        pass


class CacheAlignCallbackHandler(BaseCallbackHandler):
    """
    LangChain / LangGraph Callback Handler that monitors prompt cache hits,
    tracks token velocity, and provides pre-call optimization hooks.
    """

    def __init__(
        self,
        config: CacheAlignConfig | None = None,
        reporter: FinOpsReporter | None = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.config = config or CacheAlignConfig(verbose=verbose)
        self.reporter = reporter or FinOpsReporter(verbose=self.config.verbose)
        self.anthropic_adapter = AnthropicAdapter(config=self.config)
        self.openai_adapter = OpenAIAdapter(config=self.config)

    def optimize_chat_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """
        Normalizes chat messages and tools for LangChain/LangGraph runnables.
        Can be invoked directly or inside RunnableLambdas.
        """
        sorted_tools = tools
        if tools and self.config.enable_schema_canonicalization:
            sorted_tools = canonicalize_tool_schemas(tools)

        if not messages or not self.config.enable_tail_migration:
            return messages, sorted_tools

        system_content = None
        other_messages = []
        for msg in messages:
            if msg.get("role") == "system" and system_content is None:
                system_content = msg.get("content")
            else:
                other_messages.append(msg)

        if system_content and other_messages:
            clean_sys, clean_other, _ = partition_messages_and_system(
                system_content, other_messages, custom_patterns=self.config.custom_volatile_patterns
            )
            final_messages = [{"role": "system", "content": clean_sys}, *clean_other]
            return final_messages, sorted_tools

        return messages, sorted_tools

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Extracts token usage and cache metrics when an LLM call finishes."""
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage", {}) or llm_output.get("usage", {})

        if token_usage:
            input_tokens = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0) or token_usage.get(
                "output_tokens", 0
            )
            cached_tokens = 0

            # OpenAI cached tokens detail
            prompt_details = token_usage.get("prompt_tokens_details")
            if prompt_details and isinstance(prompt_details, dict):
                cached_tokens = prompt_details.get("cached_tokens", 0)
            # Anthropic cache read tokens
            elif "cache_read_input_tokens" in token_usage:
                cached_tokens = token_usage.get("cache_read_input_tokens", 0)

            total = input_tokens + output_tokens
            hit_rate = (cached_tokens / max(1, input_tokens)) * 100.0 if input_tokens > 0 else 0.0

            stats = UsageStats(
                total_tokens=total,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                cache_read_tokens=cached_tokens,
                cache_hit_rate=round(hit_rate, 2),
                estimated_cost_saved_usd=(cached_tokens / 1_000_000.0) * 1.25,
            )
            model_name = llm_output.get("model_name", "langchain-llm")
            self.reporter.report_turn(stats, model_name)
