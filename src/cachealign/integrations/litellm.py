# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
LiteLLM Integration for CacheAlign.
Provides CacheAlignLiteLLMHandler conforming to LiteLLM CustomLogger interface.
Optimizes outbound requests passing through LiteLLM proxy or Python SDK.
"""

from typing import Any

from cachealign.adapters.base import UsageStats
from cachealign.config import CacheAlignConfig
from cachealign.normalizers.partitioner import partition_messages_and_system
from cachealign.normalizers.schema import canonicalize_tool_schemas
from cachealign.telemetry.reporter import FinOpsReporter


class CacheAlignLiteLLMHandler:
    """
    LiteLLM CustomLogger plugin.
    Hooks into pre_call_hook to canonicalize schemas and isolate volatile prompt variables,
    and log_success_event to record FinOps savings.
    """

    def __init__(
        self,
        config: CacheAlignConfig | None = None,
        reporter: FinOpsReporter | None = None,
        verbose: bool = True,
    ):
        self.config = config or CacheAlignConfig(verbose=verbose)
        self.reporter = reporter or FinOpsReporter(verbose=self.config.verbose)

    def async_pre_call_hook(
        self,
        user_api_key_dict: dict[str, Any],
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        """Intercepts outbound LiteLLM payload before provider dispatch."""
        if not data:
            return data

        # 1. Canonicalize tools if present
        tools = data.get("tools")
        if tools and self.config.enable_schema_canonicalization:
            data["tools"] = canonicalize_tool_schemas(tools)

        # 2. Partition system prompt if present
        messages = data.get("messages")
        if messages and self.config.enable_tail_migration:
            system_content = None
            other_messages = []
            for msg in messages:
                if msg.get("role") == "system" and system_content is None:
                    system_content = msg.get("content")
                else:
                    other_messages.append(msg)

            if system_content and other_messages:
                clean_sys, clean_other, _ = partition_messages_and_system(
                    system_content,
                    other_messages,
                    custom_patterns=self.config.custom_volatile_patterns,
                )
                data["messages"] = [{"role": "system", "content": clean_sys}, *clean_other]

        return data

    def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Captures usage telemetry from completed LiteLLM response."""
        usage = getattr(response_obj, "usage", None)
        if not usage:
            return

        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)
        cached_tokens = 0

        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details:
            cached_tokens = getattr(prompt_details, "cached_tokens", 0)

        hit_rate = (cached_tokens / max(1, input_tokens)) * 100.0 if input_tokens > 0 else 0.0

        stats = UsageStats(
            total_tokens=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_read_tokens=cached_tokens,
            cache_hit_rate=round(hit_rate, 2),
            estimated_cost_saved_usd=(cached_tokens / 1_000_000.0) * 1.25,
        )
        model = kwargs.get("model", "litellm-proxy")
        self.reporter.report_turn(stats, model)
