"""
OpenAI ChatCompletions Prompt Caching Adapter.
Implements RFC 8785 tool canonicalization and volatile prefix partitioning
to ensure 1,024-token boundary alignment for OpenAI automatic prompt caching.
"""

from typing import Any

from cachealign.adapters.base import OptimizationResult, ProviderAdapter, UsageStats
from cachealign.normalizers.partitioner import partition_messages_and_system
from cachealign.normalizers.schema import canonicalize_tool_schemas

# OpenAI Pricing per 1M tokens (GPT-4o defaults)
GPT4O_BASE_INPUT_PER_M = 2.50  # $2.50 / 1M
GPT4O_CACHED_INPUT_PER_M = 1.25  # $1.25 / 1M (50% discount)
GPT4O_OUTPUT_PER_M = 10.00  # $10.00 / 1M


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI chat.completions.create API calls."""

    def optimize_request(self, kwargs: dict[str, Any]) -> OptimizationResult:
        new_kwargs = dict(kwargs)

        # 1. Canonicalize Tools & Functions
        if new_kwargs.get("tools"):
            new_kwargs["tools"] = canonicalize_tool_schemas(new_kwargs["tools"])
        if new_kwargs.get("functions"):
            new_kwargs["functions"] = canonicalize_tool_schemas(new_kwargs["functions"])

        # 2. Extract System Message & Partition
        messages = new_kwargs.get("messages", [])
        if not messages:
            return OptimizationResult(optimized_kwargs=new_kwargs)

        system_content = None
        system_idx = -1
        other_messages = []

        for idx, msg in enumerate(messages):
            if msg.get("role") == "system" and system_idx == -1:
                system_content = msg.get("content")
                system_idx = idx
            else:
                other_messages.append(msg)

        if system_content is not None and other_messages:
            clean_system, clean_other, extracted = partition_messages_and_system(
                system_content, other_messages
            )
            # Reconstruct messages with system at index 0
            final_messages = [{"role": "system", "content": clean_system}, *clean_other]
            new_kwargs["messages"] = final_messages
            return OptimizationResult(
                optimized_kwargs=new_kwargs,
                ephemeral_tokens_extracted=extracted,
                breakpoints_injected=0,
            )

        return OptimizationResult(optimized_kwargs=new_kwargs)

    def parse_usage(self, response: Any, model: str) -> UsageStats:
        usage = getattr(response, "usage", None)
        if not usage:
            return UsageStats()

        total_input = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", total_input + output_tokens)

        cached_tokens = 0
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details:
            cached_tokens = getattr(prompt_details, "cached_tokens", 0)

        hit_rate = (cached_tokens / max(1, total_input)) * 100.0 if total_input > 0 else 0.0

        # Cost savings calculation
        saved_usd = (cached_tokens / 1_000_000.0) * (
            GPT4O_BASE_INPUT_PER_M - GPT4O_CACHED_INPUT_PER_M
        )

        return UsageStats(
            total_tokens=total_tokens,
            input_tokens=total_input,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_creation_tokens=total_input - cached_tokens,
            cache_read_tokens=cached_tokens,
            estimated_cost_saved_usd=max(0.0, saved_usd),
            cache_hit_rate=round(hit_rate, 2),
        )
