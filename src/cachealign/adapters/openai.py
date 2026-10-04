# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
OpenAI ChatCompletions Prompt Caching Adapter.
Implements RFC 8785 tool canonicalization, tenant cryptographic salt injection,
and volatile prefix partitioning to ensure 1,024-token boundary alignment
for OpenAI automatic prompt caching.
"""

import hashlib
from typing import Any

from cachealign.adapters.base import OptimizationResult, ProviderAdapter, UsageStats
from cachealign.config import CacheAlignConfig
from cachealign.normalizers.partitioner import partition_messages_and_system
from cachealign.normalizers.schema import canonicalize_tool_schemas

# OpenAI Pricing per 1M tokens (GPT-4o defaults)
GPT4O_BASE_INPUT_PER_M = 2.50  # $2.50 / 1M
GPT4O_CACHED_INPUT_PER_M = 1.25  # $1.25 / 1M (50% discount)
GPT4O_OUTPUT_PER_M = 10.00  # $10.00 / 1M


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI chat.completions.create API calls."""

    def __init__(
        self,
        config: CacheAlignConfig | None = None,
        tenant_id: str | None = None,
    ):
        if config is not None:
            self.tenant_id = config.tenant_id
            self.enable_tail_migration = config.enable_tail_migration
            self.enable_schema_canonicalization = config.enable_schema_canonicalization
            self.custom_patterns = config.custom_volatile_patterns
        else:
            self.tenant_id = tenant_id
            self.enable_tail_migration = True
            self.enable_schema_canonicalization = True
            self.custom_patterns = []

    def optimize_request(self, kwargs: dict[str, Any]) -> OptimizationResult:
        new_kwargs = dict(kwargs)

        # 0. Multi-Tenant Cryptographic Salt Injection (CacheProbe Defense)
        tenant_salt_header = ""
        if self.tenant_id:
            salt_hash = hashlib.sha256(self.tenant_id.encode("utf-8")).hexdigest()[:16]
            tenant_salt_header = f"[CacheTenancy: {salt_hash}]\n"

        # 1. Canonicalize Tools & Functions
        if self.enable_schema_canonicalization:
            if new_kwargs.get("tools"):
                new_kwargs["tools"] = canonicalize_tool_schemas(new_kwargs["tools"])
            if new_kwargs.get("functions"):
                new_kwargs["functions"] = canonicalize_tool_schemas(new_kwargs["functions"])

        # 2. Extract System Message & Partition
        messages = list(new_kwargs.get("messages", []))
        # Also check if caller passed a top-level system parameter
        top_system = new_kwargs.pop("system", None) if "system" in new_kwargs else None

        system_content = top_system
        system_idx = -1
        other_messages = []

        for idx, msg in enumerate(messages):
            if msg.get("role") == "system" and system_idx == -1 and system_content is None:
                system_content = msg.get("content")
                system_idx = idx
            else:
                other_messages.append(msg)

        if system_content is not None and other_messages:
            extracted = []
            if self.enable_tail_migration:
                clean_system, clean_other, extracted = partition_messages_and_system(
                    system_content, other_messages, custom_patterns=self.custom_patterns
                )
            else:
                clean_system = system_content
                clean_other = other_messages

            if tenant_salt_header and isinstance(clean_system, str):
                clean_system = tenant_salt_header + clean_system

            final_messages = [{"role": "system", "content": clean_system}, *clean_other]
            new_kwargs["messages"] = final_messages
            return OptimizationResult(
                optimized_kwargs=new_kwargs,
                ephemeral_tokens_extracted=extracted,
                breakpoints_injected=0,
            )
        elif tenant_salt_header:
            if system_content is not None:
                clean_sys = tenant_salt_header + str(system_content)
            else:
                clean_sys = tenant_salt_header.strip()
            final_messages = [{"role": "system", "content": clean_sys}, *other_messages]
            new_kwargs["messages"] = final_messages
            return OptimizationResult(optimized_kwargs=new_kwargs)

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
