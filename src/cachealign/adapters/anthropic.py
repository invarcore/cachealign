# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
Anthropic Claude Prompt Caching Adapter.
Implements RFC 8785 schema canonicalization, static/dynamic prompt partitioning,
token threshold gatekeeping, tenant isolation salts,
and optimal multi-breakpoint cache_control injection (up to 4 blocks).
"""

import hashlib
import json
from typing import Any

from cachealign.adapters.base import OptimizationResult, ProviderAdapter, UsageStats
from cachealign.config import CacheAlignConfig
from cachealign.normalizers.partitioner import partition_messages_and_system
from cachealign.normalizers.schema import canonicalize_tool_schemas

# Anthropic Pricing per 1M tokens (Claude 3.5 Sonnet defaults)
SONNET_BASE_INPUT_PER_M = 3.00  # $3.00 / 1M
SONNET_CACHE_WRITE_PER_M = 3.75  # 1.25x = $3.75 / 1M (5-min TTL)
SONNET_CACHE_READ_PER_M = 0.30  # 0.10x = $0.30 / 1M (90% discount)
SONNET_OUTPUT_PER_M = 15.00  # $15.00 / 1M


def estimate_tokens(text_or_obj: Any) -> int:
    """Fast heuristic token estimator (~4 chars per token)."""
    if isinstance(text_or_obj, str):
        return max(1, len(text_or_obj) // 4)
    if isinstance(text_or_obj, (dict, list)):
        return max(1, len(json.dumps(text_or_obj)) // 4)
    return 1


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic Claude messages.create API calls."""

    def __init__(
        self,
        config: CacheAlignConfig | None = None,
        min_token_threshold: int = 0,
        max_breakpoints: int = 4,
        tenant_id: str | None = None,
    ):
        if config is not None:
            self.min_token_threshold = config.min_token_threshold
            self.max_breakpoints = config.max_breakpoints
            self.tenant_id = config.tenant_id
            self.enable_tail_migration = config.enable_tail_migration
            self.enable_schema_canonicalization = config.enable_schema_canonicalization
            self.custom_patterns = config.custom_volatile_patterns
        else:
            self.min_token_threshold = min_token_threshold
            self.max_breakpoints = max_breakpoints
            self.tenant_id = tenant_id
            self.enable_tail_migration = True
            self.enable_schema_canonicalization = True
            self.custom_patterns = []

    def optimize_request(self, kwargs: dict[str, Any]) -> OptimizationResult:
        new_kwargs = dict(kwargs)
        breakpoints_used = 0
        cumulative_tokens = 0

        # 0. Multi-Tenant Cryptographic Salt Injection (CacheProbe Defense)
        tenant_salt_header = ""
        if self.tenant_id:
            salt_hash = hashlib.sha256(self.tenant_id.encode("utf-8")).hexdigest()[:16]
            tenant_salt_header = f"[CacheTenancy: {salt_hash}]\n"

        # 1. Canonicalize Tool Schemas
        tools_token_count = 0
        if new_kwargs.get("tools") and self.enable_schema_canonicalization:
            new_kwargs["tools"] = canonicalize_tool_schemas(new_kwargs["tools"])
            tools_token_count = estimate_tokens(new_kwargs["tools"])
            cumulative_tokens += tools_token_count

            # Inject cache breakpoint on the last tool definition
            if (
                self.min_token_threshold == 0 or tools_token_count >= self.min_token_threshold
            ) and breakpoints_used < self.max_breakpoints:
                last_tool = dict(new_kwargs["tools"][-1])
                last_tool["cache_control"] = {"type": "ephemeral"}
                new_kwargs["tools"][-1] = last_tool
                breakpoints_used += 1

        # 2. Partition System Prompt & Message Tail
        system = new_kwargs.get("system")
        messages = new_kwargs.get("messages", [])
        extracted = []

        if self.enable_tail_migration:
            clean_system, clean_messages, extracted = partition_messages_and_system(
                system, messages, custom_patterns=self.custom_patterns
            )
        else:
            clean_system = system
            clean_messages = messages

        # Prepend tenant salt to system prompt if tenant_id is configured
        if tenant_salt_header and clean_system:
            if isinstance(clean_system, str):
                clean_system = tenant_salt_header + clean_system
            elif isinstance(clean_system, list) and clean_system:
                first_block = dict(clean_system[0])
                if first_block.get("type") == "text":
                    first_block["text"] = tenant_salt_header + first_block.get("text", "")
                    clean_system[0] = first_block

        system_tokens = estimate_tokens(clean_system) if clean_system else 0
        cumulative_tokens += system_tokens

        # 3. Inject Cache Control on System Prompt (Token-Aware Gatekeeper)
        if clean_system and breakpoints_used < self.max_breakpoints:
            if (
                self.min_token_threshold == 0
                or cumulative_tokens >= self.min_token_threshold
                or system_tokens >= self.min_token_threshold
            ):
                if isinstance(clean_system, str):
                    clean_system = [
                        {
                            "type": "text",
                            "text": clean_system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                    breakpoints_used += 1
                elif isinstance(clean_system, list) and clean_system:
                    last_block = dict(clean_system[-1])
                    last_block["cache_control"] = {"type": "ephemeral"}
                    clean_system[-1] = last_block
                    breakpoints_used += 1

        # 4. Inject Cache Control on History Turn N-2 (if conversation has >= 3 turns)
        if len(clean_messages) >= 3 and breakpoints_used < self.max_breakpoints:
            turn_idx = len(clean_messages) - 2
            history_tokens = sum(estimate_tokens(clean_messages[i]) for i in range(turn_idx + 1))
            total_history_tokens = cumulative_tokens + history_tokens

            if self.min_token_threshold == 0 or total_history_tokens >= self.min_token_threshold:
                target_msg = dict(clean_messages[turn_idx])
                content = target_msg.get("content")

                if isinstance(content, str):
                    target_msg["content"] = [
                        {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                    ]
                    clean_messages[turn_idx] = target_msg
                    breakpoints_used += 1
                elif isinstance(content, list) and content:
                    last_content_block = dict(content[-1])
                    last_content_block["cache_control"] = {"type": "ephemeral"}
                    content[-1] = last_content_block
                    target_msg["content"] = content
                    clean_messages[turn_idx] = target_msg
                    breakpoints_used += 1

        new_kwargs["system"] = clean_system
        new_kwargs["messages"] = clean_messages

        return OptimizationResult(
            optimized_kwargs=new_kwargs,
            ephemeral_tokens_extracted=extracted,
            breakpoints_injected=breakpoints_used,
        )

    def parse_usage(self, response: Any, model: str) -> UsageStats:
        usage = getattr(response, "usage", None)
        if not usage:
            return UsageStats()

        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0)
        cache_read = getattr(usage, "cache_read_input_tokens", 0)

        total_input = input_tokens + cache_creation + cache_read
        hit_rate = (cache_read / max(1, total_input)) * 100.0 if total_input > 0 else 0.0

        # Calculate cost without caching vs with caching
        cost_without_cache = (total_input / 1_000_000.0) * SONNET_BASE_INPUT_PER_M
        actual_input_cost = (
            (input_tokens / 1_000_000.0) * SONNET_BASE_INPUT_PER_M
            + (cache_creation / 1_000_000.0) * SONNET_CACHE_WRITE_PER_M
            + (cache_read / 1_000_000.0) * SONNET_CACHE_READ_PER_M
        )
        saved_usd = max(0.0, cost_without_cache - actual_input_cost)

        return UsageStats(
            total_tokens=total_input + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            estimated_cost_saved_usd=saved_usd,
            cache_hit_rate=round(hit_rate, 2),
        )
