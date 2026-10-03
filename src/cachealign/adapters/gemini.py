"""
Google Gemini Context Caching Adapter for CacheAlign.
Supports Google GenAI SDK (google-genai) Client and CachedContent API.
Implements RFC 8785 schema canonicalization, system instruction volatile partitioning,
cryptographic tenant isolation, and automated CachedContent TTL tracking.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from cachealign.adapters.base import OptimizationResult, ProviderAdapter, UsageStats
from cachealign.config import CacheAlignConfig
from cachealign.normalizers.partitioner import (
    build_ephemeral_context_block,
    extract_volatile_elements,
)
from cachealign.normalizers.schema import canonicalize_tool_schemas

# Google Gemini Pricing per 1M tokens (Gemini 2.5 Flash / Pro reference)
GEMINI_FLASH_BASE_INPUT_PER_M = 0.15  # $0.15 / 1M
GEMINI_FLASH_CACHED_INPUT_PER_M = 0.0375  # $0.0375 / 1M (75% discount)
GEMINI_FLASH_STORAGE_PER_M_HR = 0.50  # $0.50 / 1M / hr
GEMINI_FLASH_OUTPUT_PER_M = 0.60  # $0.60 / 1M

GEMINI_PRO_BASE_INPUT_PER_M = 1.25  # $1.25 / 1M
GEMINI_PRO_CACHED_INPUT_PER_M = 0.3125  # $0.3125 / 1M (75% discount)
GEMINI_PRO_STORAGE_PER_M_HR = 4.50  # $4.50 / 1M / hr
GEMINI_PRO_OUTPUT_PER_M = 5.00  # $5.00 / 1M


def estimate_tokens(text_or_obj: Any) -> int:
    """Fast heuristic token estimator (~4 chars per token)."""
    if isinstance(text_or_obj, str):
        return max(1, len(text_or_obj) // 4)
    if isinstance(text_or_obj, (dict, list)):
        return max(1, len(json.dumps(text_or_obj)) // 4)
    return 1


@dataclass
class CachedContentEntry:
    """Metadata for an active Gemini CachedContent resource."""

    name: str
    model: str
    fingerprint: str
    token_count: int
    created_at: float
    expires_at: float

    @property
    def is_valid(self) -> bool:
        # Buffer of 60 seconds before actual expiration
        return time.time() < (self.expires_at - 60.0)


class GeminiAdapter(ProviderAdapter):
    """Adapter for Google Gemini models.generate_content API calls."""

    def __init__(
        self,
        config: CacheAlignConfig | None = None,
        min_token_threshold: int = 2048,  # Gemini 2.5 Flash threshold is 2,048 tokens
        tenant_id: str | None = None,
    ):
        if config is not None:
            self.min_token_threshold = (
                config.min_token_threshold
                if config.min_token_threshold > 0
                else min_token_threshold
            )
            self.tenant_id = config.tenant_id
            self.enable_tail_migration = config.enable_tail_migration
            self.enable_schema_canonicalization = config.enable_schema_canonicalization
            self.custom_patterns = config.custom_volatile_patterns
        else:
            self.min_token_threshold = min_token_threshold
            self.tenant_id = tenant_id
            self.enable_tail_migration = True
            self.enable_schema_canonicalization = True
            self.custom_patterns = []

        # In-memory registry mapping SHA-256(prefix) -> CachedContentEntry
        self._cache_registry: dict[str, CachedContentEntry] = {}

    def register_cached_content(
        self,
        fingerprint: str,
        name: str,
        model: str,
        token_count: int,
        ttl_seconds: int = 3600,
    ) -> CachedContentEntry:
        """Manually or dynamically registers an active Gemini CachedContent resource."""
        now = time.time()
        entry = CachedContentEntry(
            name=name,
            model=model,
            fingerprint=fingerprint,
            token_count=token_count,
            created_at=now,
            expires_at=now + ttl_seconds,
        )
        self._cache_registry[fingerprint] = entry
        return entry

    def get_valid_cached_content(self, fingerprint: str) -> str | None:
        """Returns the resource name of an active cached content if unexpired."""
        entry = self._cache_registry.get(fingerprint)
        if entry and entry.is_valid:
            return entry.name
        return None

    def compute_prefix_fingerprint(
        self,
        system_instruction: Any,
        tools: Any,
        tenant_salt: str = "",
    ) -> str:
        """Computes a deterministic SHA-256 fingerprint of the invariant prefix."""
        serialized = json.dumps(
            [tenant_salt, system_instruction, tools],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def optimize_request(self, kwargs: dict[str, Any]) -> OptimizationResult:
        new_kwargs = dict(kwargs)
        extracted: list[str] = []

        # 0. Multi-Tenant Cryptographic Salt Injection
        tenant_salt_header = ""
        if self.tenant_id:
            salt_hash = hashlib.sha256(self.tenant_id.encode("utf-8")).hexdigest()[:16]
            tenant_salt_header = f"[CacheTenancy: {salt_hash}]\n"

        # 1. Normalize Config (can be a dict or types.GenerateContentConfig)
        raw_config = new_kwargs.get("config")
        config_dict: dict[str, Any] = {}
        is_obj_config = False

        if raw_config is not None:
            if isinstance(raw_config, dict):
                config_dict = dict(raw_config)
            elif hasattr(raw_config, "model_dump"):
                config_dict = raw_config.model_dump()
                is_obj_config = True
            elif hasattr(raw_config, "__dict__"):
                config_dict = dict(raw_config.__dict__)
                is_obj_config = True

        # Extract system_instruction and tools
        system_instruction = config_dict.get("system_instruction")
        tools = config_dict.get("tools")

        # 2. Canonicalize Tool Schemas
        if tools and self.enable_schema_canonicalization:
            if isinstance(tools, list):
                tools = canonicalize_tool_schemas(tools)
                config_dict["tools"] = tools

        # 3. Partition System Instruction
        clean_system = system_instruction
        if self.enable_tail_migration and isinstance(system_instruction, str):
            clean_system, extracted = extract_volatile_elements(
                system_instruction, self.custom_patterns
            )
            if tenant_salt_header:
                clean_system = tenant_salt_header + clean_system
            config_dict["system_instruction"] = clean_system
        elif tenant_salt_header and isinstance(system_instruction, str):
            clean_system = tenant_salt_header + system_instruction
            config_dict["system_instruction"] = clean_system

        # 4. Ephemeral Tail Migration on Contents
        contents = new_kwargs.get("contents")
        if extracted and contents:
            context_block = build_ephemeral_context_block(extracted)
            if isinstance(contents, str):
                new_kwargs["contents"] = contents + context_block
            elif isinstance(contents, list) and contents:
                new_contents = list(contents)
                last_item = new_contents[-1]
                if isinstance(last_item, str):
                    new_contents[-1] = last_item + context_block
                elif isinstance(last_item, dict) and "parts" in last_item:
                    parts = list(last_item.get("parts", []))
                    if parts and isinstance(parts[-1], str):
                        parts[-1] = parts[-1] + context_block
                    elif parts and isinstance(parts[-1], dict) and "text" in parts[-1]:
                        part_dict = dict(parts[-1])
                        part_dict["text"] = part_dict["text"] + context_block
                        parts[-1] = part_dict
                    else:
                        parts.append({"text": context_block})
                    last_item_dict = dict(last_item)
                    last_item_dict["parts"] = parts
                    new_contents[-1] = last_item_dict
                new_kwargs["contents"] = new_contents

        # 5. Check if CachedContent Resource is Active in Registry
        fingerprint = self.compute_prefix_fingerprint(clean_system, tools, tenant_salt_header)
        cached_name = self.get_valid_cached_content(fingerprint)
        if cached_name:
            config_dict["cached_content"] = cached_name

        if config_dict:
            if is_obj_config and hasattr(raw_config, "cached_content"):
                # If original was a Pydantic / SDK config object, update attributes
                for k, v in config_dict.items():
                    if hasattr(raw_config, k):
                        try:
                            setattr(raw_config, k, v)
                        except Exception:
                            pass
                new_kwargs["config"] = raw_config
            else:
                new_kwargs["config"] = config_dict

        return OptimizationResult(
            optimized_kwargs=new_kwargs,
            ephemeral_tokens_extracted=extracted,
            breakpoints_injected=1 if cached_name else 0,
        )

    def parse_usage(self, response: Any, model: str) -> UsageStats:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return UsageStats()

        input_tokens = getattr(usage, "prompt_token_count", 0)
        output_tokens = getattr(usage, "candidates_token_count", 0)
        total_tokens = getattr(usage, "total_token_count", input_tokens + output_tokens)
        cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0

        hit_rate = (cached_tokens / max(1, input_tokens)) * 100.0 if input_tokens > 0 else 0.0

        is_pro = "pro" in model.lower()
        base_rate = GEMINI_PRO_BASE_INPUT_PER_M if is_pro else GEMINI_FLASH_BASE_INPUT_PER_M
        cached_rate = GEMINI_PRO_CACHED_INPUT_PER_M if is_pro else GEMINI_FLASH_CACHED_INPUT_PER_M

        saved_usd = (cached_tokens / 1_000_000.0) * (base_rate - cached_rate)

        return UsageStats(
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_creation_tokens=input_tokens - cached_tokens,
            cache_read_tokens=cached_tokens,
            estimated_cost_saved_usd=max(0.0, saved_usd),
            cache_hit_rate=round(hit_rate, 2),
        )
