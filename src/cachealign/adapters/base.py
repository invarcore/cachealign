"""
Base Provider Adapter Interface for CacheAlign.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageStats:
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    estimated_cost_saved_usd: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass
class OptimizationResult:
    optimized_kwargs: dict[str, Any]
    ephemeral_tokens_extracted: list[str] = field(default_factory=list)
    breakpoints_injected: int = 0


class ProviderAdapter(ABC):
    """Abstract interface for LLM provider caching adapters."""

    @abstractmethod
    def optimize_request(self, kwargs: dict[str, Any]) -> OptimizationResult:
        """Transforms outbound request parameters to maximize provider prompt-cache hit rate."""
        pass

    @abstractmethod
    def parse_usage(self, response: Any, model: str) -> UsageStats:
        """Extracts token usage and cache metrics from provider response."""
        pass
