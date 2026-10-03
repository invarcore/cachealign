"""
CacheAlign Configuration Engine.
Provides centralized settings for token thresholds, breakpoint allocation,
fail-open reliability, multi-tenant isolation, and logging.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cachealign")


@dataclass
class CacheAlignConfig:
    """Configuration options for CacheAlign wrapper and adapters."""

    # Reliability: If True, any internal optimization exception logs a warning
    # and safely falls back to the unoptimized request, guaranteeing zero production crashes.
    fail_open: bool = True

    # Token Gatekeeper: Minimum prefix tokens required before setting a cache breakpoint.
    # Set to 0 to always inject breakpoints on system prompts and tools.
    # Set to 1024 for strict Claude 3.5 Sonnet / Opus caching thresholds.
    min_token_threshold: int = 0

    # Maximum cache breakpoints to inject per request (Anthropic hard limit is 4).
    max_breakpoints: int = 4

    # Security & Multi-Tenancy: Optional tenant identifier for cryptographic cache isolation.
    # Prevents CacheProbe timing side-channel attacks across shared enterprise agents.
    tenant_id: str | None = None

    # Normalization Controls
    enable_tail_migration: bool = True
    enable_schema_canonicalization: bool = True

    # User-defined regex patterns for domain-specific volatile prompt variables
    custom_volatile_patterns: list[str] = field(default_factory=list)

    # Reporting & Telemetry
    verbose: bool = True
    custom_logger: Any = None
