# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
CacheAlign: Autonomous prompt cache optimizer & prefix alignment middleware for AI agents.
"""

from cachealign.adapters.anthropic import AnthropicAdapter
from cachealign.adapters.base import OptimizationResult, ProviderAdapter, UsageStats
from cachealign.adapters.gemini import GeminiAdapter
from cachealign.adapters.openai import OpenAIAdapter
from cachealign.config import CacheAlignConfig
from cachealign.normalizers.partitioner import (
    extract_volatile_elements,
    partition_messages_and_system,
)
from cachealign.normalizers.schema import canonicalize_object, canonicalize_tool_schemas
from cachealign.telemetry.reporter import FinOpsReporter, SessionTelemetry
from cachealign.wrappers.client import ClientWrapper, wrap
from cachealign.wrappers.session import CacheAlignSession, optimize, session

__version__ = "0.1.0"

__all__ = [
    "AnthropicAdapter",
    "CacheAlignConfig",
    "CacheAlignSession",
    "ClientWrapper",
    "FinOpsReporter",
    "GeminiAdapter",
    "OpenAIAdapter",
    "OptimizationResult",
    "ProviderAdapter",
    "SessionTelemetry",
    "UsageStats",
    "__version__",
    "canonicalize_object",
    "canonicalize_tool_schemas",
    "extract_volatile_elements",
    "optimize",
    "partition_messages_and_system",
    "session",
    "wrap",
]
