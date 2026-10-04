# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
RFC 8785 JSON Canonicalization Scheme (JCS) Normalizer for Tool Schemas.
Ensures tool definitions and schemas have byte-identical serialized output across
different processes, threads, and runtimes to guarantee LLM prefix cache hits.
Includes recursion depth protection (anti-DoS) and circular reference detection.
"""

from typing import Any

MAX_SCHEMA_RECURSION_DEPTH = 20


def canonicalize_object(
    obj: Any,
    depth: int = 0,
    visited: set[int] | None = None,
) -> Any:
    """
    Recursively sorts dictionary keys in lexicographical order (RFC 8785)
    and formats JSON structures deterministically.

    Security & Stability:
    - Enforces a maximum recursion depth of 20 to protect against schema recursion bombs.
    - Tracks object identity to detect and reject circular references.
    """
    if depth > MAX_SCHEMA_RECURSION_DEPTH:
        raise ValueError(
            f"Schema recursion limit exceeded (max_depth={MAX_SCHEMA_RECURSION_DEPTH}). "
            "Suspected circular reference or adversarial schema bomb."
        )

    if visited is None:
        visited = set()

    obj_id = id(obj)

    if isinstance(obj, dict):
        if obj_id in visited:
            raise ValueError("Circular reference detected in schema dictionary.")
        visited.add(obj_id)
        try:
            return {
                k: canonicalize_object(v, depth=depth + 1, visited=visited)
                for k, v in sorted(obj.items(), key=lambda item: item[0])
            }
        finally:
            visited.remove(obj_id)

    elif isinstance(obj, (list, tuple)):
        if obj_id in visited:
            raise ValueError("Circular reference detected in schema list.")
        visited.add(obj_id)
        try:
            return [canonicalize_object(item, depth=depth + 1, visited=visited) for item in obj]
        finally:
            visited.remove(obj_id)

    else:
        return obj


def canonicalize_tool_schemas(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """
    Canonicalizes an array of tool schemas.
    Sorts tools deterministically by tool name, then recursively sorts all internal schema keys.
    Works seamlessly with both Anthropic schema format and OpenAI function calling format.
    """
    if not tools:
        return tools

    def _get_tool_identifier(t: dict[str, Any]) -> str:
        # Anthropic format: {"name": "...", "description": "...", "input_schema": {...}}
        if "name" in t:
            return str(t["name"])
        # OpenAI format: {"type": "function", "function": {"name": "...", ...}}
        if "function" in t and isinstance(t["function"], dict) and "name" in t["function"]:
            return str(t["function"]["name"])
        return str(t)

    # 1. Sort tools deterministically by their identifier
    sorted_tools = sorted(tools, key=_get_tool_identifier)

    # 2. Canonicalize dictionary keys within each tool
    canonical_tools = [canonicalize_object(tool) for tool in sorted_tools]
    return canonical_tools
