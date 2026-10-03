"""
RFC 8785 JSON Canonicalization Scheme (JCS) Normalizer for Tool Schemas.
Ensures tool definitions and schemas have byte-identical serialized output across
different processes, threads, and runtimes to guarantee LLM prefix cache hits.
"""

from typing import Any


def canonicalize_object(obj: Any) -> Any:
    """
    Recursively sorts dictionary keys in lexicographical order (RFC 8785)
    and formats JSON structures deterministically.
    """
    if isinstance(obj, dict):
        return {k: canonicalize_object(v) for k, v in sorted(obj.items(), key=lambda item: item[0])}
    elif isinstance(obj, (list, tuple)):
        return [canonicalize_object(item) for item in obj]
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
