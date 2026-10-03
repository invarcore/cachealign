"""
Static/Dynamic Prompt Partitioner & Ephemeral Tail Migrator.
Identifies volatile variables (timestamps, session IDs, turn counters)
in static system prompts and relocates them to the dynamic tail of the conversation
within a secure XML container, preserving byte-identical prefixes for high cache hit rates.
"""

import html
import re
from typing import Any

# Anchored regex patterns identifying volatile ephemeral content in key-value header structures.
# Strictly avoids matching instructional sentences (e.g. "Do not leak timestamp: ISO format").
DEFAULT_VOLATILE_PATTERNS = [
    # Key-value timestamps or current date/time headers on their own line
    re.compile(
        r"^\s*(?:current\s+time|timestamp|current\s+date(?:\s+and\s+time)?|system\s+time)\s*[:=]\s*(.+)$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Standalone ISO 8601 timestamps on their own line
    re.compile(
        r"^\s*\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\s*$",
        re.MULTILINE,
    ),
    # Session, Request, or Trace IDs
    re.compile(
        r"^\s*(?:session|request|conversation|trace)[_-]id\s*[:=]\s*([a-zA-Z0-9_\-\.]+)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Turn or step counters
    re.compile(
        r"^\s*(?:turn|step)\s*[:=]\s*(\d+(?:\s*(?:of|/)\s*\d+)?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]

# Alias for backward compatibility
VOLATILE_PATTERNS = DEFAULT_VOLATILE_PATTERNS


def extract_volatile_elements(
    text: str,
    custom_patterns: list[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Extracts volatile key-value lines from text, returning:
    (sanitized_static_text, list_of_extracted_volatile_snippets)

    Guarantees:
    - Only matches explicit key-value header lines, preventing false-positive stripping of instructions.
    - Preserves all preceding and following lines intact.
    """
    if not text:
        return text, []

    patterns = list(DEFAULT_VOLATILE_PATTERNS)
    if custom_patterns:
        for p in custom_patterns:
            patterns.append(re.compile(p, re.IGNORECASE | re.MULTILINE))

    extracted: list[str] = []
    lines = text.splitlines(keepends=True)
    kept_lines: list[str] = []

    for line in lines:
        matched = False
        trimmed_line = line.strip()
        if trimmed_line:
            for pattern in patterns:
                # Test against trimmed line to ensure line-level anchor matching
                if pattern.match(trimmed_line):
                    extracted.append(trimmed_line)
                    matched = True
                    break
        if not matched:
            kept_lines.append(line)

    sanitized = "".join(kept_lines).strip()
    return sanitized, extracted


def build_ephemeral_context_block(extracted_items: list[str]) -> str:
    """
    Builds a secure, XML-delimited context block to contain migrated volatile variables.
    Escapes closing tags to prevent delimiter injection.
    """
    escaped_items = [
        html.escape(item, quote=False).replace("</cachealign_ephemeral_context>", "")
        for item in extracted_items
    ]
    inner = "; ".join(escaped_items)
    return f"\n<cachealign_ephemeral_context>\n[Context: {inner}]\n</cachealign_ephemeral_context>"


def partition_messages_and_system(
    system: str | list[dict[str, Any]] | None,
    messages: list[dict[str, Any]],
    custom_patterns: list[str] | None = None,
) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]], list[str]]:
    """
    Partitions the system prompt and message history.
    Any volatile tokens extracted from the static system prefix are appended to the dynamic
    tail of the conversation.

    Tool Result & Message Safety:
    - If the final message is a tool execution result (e.g. OpenAI `role: "tool"` or Anthropic
      `tool_result` block), the volatile context is safely appended as a distinct text block
      or migrated to the preceding user turn, preventing JSON parse failures in tool responses.
    """
    all_extracted: list[str] = []

    # 1. Partition System Prompt if provided as string
    clean_system = system
    if isinstance(system, str):
        clean_system, extracted = extract_volatile_elements(system, custom_patterns)
        all_extracted.extend(extracted)
    elif isinstance(system, list):
        # Anthropic structured system blocks
        new_blocks = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text, extracted = extract_volatile_elements(block.get("text", ""), custom_patterns)
                all_extracted.extend(extracted)
                new_block = dict(block)
                new_block["text"] = text
                new_blocks.append(new_block)
            else:
                new_blocks.append(block)
        clean_system = new_blocks

    if not all_extracted or not messages:
        return clean_system, messages, all_extracted

    # 2. Ephemeral Tail Migration:
    # Build secure XML container for extracted items
    context_block = build_ephemeral_context_block(all_extracted)
    new_messages = [dict(m) for m in messages]

    # Inspect the final message to safely inject context
    last_msg = dict(new_messages[-1])
    role = last_msg.get("role")
    content = last_msg.get("content")

    # Case A: OpenAI Tool Message (role: "tool") -> Do NOT corrupt raw tool output.
    # Append to the latest preceding user message if available.
    if role == "tool":
        placed = False
        for i in range(len(new_messages) - 2, -1, -1):
            prev_msg = dict(new_messages[i])
            if prev_msg.get("role") == "user":
                if isinstance(prev_msg.get("content"), str):
                    prev_msg["content"] = prev_msg["content"] + context_block
                elif isinstance(prev_msg.get("content"), list):
                    prev_msg["content"] = [
                        *prev_msg["content"],
                        {"type": "text", "text": context_block},
                    ]
                new_messages[i] = prev_msg
                placed = True
                break
        if not placed:
            # Fallback: append as an ephemeral user message
            new_messages.append({"role": "user", "content": context_block})
        return clean_system, new_messages, all_extracted

    # Case B: Standard String Content (User or Assistant)
    if isinstance(content, str):
        last_msg["content"] = content + context_block
        new_messages[-1] = last_msg

    # Case C: Structured Content Array (Anthropic or multimodal)
    elif isinstance(content, list):
        content_list = list(content)
        # Check if the content is solely tool_result blocks
        has_tool_results = any(
            isinstance(item, dict) and item.get("type") == "tool_result" for item in content_list
        )
        if has_tool_results:
            # Append a distinct text block to avoid corrupting tool_result schema
            content_list.append({"type": "text", "text": context_block})
        else:
            # Find last text block or append one
            appended = False
            for item in reversed(content_list):
                if isinstance(item, dict) and item.get("type") == "text":
                    item["text"] = item.get("text", "") + context_block
                    appended = True
                    break
            if not appended:
                content_list.append({"type": "text", "text": context_block})
        last_msg["content"] = content_list
        new_messages[-1] = last_msg

    return clean_system, new_messages, all_extracted
