"""
Static/Dynamic Prompt Partitioner & Ephemeral Tail Migrator.
Identifies volatile variables (timestamps, session IDs, turn counters)
in system prompts and moves them to the dynamic tail of the conversation,
preserving long byte-identical prefixes for high cache hit rates.
"""

import re
from typing import Any

# Regex patterns identifying volatile ephemeral content commonly injected into prompts
VOLATILE_PATTERNS = [
    # Full ISO timestamps or dates with time
    re.compile(
        r"(?:current\s+time|now|timestamp|current\s+date\s+and\s+time)[\s:=]+([^\n\r.]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b",
        re.IGNORECASE,
    ),
    # Session or Request IDs
    re.compile(
        r"(?:session|request|conversation|trace)[_-]id[\s:=]+([a-zA-Z0-9_-]+)", re.IGNORECASE
    ),
    # Turn or step counters
    re.compile(r"(?:turn|step)[\s:=]+(\d+)\s*(?:of\s*\d+)?", re.IGNORECASE),
]


def extract_volatile_elements(text: str) -> tuple[str, list[str]]:
    """
    Extracts volatile lines or tokens from text, returning:
    (sanitized_static_text, list_of_extracted_volatile_snippets)
    """
    if not text:
        return text, []

    extracted: list[str] = []
    lines = text.splitlines(keepends=True)
    kept_lines: list[str] = []

    for line in lines:
        matched = False
        for pattern in VOLATILE_PATTERNS:
            if pattern.search(line):
                stripped = line.strip()
                if stripped:
                    extracted.append(stripped)
                matched = True
                break
        if not matched:
            kept_lines.append(line)

    sanitized = "".join(kept_lines).strip()
    return sanitized, extracted


def partition_messages_and_system(
    system: str | list[dict[str, Any]] | None,
    messages: list[dict[str, Any]],
) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]], list[str]]:
    """
    Partitions the system prompt and message history.
    Any volatile tokens extracted from the static prefix are appended to the dynamic
    tail of the final user message.
    """
    all_extracted: list[str] = []

    # 1. Partition System Prompt if provided as string
    clean_system = system
    if isinstance(system, str):
        clean_system, extracted = extract_volatile_elements(system)
        all_extracted.extend(extracted)
    elif isinstance(system, list):
        # Anthropic structured system blocks
        new_blocks = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text, extracted = extract_volatile_elements(block.get("text", ""))
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
    # Append the extracted volatile headers to the tail of the final user turn
    new_messages = [dict(m) for m in messages]
    volatile_header = "\n[Context: " + "; ".join(all_extracted) + "]"

    last_msg = new_messages[-1]
    if isinstance(last_msg.get("content"), str):
        last_msg["content"] = last_msg["content"] + volatile_header
    elif isinstance(last_msg.get("content"), list):
        content_list = list(last_msg["content"])
        # Find last text block or append one
        appended = False
        for item in reversed(content_list):
            if isinstance(item, dict) and item.get("type") == "text":
                item["text"] = item.get("text", "") + volatile_header
                appended = True
                break
        if not appended:
            content_list.append({"type": "text", "text": volatile_header})
        last_msg["content"] = content_list

    return clean_system, new_messages, all_extracted
