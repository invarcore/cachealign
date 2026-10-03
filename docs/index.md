# Welcome to CacheAlign

**CacheAlign** is an open-source prompt topology optimizer and prefix alignment middleware designed for multi-turn AI agents and production LLM applications.

By automatically restructuring prompt payloads at runtime, CacheAlign enables your agents to achieve **85%+ prompt-cache hit rates** across **Anthropic Claude**, **OpenAI GPT-4o**, and **Google Gemini**—reducing inference costs by up to **80%** and cutting Time-to-First-Token (TTFT) in half with **zero application code rewrites**.

---

## ⚡ The Problem: The Prompt Cache-Busting Epidemic

Cloud LLMs offer deep discounts on cached prompt tokens:
* **Anthropic Claude 3.5**: 90% discount on cache reads ($0.30/1M vs $3.00/1M base).
* **OpenAI GPT-4o**: 50% discount on cached reads ($1.25/1M vs $2.50/1M base).
* **Google Gemini**: 90% discount on context cache reads.

However, all prompt caching engines require **strict byte-exact prefix matching from index 0**.

In real-world multi-turn agent loops (LangGraph, CrewAI, AutoGen, custom ReAct), prompt caches silently bust because:
1. **Dynamic Timestamps**: Inserting `Current Time: 2026-10-03 10:15:32` at the top of a system prompt changes character 15, invalidating all subsequent 25,000+ cached tokens.
2. **JSON Schema Non-Determinism**: Python dictionaries serialized to JSON produce arbitrary key orderings across worker processes, breaking byte hashes.
3. **Provider Syntax Fragmentation**: Manually placing Anthropic's 4 `cache_control` blocks at exact 1,024-token boundaries across dynamic conversational turns is tedious and brittle.

As a result, most production agents operate with **less than 25% cache hit rates**.

---

## 💡 How CacheAlign Solves This

```mermaid
graph LR
    A["Your Agent Loop"] -->|calls| B["cachealign.wrap(client)"]
    B --> C["1. Sorts JSON Schemas (RFC 8785)"]
    C --> D["2. Relocates Timestamps to Tail"]
    D --> E["3. Injects Cache Breakpoints"]
    E -->|Optimized Call| F["Anthropic / OpenAI / Gemini"]
    F -->|85%+ Cache Hits| G["80% Cost Reduction & 50% Lower TTFT"]
```

CacheAlign intercepts outbound model calls in-memory:
* **Canonicalizes JSON Tool Schemas**: Sorts keys lexicographically according to RFC 8785 so schemas always hash to identical bytes.
* **Partitions Volatile Variables**: Automatically detects timestamps, turn counters, and session IDs in system prompts and shifts them to a synthesized context footer on the latest user message.
* **Injects Provider-Native Breakpoints**: Dynamically calculates token counts and inserts optimal caching headers (such as Anthropic's 4 `cache_control` blocks).

---

## 🚀 60-Second Quickstart

```python
import anthropic
from cachealign import wrap

# Wrap your existing client in 1 line
client = wrap(anthropic.Anthropic())

# Run your agent loop normally
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="Current Time: 2026-10-03 10:00:00\n\nYou are an enterprise research agent...",
    tools=my_tools,
    messages=conversation_history,
)

# Output in your terminal:
# [CacheAlign] Turn 3 (claude-3-5-sonnet) | Hit: 91.4% (18,432 cached) | Saved: $0.0829 (Session: $0.341)
```

---

## 📖 Documentation Guide

* **[Getting Started: Quickstart](getting-started/quickstart.md)** — Installation and 5-minute setup.
* **[Concepts: Prompt Caching Mechanics](concepts/prompt-caching-mechanics.md)** — In-depth breakdown of provider caching rules, token thresholds, and write penalties.
* **[Concepts: Architecture](concepts/architecture.md)** — How RFC 8785 canonicalization and ephemeral tail migration work.
* **[Guides: Framework Integrations](guides/framework-integrations.md)** — Using CacheAlign with LangGraph, CrewAI, AutoGen, and raw ReAct loops.
* **[Guides: CLI Reference](guides/cli-reference.md)** — Linting prompts and canonicalizing schemas from the command line.
* **[Reference: API Reference](reference/api-reference.md)** — Complete Python SDK class and function documentation.
* **[Troubleshooting: FAQ](troubleshooting/faq.md)** — Common questions, reasoning fidelity, and debugging 0% hit rates.