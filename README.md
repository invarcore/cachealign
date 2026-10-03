# CacheAlign

<div align="center">

<h3>Autonomous Prompt Cache Optimizer & Prefix Alignment Middleware for AI Agents</h3>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests: Passing](https://img.shields.io/badge/tests-38%20passed-brightgreen.svg)](https://github.com/sagarv48/cachealign)

**Stop busting your prompt cache.** A 1-line drop-in SDK wrapper and self-hosted Rust sidecar proxy that automatically aligns AI agent prompt prefixes to achieve 85%+ cache hit rates across Anthropic, OpenAI, and Google Gemini.

[Quickstart](#-quickstart-python-sdk) • [Benchmarks](#-empirical-benchmarks) • [Architecture](#-architecture) • [Framework Integrations](#-framework-integrations) • [Rust Sidecar](#-rust-sidecar-proxy)

</div>

---

## 💥 The Problem: The Prompt Cache-Busting Epidemic

Cloud LLM providers offer **75% to 90% discounts** and up to **50% faster Time-to-First-Token (TTFT)** on prompt-cached tokens:
- **Anthropic Claude**: 90% discount on cached reads ($0.30/1M vs $3.00/1M on Sonnet 3.5).
- **OpenAI GPT-4o**: 50% discount on cached input ($1.25/1M vs $2.50/1M).
- **Google Gemini**: 75% to 90% discount on cached context reads.

However, prompt caching strictly requires **exact byte-identical prefix matching from token 0**.

In real-world multi-turn agent loops (LangGraph, CrewAI, AutoGen, custom ReAct), prompt caches silently bust on every turn due to:
1. **Volatile Injections**: Placing `Current Time: {now}` or `Turn: {step}` near the top of system prompts changes character 15, invalidating 20,000+ cached tokens downstream.
2. **JSON Schema Non-Determinism**: Python dictionary serialization produces non-deterministic key orders across worker nodes, causing prefix hash mismatches.
3. **Provider-Specific Breakpoint Friction**: Anthropic requires explicit `cache_control: {"type": "ephemeral"}` blocks with 1,024-token minimums; Gemini requires managing `CachedContent` TTLs; OpenAI requires strict 1,024-token boundary padding.

**Result**: Production agents run with **$<25\%$ cache hit rates**, burning 4x to 5x more money than necessary.

---

## 💡 The Solution: CacheAlign

CacheAlign is a zero-latency optimization layer available in two form factors:
1. **In-Process Python SDK (`pip install cachealign`)**: 1-line wrapper over Anthropic, OpenAI, or Gemini clients with in-memory execution and zero network hops.
2. **Self-Hosted Sidecar Proxy (`cachealign-proxy`)**: Compiled Rust binary for Kubernetes and polyglot microservice environments.

### Core Features
* **RFC 8785 Schema Canonicalization (JCS)**: Deterministically sorts JSON tool schemas and parameter keys lexicographically with cycle detection and anti-DoS depth limits.
* **Static/Dynamic Prompt Partitioning**: Detects volatile variables (timestamps, nonces, session counters) and automatically relocates them to the dynamic prompt tail within isolated `<cachealign_ephemeral_context>` XML blocks.
* **Provider-Native Multi-Breakpoint Injection**: Automatically inserts Anthropic's 4 `cache_control` blocks at optimal boundaries; manages Google Gemini `CachedContent` TTLs.
* **Enterprise Security & Isolation**: Injects cryptographic salts (`tenant_id`) to prevent cross-tenant **CacheProbe** timing side-channel attacks; enforces **Fail-Open Policy** (`fail_open=True`) so an optimization bug can never crash a production agent.
* **Real-Time FinOps Terminal Reporter**: Shows instant feedback on hit rate %, cached tokens, and actual dollars saved per turn.

---

## 📊 Empirical Benchmarks

Simulated 10-turn enterprise ReAct agent loop (Claude 3.5 Sonnet token pricing, 3,500-token system prompt, 15 tool definitions):

| Turn | Baseline Cost | CacheAlign Cost | Cache Hit Rate | Dollars Saved |
| :---: | :---: | :---: | :---: | :---: |
| **Turn 1** | $0.0215 | $0.0256 (Cache Write) | 0.0% | $0.0000 |
| **Turn 2** | $0.0219 | $0.0070 | **94.8%** | **$0.0149** |
| **Turn 3** | $0.0224 | $0.0075 | **92.4%** | **$0.0149** |
| **Turn 4** | $0.0228 | $0.0079 | **90.2%** | **$0.0149** |
| **Turn 5** | $0.0233 | $0.0084 | **88.0%** | **$0.0149** |
| **Turn 6** | $0.0237 | $0.0089 | **85.9%** | **$0.0149** |
| **Turn 7** | $0.0242 | $0.0093 | **84.0%** | **$0.0149** |
| **Turn 8** | $0.0246 | $0.0097 | **82.1%** | **$0.0149** |
| **Turn 9** | $0.0250 | $0.0102 | **80.3%** | **$0.0149** |
| **Turn 10** | $0.0255 | $0.0106 | **78.6%** | **$0.0149** |
| **TOTAL** | **$0.2348** | **$0.1052** | **74.7% (Overall)** | **$0.1295 (55.2% Saved)** |

*Run the benchmark locally: `python benchmarks/eval_react_loop.py`*

---

## 🚀 Quickstart (Python SDK)

### 1. Installation
```bash
pip install cachealign
# Or with specific providers:
pip install "cachealign[anthropic,openai,gemini]"
```

### 2. Anthropic Claude (1-Line Wrap)
```python
import anthropic
from cachealign import wrap

# Wrap any Anthropic client (Sync or Async)
client = wrap(anthropic.Anthropic())

# Works with standard messages.create:
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="You are an enterprise research assistant.\nCurrent Time: 2026-10-03 10:00:00",
    tools=[{"name": "search_db", "description": "...", "input_schema": {...}}],
    messages=[{"role": "user", "content": "Analyze Q3 revenue."}],
)

# Works with streaming context managers:
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### 3. OpenAI GPT-4o
```python
import openai
from cachealign import wrap

client = wrap(openai.OpenAI())

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a coding assistant.\nSession-ID: sess_9481"},
        {"role": "user", "content": "Refactor this function."},
    ],
    tools=[...],
)
```

### 4. Google Gemini (google-genai SDK)
```python
from google import genai
from cachealign import wrap

client = wrap(genai.Client())

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain quantum computing.",
    config={
        "system_instruction": "You are a physics professor.\nCurrent Time: 2026-10-03",
        "tools": [...],
    },
)
```

---

## 🔌 Framework Integrations

### LangChain & LangGraph
```python
from cachealign.integrations import CacheAlignCallbackHandler

# Attach to any RunnableConfig or LangGraph invocation:
handler = CacheAlignCallbackHandler()
response = agent.invoke(
    {"messages": [("user", "Analyze customer churn")]},
    config={"callbacks": [handler]},
)
```

### LiteLLM Proxy
In your LiteLLM `config.yaml`:
```yaml
litellm_settings:
  callbacks: ["cachealign.integrations.litellm.CacheAlignLiteLLMHandler"]
```

---

## 🦀 Rust Sidecar Proxy

For polyglot stacks (Node.js, Go, Java, Python) or Kubernetes deployments:

### Run via Docker
```bash
docker run -p 8080:8080 \
  -e UPSTREAM_ANTHROPIC="https://api.anthropic.com" \
  -e UPSTREAM_OPENAI="https://api.openai.com" \
  ghcr.io/sagarv48/cachealign-proxy:latest
```

### Route Agent Traffic via Localhost
```python
import anthropic

# Simply point your SDK's base_url to localhost:8080
client = anthropic.Anthropic(base_url="http://localhost:8080")
response = client.messages.create(...)
```

---

## 🔒 Security & STRIDE Threat Model

CacheAlign incorporates built-in enterprise defense-in-depth:
- **Fail-Open Policy**: Optimization errors log a diagnostic warning and fallback cleanly without crashing.
- **Tenant Salt Isolation (`tenant_id`)**: Injects deterministic cryptographic salts to prevent cross-tenant **CacheProbe** timing side-channel attacks.
- **Schema Recursion Guard**: Caps dictionary nesting at depth 20 with cycle detection to prevent schema recursion bombs.
- **Safe Delimiters**: Volatile context is encapsulated within `<cachealign_ephemeral_context>` with closing tag escaping.

---

## 📄 License

MIT © [Vinay Kumar Ksheera Sagar](https://github.com/sagarv48)