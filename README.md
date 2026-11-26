<div align="center">

<img src="assets/logo.png" alt="CacheAlign Logo" width="320" />

# CacheAlign

<h3>Autonomous Prompt Cache Optimizer & Prefix Alignment Middleware for AI Agents</h3>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Status](https://img.shields.io/badge/Status-Pre--release%20%2F%20Source%20Install-orange.svg)](https://github.com/invarcore/cachealign)
[![Live CI Evaluation](https://github.com/invarcore/cachealign/actions/workflows/live-eval.yml/badge.svg)](https://github.com/invarcore/cachealign/actions/workflows/live-eval.yml)

**Stop busting your prompt cache.** A 1-line drop-in SDK wrapper and self-hosted Rust sidecar proxy that deterministically stabilizes AI agent prompt prefixes across Anthropic, OpenAI, and Google Gemini (*see simulation and live smoke caveats below*).

[Quickstart](#-quickstart-python-sdk) • [Benchmarks](#-empirical-benchmarks) • [Architecture](#-architecture) • [Framework Integrations](#-framework-integrations) • [Rust Sidecar](#-rust-sidecar-proxy)

</div>

---

## 💥 The Problem: The Prompt Cache-Busting Epidemic

Cloud LLM providers offer **75% to 90% discounts** (up to 95% on Sonnet 5.5 / Haiku 5.5) on prompt-cached prefix tokens:
- **Anthropic Claude**: 90% discount on cached reads on Claude Sonnet 4.6 (`$0.30/1M` cached vs `$3.00/1M` base input; `$3.75/1M` 5-min write); current Sonnet 5.5 bills `$2.00/1M` base input (`$0.10/1M` cached) and Haiku 5.5 bills `$0.10/1M` base input ($\le 100\text{K}$).
- **OpenAI GPT-4o**: 50% discount on cached input ($1.25/1M vs $2.50/1M).
- **Google Gemini**: 75% to 90% discount on cached context reads.

However, prompt caching strictly requires **exact byte-identical prefix matching from token 0**.

In real-world multi-turn agent loops (LangGraph, CrewAI, AutoGen, custom ReAct), prompt caches silently bust on every turn due to:
1. **Volatile Injections**: Placing `Current Time: {now}` or `Turn: {step}` near the top of system prompts changes character 15, invalidating 20,000+ cached tokens downstream.
2. **JSON Schema Non-Determinism**: Python dictionary serialization produces non-deterministic key orders across worker nodes, causing prefix hash mismatches.
3. **Provider-Specific Breakpoint Friction**: Anthropic requires explicit `cache_control: {"type": "ephemeral"}` blocks with 1,024-token minimums; Gemini requires managing `CachedContent` TTLs; OpenAI requires strict 1,024-token boundary padding.

**Illustration**: Without prefix alignment, single-character header mutations invalidate downstream KV cache blocks on every turn, multiplying uncached prefill spend.

---

## 💡 The Solution: CacheAlign

CacheAlign is a zero-latency optimization layer available in two form factors:
1. **In-Process Python SDK (Pre-release / Git Install)**: 1-line wrapper over Anthropic, OpenAI, or Gemini clients with in-memory execution and zero network hops.
2. **Self-Hosted Sidecar Proxy (`cachealign-proxy`)**: Compiled Rust binary for Kubernetes and polyglot microservice environments.

### Core Features
* **RFC 8785 Schema Canonicalization (JCS)**: Deterministically canonicalizes tool schema parameter keys lexicographically under RFC 8785 (JCS) and sorts tool definitions by name, eliminating non-deterministic dictionary serialization with cycle detection and anti-DoS depth limits.
* **Static/Dynamic Prompt Partitioning**: Detects volatile variables (timestamps, nonces, session counters) and automatically relocates them to the dynamic prompt tail within isolated `<cachealign_ephemeral_context>` XML blocks.
* **Provider-Native Multi-Breakpoint Injection**: Automatically inserts Anthropic's 4 `cache_control` blocks at optimal boundaries; manages Google Gemini `CachedContent` TTLs.
* **Enterprise Security & Isolation**: Injects cryptographic salts (`tenant_id`) to prevent cross-tenant **CacheProbe** timing side-channel attacks; enforces **Fail-Open Policy** (`fail_open=True`) so an optimization bug can never crash a production agent.
* **Zero Synthetic Padding Guardrail**: Enforces strictly zero synthetic padding tokens on closed-tokenizer APIs, eliminating model prompt boundary distortion.
* **Multi-Provider Smoke Probes**: Automated daily smoke testing across Anthropic, OpenAI, AWS Bedrock, and NVIDIA NIM GPU endpoints (`observatory/run_daily_observatory.py`).
* **Real-Time FinOps Terminal Reporter**: Shows instant feedback on hit rate %, cached tokens, and actual dollars saved per turn.

---

## 📊 Empirical Benchmarks

Offline cost simulation: 10-turn enterprise ReAct agent loop (Claude Sonnet 4.6 pricing ratios; 3,500-char system instructions + 15 tool definitions tokenized via `estimate_tokens`):

| Turn | Baseline Cost | CacheAlign Cost | Cache Hit Rate | Dollars Saved |
| :---: | :---: | :---: | :---: | :---: |
| **Turn 1** | $0.0113 | $0.0132 (Cache Write) | 0.0% | $0.0000 |
| **Turn 2** | $0.0115 | $0.0056 | **94.7%** | **$0.0059** |
| **Turn 3** | $0.0116 | $0.0057 | **92.8%** | **$0.0059** |
| **Turn 4** | $0.0118 | $0.0059 | **91.0%** | **$0.0059** |
| **Turn 5** | $0.0119 | $0.0060 | **89.3%** | **$0.0059** |
| **Turn 6** | $0.0121 | $0.0061 | **87.6%** | **$0.0059** |
| **Turn 7** | $0.0122 | $0.0063 | **86.0%** | **$0.0059** |
| **Turn 8** | $0.0124 | $0.0064 | **84.4%** | **$0.0059** |
| **Turn 9** | $0.0125 | $0.0066 | **82.9%** | **$0.0059** |
| **Turn 10** | $0.0126 | $0.0067 | **81.5%** | **$0.0059** |
| **TOTAL** | **$0.1199** | **$0.0685** | **71.3% (Overall)** | **$0.0514 (42.9% Saved)** |

*Run the offline simulation locally: `python benchmarks/eval_react_loop.py`*

### Historical Single-Run Live Hardware Trace (NVIDIA NIM / Nemotron 3 Ultra 550B — Methodology Caveat)

Archived single-run GitHub Actions CI trace on NVIDIA NIM ([Run #43703220052](https://github.com/invarcore/cachealign/actions/runs/43703220052)) across 4 arms on the Enterprise Data Agent workload. *Note on caveats uncovered in audit:* (a) arms ran sequentially without cross-arm prefix isolation, causing Arm 3b Turn 1 to reuse warm prefix blocks cached by Arm 3a; (b) `Cost` used proxy Anthropic `$3.00/$3.75/$0.30` accounting rather than native NIM billing and double-charged uncached cold tokens (`cache_write + base_input`) in the legacy script; and (c) latency measured blocking non-streaming completion time rather than true streaming TTFT (explaining why Arm 3a/3b total response time exceeded Arm 1).

| Arm | Architecture / Staging | Input Tokens | Cached Tokens | CTR (%) | Proxy Cost (Legacy Bug) | Non-Streaming Duration |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Arm 1** | Raw Baseline (Shuffled tools, volatile timestamp at token 0) | 11,576 | 0 | **0.0%** | $0.0856 (archived bug) | 6,207 ms |
| **Arm 2** | Naive Provider (Developer prompt caching, unaligned header) | 11,576 | 0 | **0.0%** | $0.0856 (archived bug) | 5,340 ms |
| **Arm 3a** | **CacheAlign Direct** (`OpenAIAdapter`, canonical tools, tail migration) | 11,614 | 8,768 | **75.5%** | $0.0187 (archived) | 8,456 ms |
| **Arm 3b** | **CacheAlign LiteLLM** (reused warm prefix from Arm 3a) | 11,614 | 8,768 | **75.5%** | $0.0187 (archived) | 11,185 ms |

*In isolated 2-turn smoke probes (`cache_probe_runner.py`), warm-turn response latency dropped from **7,550 ms to 1,970 ms (-73.9%)** with **82.2% CTR** (note that historical row 1 of `observatory/cache_probe_history.jsonl` shows pre-warmed `cache_read_tokens: 4384` on `cold_seed` due to an un-salted prior prefix). Live multi-run provider re-validation is pending API credits.*

### Offline Trace-Driven KV-Cache Simulation Matrix (N=30 Deterministic Replay Iterations)

Statistically evaluated across $N=30$ deterministic trace replay iterations per provider/workload arm with Student's $t$ confidence bounds ($\nu=29, t_{0.025, 29} = 2.045$):

| Provider | Workload | Arm | Cached Token Ratio (CTR) | Cost Savings vs Raw | Invalidation Penalty | p50 TTFT (Simulated Hardware) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Anthropic** | Enterprise Data Agent | Arm 1 (Raw) | 0.0% ± 0.0% | — | — | 373.5 ± 3.4 ms |
| | | Arm 2 (Naive) | 0.0% ± 0.0% | — | +19.3% | 400.5 ± 5.9 ms |
| | | **Arm 3a (Direct)** | **86.4% ± 0.0%** | **59.7% ± 0.0%** | — | **50.3 ± 1.0 ms** |
| | | **Arm 3b (LiteLLM)** | **86.6% ± 0.0%** | **60.1% ± 0.0%** | — | **50.0 ± 1.2 ms** |
| **OpenAI** | Enterprise Data Agent | Arm 1 (Raw) | 0.0% ± 0.0% | — | — | 374.8 ± 3.9 ms |
| | | Arm 2 (Naive) | 0.0% ± 0.0% | — | — | 398.9 ± 3.9 ms |
| | | **Arm 3a (Direct)** | **86.7% ± 0.0%** | **34.5% ± 0.0%** | — | **49.7 ± 0.9 ms** |
| | | **Arm 3b (LiteLLM)** | **86.6% ± 0.0%** | **35.8% ± 0.0%** | — | **49.3 ± 1.1 ms** |
| **vLLM** | Enterprise Data Agent | Arm 1 (Raw) | 0.0% ± 0.0% | — | — | 379.9 ± 4.0 ms |
| | | Arm 2 (Naive) | 0.0% ± 0.0% | — | — | 398.1 ± 4.5 ms |
| | | **Arm 3a (Direct)** | **86.4% ± 0.0%** | **59.5% ± 0.0%** | — | **50.1 ± 1.0 ms** |
| | | **Arm 3b (LiteLLM)** | **86.6% ± 0.0%** | **60.0% ± 0.0%** | — | **49.4 ± 0.9 ms** |

> **Measurement Methodology & Latency Physics (Synthetic / Non-Live):**
> This table reflects an **offline trace-replay simulation** ($N=30$ iterations) using `HardenedKVCacheSimulator` (`execution_mode: deterministic_simulation`). Because input token sequences are deterministic across iterations, token-accounting variance is $\pm 0.0\%$. Simulated p50 TTFT ($\sim 50\text{ ms}$) is a synthetic uniform draw modeling cached block lookup vs $\sim 375\text{ ms}$ prefill, **not** live API calls over the network. Live multi-run provider validation across current models remains open and is required before claiming live cost savings.
> 
> **Task Success & Model Behavior Guardrail:** Tool declaration canonicalization alters the sequence in which tool definitions are presented to the model. While syntactic tool execution schemas remain identical under RFC 8785, token ordering can influence tool choice in edge cases. Live multi-turn task success benchmarking across complex agentic workflows has not yet been executed and remains an active requirement before claiming cost reduction without output divergence.

*Run the simulation locally: `python benchmarks/run_multirun_benchmark.py --iterations 30 --workload all --provider all`*

### Live Multi-Run Statistical Benchmark (NVIDIA NIM, N=30 Iterations)

Statistically evaluated across $N=30$ **live API** iterations (`240` logged turns, `2` turns per run, `4,384 / 5,849` warm prefix tokens cached per Turn 2, salted per-iteration prefixes, `95%` Student's $t$ confidence bounds) against `nvidia/nemotron-3-ultra-550b-a55b` ([multirun_live_summary.json](benchmarks/results/multirun_live_summary.json)):

| Arm | Cached Token Ratio (CTR) [Mean ± 95% CI] | Cached Tokens (Turn 2) | Live Hosted p50 TTFT [Mean ± 95% CI] |
| :--- | :---: | :---: | :---: |
| **Arm 1 (`raw_baseline`)** | `0.0% ± 0.00%` | `0 / 5,827` | `10,807.6 ± 1,661.9 ms` (`5,024.7–27,625.5 ms`) |
| **Arm 2 (`naive_provider`)** | `0.0% ± 0.00%` | `0 / 5,827` | `13,290.1 ± 2,149.4 ms` (`6,721.1–32,785.1 ms`) |
| **Arm 3a (`cachealign_direct`)** | **`37.6% ± 0.00%`** (`37.60%`) | **`4,384 / 5,849`** | `11,945.6 ± 2,682.2 ms` (`5,714.5–36,611.5 ms`) |
| **Arm 3b (`cachealign_litellm`)** | **`37.6% ± 0.00%`** (`37.58%`) | **`4,384 / 5,849`** | `13,428.5 ± 4,122.2 ms` (`4,095.5–59,317.7 ms`) |

*Live Scope & Credit Boundary:* Over 2 turns on NVIDIA NIM (`min_prefix ~ 4,000` tokens), Turn 1 populates the cache (`0%`) and Turn 2 achieves ~75% warm prefix reuse (`37.6%` overall CTR across 2 turns; `rotating_rag` and `coding_loop` workloads remain below NIM's ~4K caching threshold). Multi-provider paid `N=30` runs across OpenRouter/Anthropic/OpenAI/Gemini remain pending paid API credits (initial OpenRouter probe hit `HTTP 402 Payment Required` at `$0.00` account balance). Resume-safe execution with periodic summary checkpoints (`--resume --checkpoint-every 5`) is supported in `benchmarks/run_multirun_benchmark.py`.

---

## 🚀 Quickstart (Python SDK)

### 1. Installation

> **Note:** PyPI publication is pre-release / source install only. Install directly from Git:

```bash
# Source install via pip
pip install "git+https://github.com/invarcore/cachealign.git"

# Or with uv
uv add "git+https://github.com/invarcore/cachealign.git"

# Or with optional provider extras
pip install "cachealign[anthropic,openai,gemini] @ git+https://github.com/invarcore/cachealign.git"
```

### 2. Anthropic Claude (1-Line Wrap)
```python
import anthropic
from cachealign import wrap

# Wrap any Anthropic client (Sync or Async)
client = wrap(anthropic.Anthropic())

# Works with standard messages.create:
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are an enterprise research assistant.\nCurrent Time: 2026-10-03 10:00:00",
    tools=[{"name": "search_db", "description": "...", "input_schema": {...}}],
    messages=[{"role": "user", "content": "Analyze Q3 revenue."}],
)

# Works with streaming context managers:
with client.messages.stream(
    model="claude-sonnet-4-6",
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
  ghcr.io/invarcore/cachealign-proxy:latest
```

### Route Agent Traffic via Localhost
```python
import anthropic

# Simply point your SDK's base_url to localhost:8080
client = anthropic.Anthropic(base_url="http://localhost:8080")
response = client.messages.create(...)
```

### CLI Proxy Wrapper (`cachealign exec`)
Export `HTTPS_PROXY`, `HTTP_PROXY`, `NODE_USE_ENV_PROXY=1`, and `NODE_EXTRA_CA_CERTS` automatically for Node.js and terminal agent CLIs:
```bash
cachealign exec --proxy-url http://127.0.0.1:8080 -- copilot
```
> Forward-proxy `HTTP CONNECT` TLS tunneling and `cachealign-proxy ca init` in the Rust sidecar are tracked in [Issue #1](https://github.com/invarcore/cachealign/issues/1).

---

---

## 🧪 Hermetic Verification with Docker Compose

Validate CacheAlign's end-to-end prompt caching, TTFT latency reductions, and fail-open guarantees hermetically without external API keys or GPU runtimes:

```bash
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from smoke-test
```

This spins up:
1. `mock-cache-server`: Lightweight local KV-cache simulation server emulating real Anthropic & OpenAI prefix caching with latency simulation.
2. `smoke-test`: Isolated client container running multi-turn ReAct loops across the bridge network to verify **~88% TTFT reduction** and **1,000 cached tokens**.

## ❓ Frequently Asked Questions (FinOps Physics)

### Why can't I just sort my tools with `sorted()` in Python?
Sorting the top-level tool list with `sorted(tools, key=lambda x: x['name'])` is necessary but insufficient to guarantee provider KV-cache hits:
1. **Nested Schema Dictionaries**: Enterprise tools feature deeply nested JSON schemas (`input_schema.properties.<param>.properties`). Python dictionaries preserve insertion order; any variation in dynamic dictionary construction or framework serialization across turns breaks byte-level identity. CacheAlign performs recursive object key canonicalization.
2. **RFC 8785 Canonical Encoding**: CacheAlign applies RFC 8785 (JSON Canonicalization Scheme) to serialize tool definitions into byte-identical UTF-8 wire representations while preserving semantic array ordering (Section 3.2.3).
3. **Cross-Provider Schema Mapping**: Normalizes tool schema representations across Anthropic (`name`, `input_schema`), OpenAI (`type: "function"`, `function: {name, parameters}`), and Gemini (`function_declarations`).

### What is "Volatile Tail Migration" and why does token position matter?
Provider KV caches (such as Anthropic prompt caching, OpenAI automatic caching, and vLLM PagedAttention) evaluate token sequences sequentially starting from index 0. If a runtime framework injects dynamic context (e.g. `Current Time: 2026-11-26 11:53:00`, transient user IDs, or ephemeral request headers) at token index 0, the KV-cache engine treats the entire downstream prompt as a cache miss ($0\%$ CTR). 

CacheAlign partitions the request:
- **Aligned Prefix ($[0:m]$)**: Static system instructions and canonical tool schemas are locked at the beginning of the context window.
- **Ephemeral Tail ($[m:]$)**: Volatile timestamps and transient session context are migrated to the prompt tail enclosed in `<cachealign_ephemeral_context>`.

This stabilizes prefix matching on subsequent turns so cached tokens hit the warm-read tier rather than paying cold prefill rates (in the offline simulation matrix above, saving $\sim 59\%$–$60\%$ vs raw uncached Anthropic prompts).

### What is the "Cache Invalidation Surcharge"?
On Anthropic Claude, setting `cache_control: {"type": "ephemeral"}` incurs an explicit **1.25× cache write surcharge** on first write (\$3.75/Mtok vs \$3.00/Mtok base input on Claude Sonnet 4.6). 

If an application's prompt prefix changes on every turn due to unaligned tools or volatile timestamps, the inference engine creates a new cache entry on *every turn* without ever reading from cache. Instead of saving money, the application pays the 1.25× write fee on every request—resulting in a modeled **+19.3% net cost penalty** over uncached prompts in our offline simulation. CacheAlign preserves prefix stability so turns 2–10 hit the $0.10\times$ cache read tier (\$0.30/Mtok on Sonnet 4.6).

### Can I deploy CacheAlign without modifying client application code?
Yes. CacheAlign supports passive infrastructure deployment:
- **LiteLLM Gateway Proxy**: Add `"cachealign.integrations.litellm.CacheAlignLiteLLMHandler"` to your LiteLLM `config.yaml` `callbacks`. Every upstream agent request routed through the proxy is stabilized automatically.
- **Docker Sidecar Proxy**: Run `ghcr.io/invarcore/cachealign-proxy:latest` on localhost:8080 for polyglot runtimes (Node.js, Go, Rust, Java).
- **LangChain / LangGraph**: Attach `CacheAlignCallbackHandler()` to any `RunnableConfig`.

---

## 🔒 Security & STRIDE Threat Model

CacheAlign incorporates built-in enterprise defense-in-depth:
- **Fail-Open Policy**: Optimization errors log a diagnostic warning and fallback cleanly without crashing.
- **Tenant Salt Isolation (`tenant_id`)**: Injects deterministic cryptographic salts to prevent cross-tenant **CacheProbe** timing side-channel attacks.
- **Schema Recursion Guard**: Caps dictionary nesting at depth 20 with cycle detection to prevent schema recursion bombs.
- **Safe Delimiters**: Volatile context is encapsulated within `<cachealign_ephemeral_context>` with closing tag escaping.

---

---

## 🏛️ Invarcore Verification Fabric

This engine is part of the **[Invarcore](https://invarcore.com)** enterprise verification fabric. Invarcore develops mathematical invariants, cryptographic policy contracts, and execution runtimes for autonomous AI systems.

* **Official Website & Architecture**: [https://invarcore.com](https://invarcore.com)
* **Technical Whitepapers & Invariant Specs**: [https://invarcore.com/#whitepapers](https://invarcore.com/#whitepapers)
* **GitHub Organization**: [https://github.com/invarcore](https://github.com/invarcore)
* **Security & Vulnerability Disclosure**: [security@invarcore.com](mailto:security@invarcore.com)

## 📄 License

MIT © [Vinay Kumar Ksheera Sagar](https://github.com/sagarv48)