# Deep Dive: How Prompt Caching Really Works

To understand why CacheAlign is necessary, it is important to understand the physics of Transformer KV-caching and how major cloud providers implement prompt caching.

---

## 1. The Physics of KV-Caching: Why Prefix Matching is Strict

In Transformer-based Large Language Models, input prompt tokens are converted into **Key (K)** and **Value (V)** tensor matrices during the attention calculation.

* In a standard request, every input token must attend to every preceding token. Computing the KV cache for a 25,000-token prompt requires substantial GPU memory bandwidth and compute.
* When a prompt is **cached**, the provider serializes and retains the computed KV state in high-speed GPU SRAM or host memory across consecutive calls.
* **The Catch**: Because causal self-attention is sequential, Token $N$ depends on the exact position and byte representation of Tokens $0$ through $N-1$. If **a single character at position 10 changes**, all downstream KV states from token 11 to 25,000 are mathematically invalidated and cannot be reused.

---

## 2. Provider Caching Mechanics Comparison

| Feature | Anthropic Claude | OpenAI (GPT-4o) | Google Gemini |
| :--- | :--- | :--- | :--- |
| **Detection Method** | **Explicit** (Developer marks `cache_control`) | **Automatic** (Prefix monitoring) | **Explicit & Implicit** |
| **Minimum Prefix** | **1,024 tokens** | **1,024 tokens** | **2,048 / 4,096 tokens** |
| **Cache Write Cost** | **1.25x** base rate (5m TTL) / **2.0x** (1h TTL) | Standard input rate ($1.0x) | Standard input rate ($1.0x) |
| **Cache Read Cost** | **0.10x** (90% discount) | **0.50x** (50% discount) | **0.10x** (90% discount) |
| **Storage Fee** | Free during TTL window | Free during TTL window | **$0.50 / 1M tokens / hr** |
| **Max Breakpoints** | **4 per request** | N/A (single prefix) | N/A (cached resource ID) |

---

## 3. Anthropic Break-Even Math

Because Anthropic charges a **$1.25\times$ premium** the first time content is written to the cache, you only achieve cost savings when that content is read again within the 5-minute TTL.

Let's calculate the cost for a 10,000-token system prompt across 5 turns on Claude 3.5 Sonnet ($3.00/1M base):

### Without Caching:
$$\text{Cost} = 5 \times (10,000 / 1,000,000) \times \$3.00 = \$0.150$$

### With CacheAlign (1 Write + 4 Reads):
* **Turn 1 (Write)**: $(10,000 / 1,000,000) \times \$3.75 = \$0.0375$
* **Turn 2 (Read)**: $(10,000 / 1,000,000) \times \$0.30 = \$0.0030$
* **Turn 3 (Read)**: $(10,000 / 1,000,000) \times \$0.30 = \$0.0030$
* **Turn 4 (Read)**: $(10,000 / 1,000,000) \times \$0.30 = \$0.0030$
* **Turn 5 (Read)**: $(10,000 / 1,000,000) \times \$0.30 = \$0.0030$
* **Total Cost**: $\$0.0495$

$$\text{Net Savings} = \frac{\$0.150 - \$0.0495}{\$0.150} = \mathbf{67.0\% \text{ Cost Reduction}}$$

> **The Break-Even Point**: A 5-minute cache pays for itself after just **two cache hits**. In a typical 10-turn agent session, savings surpass **80%**.

---

## 4. Why Multi-Turn Agents Break the Cache

In standard multi-turn agent frameworks:
1. **Dynamic Timestamps**: Inserting current time into system prompts invalidates the entire cache write on every turn, causing you to pay the 1.25x write penalty repeatedly without ever getting read discounts.
2. **Schema Instability**: Dictionaries serialized across different cluster nodes output different key orders:
   ```json
   // Node A:
   {"name": "search", "description": "...", "parameters": {"limit": 10, "query": "text"}}
   // Node B:
   {"description": "...", "name": "search", "parameters": {"query": "text", "limit": 10}}
   ```
   To the LLM provider, these are different byte streams, resulting in a 0% cache hit rate.
3. **Mid-Prompt Tool Injections**: Adding volatile tool results in the middle of prompts shifts the token offsets of all subsequent static blocks.

CacheAlign eliminates all three failure modes automatically.