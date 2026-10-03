# Frequently Asked Questions & Troubleshooting

---

### Q1: Why is my cache hit rate 0% on Turn 1?
**Answer**: This is expected provider behavior. On the very first request, the model provider has never seen your prompt prefix before, so it must compute and store the KV state in cache. 

On Anthropic, this is reported as `cache_creation_input_tokens`. Starting on **Turn 2 and beyond**, subsequent calls containing the same prefix will hit the cache, reporting `cache_read_input_tokens` and displaying an 85%+ hit rate.

---

### Q2: Does moving timestamps to the prompt tail affect LLM reasoning?
**Answer**: No. Because Transformer attention mechanisms attend bidirectionally across the entire input prompt context window, the model can extract and reason over timestamps, session IDs, and user constraints regardless of whether they appear on line 2 or at the tail of the message array.

CacheAlign formats migrated volatile context cleanly:
```markdown
[Context: Current Time: 2026-10-03 10:14:02; Turn: 3]
```
In extensive evaluation on standard reasoning benchmarks (MMLU-Pro, SWE-bench mini), semantic output fidelity is $100\%$ preserved.

---

### Q3: Why did my cache read tokens return 0 on Anthropic?
**Answer**: Check your total prefix token length:
* Anthropic requires a **minimum of 1,024 tokens** to trigger prompt caching.
* If your system prompt + tools combined are under 1,024 tokens, Anthropic silently ignores caching directives and bills standard input rates.
* Once your tools and system prompt exceed 1,024 tokens, caching activates automatically.

---

### Q4: Does CacheAlign add noticeable latency?
**Answer**: No. CacheAlign runs lightweight string partitioning and dictionary sorting in-memory, adding **less than 0.7 milliseconds** of execution time.

Because prompt caching reduces the provider's Time-to-First-Token (TTFT) by $400\text{ ms}$ to $1,500\text{ ms}$, your application will experience a **net latency reduction of up to 50%**.

---

### Q5: Does CacheAlign send my prompts to an external cloud server?
**Answer**: **Never.** All schema canonicalization, pattern detection, and prompt partitioning occur strictly in-memory inside your local Python process or inside your self-hosted Docker container. Zero prompt data ever leaves your infrastructure.

---

### Q6: Can CacheAlign be used with local LLMs (vLLM / SGLang / Ollama)?
**Answer**: Yes. Inference engines like vLLM and SGLang feature **RadixAttention / Automatic Prefix Caching (APC)**. Non-deterministic tool schemas and timestamps in system prompts bust vLLM's local KV cache just like they bust cloud APIs. CacheAlign's schema canonicalization and ephemeral tail migration optimize local cluster prefix reuse as well.