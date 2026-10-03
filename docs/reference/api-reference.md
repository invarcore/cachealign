# Python API Reference

Comprehensive reference for the `cachealign` Python SDK.

---

## 1. Primary Entrypoints

### `cachealign.wrap(client, verbose: bool = True) -> ClientWrapper`
Wraps an Anthropic or OpenAI client instance to transparently optimize requests and track caching metrics.

#### Parameters:
* **`client`** (`Any`): An initialized instance of `anthropic.Anthropic`, `anthropic.AsyncAnthropic`, `openai.OpenAI`, or `openai.AsyncOpenAI`.
* **`verbose`** (`bool`, default `True`): If `True`, logs per-turn cache hit rates and estimated dollar savings to stderr using Rich formatting. Set to `False` in production services or headless CI.

#### Returns:
* **`ClientWrapper`**: A proxy object exposing identical methods and attributes (`.messages.create()`, `.chat.completions.create()`), with an attached `.telemetry` property.

#### Example:
```python
import anthropic
from cachealign import wrap

client = wrap(anthropic.Anthropic(), verbose=True)
response = client.messages.create(...)
```

---

## 2. Normalization Functions

### `cachealign.canonicalize_tool_schemas(tools: list[dict] | None) -> list[dict] | None`
Deterministically sorts an array of tool definitions.

* Sorts the outer list of tools alphabetically by tool name.
* Recursively sorts all internal dictionary keys and parameter properties according to RFC 8785 (UTF-16 code unit order).

#### Example:
```python
from cachealign import canonicalize_tool_schemas

clean_tools = canonicalize_tool_schemas(my_raw_tools)
```

---

### `cachealign.partition_messages_and_system(system, messages) -> tuple`
Inspects system prompt and conversation messages for volatile variables (timestamps, turn counters, session IDs).

#### Parameters:
* **`system`** (`str | list[dict] | None`): System prompt text or structured block array.
* **`messages`** (`list[dict]`): Array of conversation messages.

#### Returns:
* **`tuple`**: `(clean_system, clean_messages, extracted_volatile_tokens)`
  * `clean_system`: The invariant static system prompt stripped of volatile lines.
  * `clean_messages`: Messages with volatile tokens migrated to the tail of the last user turn.
  * `extracted_volatile_tokens`: List of string tokens detected and relocated.

---

## 3. Telemetry & FinOps Classes

### `cachealign.telemetry.reporter.SessionTelemetry`
Dataclass recording cumulative session statistics across multiple turns.

#### Attributes:
* **`turns_count`** (`int`): Total completed API turns in this session.
* **`total_tokens`** (`int`): Cumulative total tokens processed.
* **`total_cached_tokens`** (`int`): Cumulative prompt tokens read from cache.
* **`total_saved_usd`** (`float`): Total estimated financial savings in USD.
* **`overall_hit_rate`** (`float`): Cumulative percentage of tokens read from cache (`0.0` to `100.0`).

#### Example:
```python
client = wrap(anthropic.Anthropic())
# ... run 10 turns ...
print(f"Session Saved: ${client.telemetry.session.total_saved_usd:.4f}")
print(f"Overall Hit Rate: {client.telemetry.session.overall_hit_rate}%")
```

---

### `cachealign.adapters.base.UsageStats`
Dataclass returned per request turn.

#### Attributes:
* **`total_tokens`** (`int`): Input tokens + Output tokens.
* **`input_tokens`** (`int`): Base uncached input tokens.
* **`output_tokens`** (`int`): Output completion tokens.
* **`cached_tokens`** (`int`): Tokens read from cache.
* **`cache_creation_tokens`** (`int`): Tokens written to cache.
* **`cache_read_tokens`** (`int`): Tokens read from cache.
* **`estimated_cost_saved_usd`** (`float`): Estimated dollars saved on this turn.
* **`cache_hit_rate`** (`float`): Percentage of prompt tokens read from cache on this turn.