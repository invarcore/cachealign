# Quickstart Guide

Get started with **CacheAlign** in under 60 seconds.

---

## 1. Installation

Install CacheAlign via pip:

```bash
# Basic installation
pip install cachealign

# Or install with optional provider SDK dependencies
pip install "cachealign[anthropic]"
pip install "cachealign[openai]"
pip install "cachealign[all]"
```

---

## 2. Quickstart with Anthropic Claude

Wrap your standard `anthropic.Anthropic` client with `cachealign.wrap`:

```python
import anthropic
from cachealign import wrap

# Initialize and wrap your client
raw_client = anthropic.Anthropic()
client = wrap(raw_client)

# Define tools (keys do not need to be sorted manually)
tools = [
    {
        "name": "lookup_customer",
        "description": "Retrieve customer details",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "include_history": {"type": "boolean"}
            }
        }
    }
]

# Run a multi-turn conversation
conversation = [
    {"role": "user", "content": "Fetch details for customer CUST-1049."}
]

# Notice: Even if you put a dynamic timestamp in the system prompt,
# CacheAlign automatically moves it to the dynamic tail so the prefix remains cached!
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="You are an enterprise support assistant.\nCurrent Time: 2026-10-03 10:14:00",
    tools=tools,
    messages=conversation,
)

print(response.content[0].text)
```

**Terminal Output:**
```
[CacheAlign] Turn 1 (claude-3-5-sonnet-20241022) | Hit: 0.0% (0 cached) | Saved: $0.0000 (Session: $0.000)
```
On subsequent turns:
```
[CacheAlign] Turn 2 (claude-3-5-sonnet-20241022) | Hit: 94.2% (14,200 cached) | Saved: $0.0383 (Session: $0.038)
[CacheAlign] Turn 3 (claude-3-5-sonnet-20241022) | Hit: 94.8% (15,100 cached) | Saved: $0.0408 (Session: $0.079)
```

---

## 3. Quickstart with OpenAI GPT-4o

Wrap your standard `openai.OpenAI` client:

```python
import openai
from cachealign import wrap

client = wrap(openai.OpenAI())

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_metrics",
            "parameters": {
                "values": {"type": "array", "items": {"type": "number"}}
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a data analysis agent.\nSession: 84920"},
        {"role": "user", "content": "Compute variance on the dataset."}
    ],
    tools=tools,
)

print(response.choices[0].message.content)
```

---

## 4. Configuration Options

### Silent Mode
To disable terminal output (e.g. in production microservices or CI/CD pipelines), pass `verbose=False`:

```python
client = wrap(anthropic.Anthropic(), verbose=False)
```

### Inspecting Session Telemetry Programmatically
You can access telemetry metrics programmatically at any time:

```python
stats = client.telemetry.session

print(f"Total turns: {stats.turns_count}")
print(f"Total cached tokens: {stats.total_cached_tokens:,}")
print(f"Overall cache hit rate: {stats.overall_hit_rate}%")
print(f"Total estimated savings: ${stats.total_saved_usd:.4f}")
```