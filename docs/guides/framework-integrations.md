# Framework Integration Guides

CacheAlign is framework-agnostic. Because it wraps the underlying client instance (`anthropic.Anthropic`, `openai.OpenAI`), any framework that accepts a custom client or model instance works with CacheAlign out of the box.

---

## 1. LangGraph & LangChain

In LangChain and LangGraph, pass the wrapped client or configure the model with the wrapped client.

### Anthropic Claude with LangChain / LangGraph:
```python
import anthropic
from langchain_anthropic import ChatAnthropic
from cachealign import wrap

# 1. Wrap the raw client
wrapped_client = wrap(anthropic.Anthropic())

# 2. Pass the wrapped client directly to ChatAnthropic
llm = ChatAnthropic(
    model_name="claude-3-5-sonnet-20241022",
    client=wrapped_client
)

# 3. Use in your LangGraph ReAct agent
# All graph cycles and tool turns now benefit from 85%+ cache hit rates!
```

---

## 2. CrewAI

CrewAI agents frequently execute multi-step hierarchical delegation. Wrap the underlying LLM client:

```python
from crewai import Agent, Crew, Process, Task
from langchain_anthropic import ChatAnthropic
import anthropic
from cachealign import wrap

wrapped_client = wrap(anthropic.Anthropic())
custom_llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", client=wrapped_client)

researcher = Agent(
    role="Senior Market Analyst",
    goal="Discover high-growth AI infrastructure tools",
    backstory="You are an expert tech analyst.",
    llm=custom_llm,
    tools=[]
)
```

---

## 3. OpenAI Client Integration (Native & Assistants)

```python
import openai
from cachealign import wrap

# Wrap OpenAI client
client = wrap(openai.OpenAI())

# Standard OpenAI client usage
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a customer assistant.\nCurrent Time: 2026-10-03"},
        {"role": "user", "content": "Help me reset my password."}
    ]
)
```

---

## 4. Custom ReAct Loops (Zero-Dependency Python)

If you build autonomous agents with native Python `while`-loops, CacheAlign requires zero architectural changes:

```python
import anthropic
from cachealign import wrap

client = wrap(anthropic.Anthropic())

history = [{"role": "user", "content": "Solve the user query."}]

while not task_finished:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=MASSIVE_SYSTEM_INSTRUCTIONS, # 20,000+ tokens
        tools=ALL_TOOLS,                   # 30+ tools
        messages=history,
    )
    
    # Process tool calls or assistant reply
    # CacheAlign automatically ensures previous turns stay cached!
    if response.stop_reason == "tool_use":
        execute_tool_and_append_result(history, response)
    else:
        break
```