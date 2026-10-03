"""
Example: Multi-turn ReAct agent loop with CacheAlign.
Demonstrates how CacheAlign transparently preserves prompt caching
even when dynamic timestamps and volatile IDs are passed into each turn.
"""

from datetime import datetime

from cachealign import wrap


class MockAnthropicClient:
    """Mock client simulating Anthropic API responses with prompt caching metrics."""

    class Messages:
        def __init__(self):
            self.turn = 0

        def create(self, **kwargs):
            self.turn += 1
            # Simulate Anthropic response usage
            from types import SimpleNamespace

            is_first_turn = self.turn == 1
            return SimpleNamespace(
                id=f"msg_turn_{self.turn}",
                role="assistant",
                content=[{"type": "text", "text": f"Executed step {self.turn} successfully."}],
                usage=SimpleNamespace(
                    input_tokens=250 if not is_first_turn else 12500,
                    output_tokens=85,
                    cache_creation_input_tokens=12500 if is_first_turn else 0,
                    cache_read_input_tokens=12500 if not is_first_turn else 0,
                ),
            )

    def __init__(self):
        self.messages = self.Messages()


def main():
    print("Initializing Multi-Turn Agent with CacheAlign...")
    client = wrap(MockAnthropicClient())

    tools = [
        {
            "name": "fetch_user",
            "description": "Fetches user data",
            "input_schema": {"type": "object"},
        },
        {
            "name": "query_database",
            "description": "Queries database",
            "input_schema": {"type": "object"},
        },
    ]

    for turn in range(1, 6):
        # Notice: Volatile timestamp dynamically passed every turn!
        system_prompt = (
            f"You are an enterprise research agent.\n"
            f"Current Timestamp: {datetime.now().isoformat()}\n"
            f"Session Turn: {turn}\n"
            f"Instructions: " + ("Follow safety rules and execute tasks. " * 200)
        )

        _ = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            system=system_prompt,
            tools=tools,
            messages=[{"role": "user", "content": f"Turn {turn}: Process financial records."}],
        )


if __name__ == "__main__":
    main()
