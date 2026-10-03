"""Integration tests verifying cross-provider functional parity and end-to-end capabilities."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from cachealign import (
    AnthropicAdapter,
    CacheAlignConfig,
    GeminiAdapter,
    canonicalize_object,
    canonicalize_tool_schemas,
    extract_volatile_elements,
    wrap,
)


class TestCrossProviderCanonicalizationParity:
    """Verifies RFC 8785 schema canonicalization produces identical representations."""

    def test_shuffled_tool_definitions_canonicalize_identically(self):
        tools_variant_a = [
            {
                "name": "lookup_customer",
                "description": "Lookup customer by account ID.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string", "description": "Customer UUID"},
                        "include_billing": {"type": "boolean", "default": False},
                    },
                    "required": ["account_id"],
                },
            },
            {
                "name": "audit_log",
                "description": "Record security event.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["INFO", "WARN", "CRIT"]},
                        "message": {"type": "string"},
                    },
                    "required": ["severity", "message"],
                },
            },
        ]

        tools_variant_b = [
            {
                "name": "audit_log",
                "description": "Record security event.",
                "input_schema": {
                    "properties": {
                        "message": {"type": "string"},
                        "severity": {"enum": ["INFO", "WARN", "CRIT"], "type": "string"},
                    },
                    "required": ["severity", "message"],
                    "type": "object",
                },
            },
            {
                "name": "lookup_customer",
                "input_schema": {
                    "required": ["account_id"],
                    "properties": {
                        "include_billing": {"default": False, "type": "boolean"},
                        "account_id": {"description": "Customer UUID", "type": "string"},
                    },
                    "type": "object",
                },
                "description": "Lookup customer by account ID.",
            },
        ]

        canon_a = canonicalize_tool_schemas(tools_variant_a)
        canon_b = canonicalize_tool_schemas(tools_variant_b)

        assert canon_a == canon_b, (
            "Canonical tool schemas must be identical regardless of key or list ordering"
        )
        assert canon_a[0]["name"] == "audit_log"
        assert canon_a[1]["name"] == "lookup_customer"

    def test_nested_object_canonicalization(self):
        obj_1 = {"z": 1, "a": {"y": 2, "x": [3, {"b": 4, "a": 5}]}}
        obj_2 = {"a": {"x": [3, {"a": 5, "b": 4}], "y": 2}, "z": 1}
        assert canonicalize_object(obj_1) == canonicalize_object(obj_2)


class TestMultiTurnReActLoopParity:
    """Verifies prefix stability across multi-turn agent execution loops."""

    def test_anthropic_multi_turn_prefix_stability(self):
        mock_raw_client = MagicMock()
        mock_raw_client.messages.create.return_value = SimpleNamespace(
            id="msg_123",
            role="assistant",
            content=[SimpleNamespace(type="text", text="Processing query")],
            usage=SimpleNamespace(
                input_tokens=1500, output_tokens=120, cache_read_input_tokens=1200
            ),
        )

        # min_token_threshold=0 ensures breakpoint is injected on system prompt
        client = wrap(mock_raw_client, config=CacheAlignConfig(min_token_threshold=0))

        system_prompt = (
            "You are an enterprise financial fraud analyst. Follow strict verification steps.\n"
            "Reference Policy: All transactions >$10k require double KYC validation."
        )

        tools = [
            {
                "name": "fetch_account",
                "description": "Fetch account",
                "input_schema": {"type": "object"},
            },
            {
                "name": "flag_transaction",
                "description": "Flag transaction",
                "input_schema": {"type": "object"},
            },
        ]

        # Turn 1
        messages_turn_1 = [{"role": "user", "content": "Analyze account #9821."}]
        client.messages.create(
            model="claude-3-5-sonnet-20241022",
            system=system_prompt,
            tools=tools,
            messages=messages_turn_1,
            max_tokens=500,
        )
        call_1_kwargs = mock_raw_client.messages.create.call_args.kwargs

        # Turn 2: Assistant tool call and tool result appended
        messages_turn_2 = [
            *messages_turn_1,
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "fetch_account", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "Balance: $500,000"}
                ],
            },
        ]
        client.messages.create(
            model="claude-3-5-sonnet-20241022",
            system=system_prompt,
            tools=tools,
            messages=messages_turn_2,
            max_tokens=500,
        )
        call_2_kwargs = mock_raw_client.messages.create.call_args.kwargs

        # Verify system prompt in both turns has cache_control on the block
        sys_blocks_1 = call_1_kwargs["system"]
        sys_blocks_2 = call_2_kwargs["system"]

        assert isinstance(sys_blocks_1, list)
        assert isinstance(sys_blocks_2, list)
        assert sys_blocks_1[0]["cache_control"] == {"type": "ephemeral"}
        assert sys_blocks_2[0]["cache_control"] == {"type": "ephemeral"}
        assert sys_blocks_1[0]["text"] == sys_blocks_2[0]["text"]

    def test_openai_tool_sorting_parity(self):
        mock_raw_client = MagicMock()
        mock_raw_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Done"))]
        )

        client = wrap(mock_raw_client)

        tools_input = [
            {
                "type": "function",
                "function": {"name": "zeta_func", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "alpha_func", "parameters": {"type": "object"}},
            },
        ]

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant."}],
            tools=tools_input,
        )

        called_tools = mock_raw_client.chat.completions.create.call_args.kwargs["tools"]
        assert called_tools[0]["function"]["name"] == "alpha_func"
        assert called_tools[1]["function"]["name"] == "zeta_func"


class TestEphemeralBoundaryAndTenantIsolation:
    """Verifies isolation of ephemeral timestamps and multi-tenant security boundaries."""

    def test_ephemeral_context_isolation_across_timestamps(self):
        base_prompt = "You are a database analyzer. Verify SQL dialect."
        p1 = f"{base_prompt}\nTimestamp: 2026-10-03T10:00:00Z\nRequest-ID: req_111"
        p2 = f"{base_prompt}\nTimestamp: 2026-10-03T10:05:00Z\nRequest-ID: req_222"

        static_1, ephem_1 = extract_volatile_elements(p1)
        static_2, ephem_2 = extract_volatile_elements(p2)

        # Static invariant partition must be identical across both prompts despite changing timestamps
        assert static_1 == static_2
        assert ephem_1 != ephem_2
        assert any("Timestamp:" in item for item in ephem_1)
        assert any("Timestamp:" in item for item in ephem_2)

    def test_tenant_cryptographic_isolation_difference(self):
        system_text = "Enterprise Knowledge Graph query guidelines."
        messages = [{"role": "user", "content": "Execute query"}]

        adapter_a = AnthropicAdapter(tenant_id="tenant_alpha")
        adapter_b = AnthropicAdapter(tenant_id="tenant_beta")
        adapter_a_dup = AnthropicAdapter(tenant_id="tenant_alpha")

        res_a = adapter_a.optimize_request({"system": system_text, "messages": messages})
        res_b = adapter_b.optimize_request({"system": system_text, "messages": messages})
        res_a_dup = adapter_a_dup.optimize_request({"system": system_text, "messages": messages})

        sys_a = res_a.optimized_kwargs["system"][0]["text"]
        sys_b = res_b.optimized_kwargs["system"][0]["text"]
        sys_a_dup = res_a_dup.optimized_kwargs["system"][0]["text"]

        assert sys_a == sys_a_dup
        assert sys_a != sys_b
        assert "[CacheTenancy:" in sys_a
        assert "[CacheTenancy:" in sys_b


class TestGeminiContextCachingParity:
    """Verifies Google Gemini CachedContent integration and cost calculation formulas."""

    def test_gemini_adapter_optimizes_system_instruction(self):
        adapter = GeminiAdapter()
        kwargs = {
            "model": "gemini-2.5-flash",
            "contents": "Test query",
            "config": {
                "system_instruction": "Professor guidelines.\nTimestamp: 2026-10-03 10:00:00",
                "tools": [
                    {"name": "beta_tool", "parameters": {"type": "object"}},
                    {"name": "alpha_tool", "parameters": {"type": "object"}},
                ],
            },
        }

        res = adapter.optimize_request(kwargs)
        opt_cfg = res.optimized_kwargs["config"]

        # Tools are sorted
        assert opt_cfg["tools"][0]["name"] == "alpha_tool"
        assert opt_cfg["tools"][1]["name"] == "beta_tool"
        # Volatile timestamp is isolated from system instruction
        assert "Timestamp:" not in opt_cfg["system_instruction"]
