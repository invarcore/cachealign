import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

from cachealign.adapters.anthropic import AnthropicAdapter
from cachealign.adapters.gemini import GeminiAdapter
from cachealign.adapters.gemini import estimate_tokens as gemini_estimate_tokens
from cachealign.adapters.openai import OpenAIAdapter
from cachealign.cli.main import canonicalize_tools_cli
from cachealign.config import CacheAlignConfig
from cachealign.normalizers.partitioner import (
    extract_volatile_elements,
    partition_messages_and_system,
)
from cachealign.wrappers.client import (
    WrappedAsyncGeminiModelsResource,
    wrap,
)
from cachealign.wrappers.streaming import (
    WrappedAsyncStream,
    WrappedMessageStreamManager,
)


class TestPartitionerCoverageGaps:
    def test_custom_patterns_compilation_and_extraction(self):
        text = "You are an assistant.\nOrder ID: ORD-99482\nCurrent Time: 2026-10-03\nDo your job."
        patterns = [r"^Order ID:\s*ORD-\d+"]
        clean, extracted = extract_volatile_elements(text, custom_patterns=patterns)
        assert any("ORD-99482" in item for item in extracted)
        assert any("2026-10-03" in item for item in extracted)
        assert "ORD-99482" not in clean
        assert "Do your job." in clean

    def test_system_as_content_blocks_list(self):
        system_blocks = [
            {"type": "text", "text": "You are a bot.\nCurrent Time: 2026-10-03"},
            {"type": "other", "data": 123},
        ]
        messages = [{"role": "user", "content": "Hello"}]
        clean_sys, clean_msgs, extracted = partition_messages_and_system(system_blocks, messages)
        assert len(clean_sys) == 2
        assert "Current Time:" not in clean_sys[0]["text"]
        assert len(extracted) == 1
        assert "2026-10-03" in extracted[0]
        assert "<cachealign_ephemeral_context>" in clean_msgs[-1]["content"]

    def test_messages_ending_with_tool_preceding_list_content(self):
        system = "Current Time: 2026-10-03"
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Run analysis"}]},
            {"role": "tool", "content": "Tool result 1"},
        ]
        _clean_sys, clean_msgs, extracted = partition_messages_and_system(system, messages)
        assert len(extracted) == 1
        user_content = clean_msgs[0]["content"]
        assert isinstance(user_content, list)
        assert any(
            "<cachealign_ephemeral_context>" in item.get("text", "")
            for item in user_content
            if isinstance(item, dict)
        )

    def test_messages_ending_with_tool_without_preceding_user(self):
        system = "Current Time: 2026-10-03"
        messages = [
            {"role": "tool", "content": "Initial tool event"},
        ]
        _clean_sys, clean_msgs, extracted = partition_messages_and_system(system, messages)
        assert len(extracted) == 1
        assert clean_msgs[-1]["role"] == "user"
        assert "<cachealign_ephemeral_context>" in clean_msgs[-1]["content"]

    def test_messages_structured_content_with_tool_results(self):
        system = "Current Time: 2026-10-03"
        messages = [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "OK"}],
            }
        ]
        _clean_sys, clean_msgs, _extracted = partition_messages_and_system(system, messages)
        last_content = clean_msgs[-1]["content"]
        assert isinstance(last_content, list)
        assert any(
            item.get("type") == "text" and "<cachealign_ephemeral_context>" in item.get("text", "")
            for item in last_content
        )

    def test_messages_structured_content_with_existing_text_blocks(self):
        system = "Current Time: 2026-10-03"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part. "},
                    {"type": "text", "text": "Second part."},
                ],
            }
        ]
        _clean_sys, clean_msgs, _extracted = partition_messages_and_system(system, messages)
        last_content = clean_msgs[-1]["content"]
        assert isinstance(last_content, list)
        assert "<cachealign_ephemeral_context>" in last_content[-1]["text"]


class TestGeminiAdapterCoverageGaps:
    def test_approx_token_count_variants(self):
        assert gemini_estimate_tokens("abcd") == 1
        assert gemini_estimate_tokens({"a": "b"}) >= 1
        assert gemini_estimate_tokens(["a", "b"]) >= 1
        assert gemini_estimate_tokens(12345) == 1

    def test_gemini_object_config_and_tenant_salt(self):
        adapter = GeminiAdapter(tenant_id="tenant-acme-prod")
        raw_config = SimpleNamespace(
            system_instruction="Analyze database.\nCurrent Time: 2026-10-03", cached_content=None
        )
        kwargs = {"contents": "Execute query.", "config": raw_config}
        res = adapter.optimize_request(kwargs)
        opt_cfg = res.optimized_kwargs["config"]
        assert hasattr(opt_cfg, "system_instruction")
        assert "[CacheTenancy:" in opt_cfg.system_instruction

    def test_gemini_contents_as_list_and_parts_dict(self):
        adapter = GeminiAdapter()
        kwargs = {
            "contents": [{"role": "user", "parts": ["Initial query", {"text": "Details here"}]}],
            "config": {"system_instruction": "Current Time: 2026-10-03"},
        }
        res = adapter.optimize_request(kwargs)
        opt_contents = res.optimized_kwargs["contents"]
        assert len(opt_contents) == 1
        parts = opt_contents[0]["parts"]
        assert any(
            "<cachealign_ephemeral_context>" in (p if isinstance(p, str) else p.get("text", ""))
            for p in parts
        )

    def test_gemini_contents_as_strings_list(self):
        adapter = GeminiAdapter()
        kwargs = {
            "contents": ["Query 1", "Query 2"],
            "config": {"system_instruction": "Current Time: 2026-10-03"},
        }
        res = adapter.optimize_request(kwargs)
        assert "<cachealign_ephemeral_context>" in res.optimized_kwargs["contents"][-1]


class TestAnthropicAdapterCoverageGaps:
    def test_anthropic_system_as_blocks_with_tenant_salt_and_cache_control(self):
        adapter = AnthropicAdapter(tenant_id="enterprise-tenant-1")
        system_blocks = [{"type": "text", "text": "Base prompt." * 300}]
        messages = [{"role": "user", "content": "Help me."}]
        res = adapter.optimize_request({"system": system_blocks, "messages": messages})
        opt_sys = res.optimized_kwargs["system"]
        assert isinstance(opt_sys, list)
        assert "[CacheTenancy:" in opt_sys[0]["text"]
        assert opt_sys[-1].get("cache_control") == {"type": "ephemeral"}

    def test_anthropic_history_turn_string_breakpoint(self):
        adapter = AnthropicAdapter()
        messages = [
            {"role": "user", "content": "First long query." * 200},
            {"role": "assistant", "content": "Assistant long response." * 200},
            {"role": "user", "content": "Follow-up question."},
        ]
        res = adapter.optimize_request({"messages": messages})
        opt_msgs = res.optimized_kwargs["messages"]
        turn_1_content = opt_msgs[1]["content"]
        assert isinstance(turn_1_content, list)
        assert turn_1_content[0].get("cache_control") == {"type": "ephemeral"}

    def test_anthropic_parse_usage_none(self):
        adapter = AnthropicAdapter()
        stats = adapter.parse_usage(None, "claude-3-5-sonnet")
        assert stats.input_tokens == 0
        assert stats.cached_tokens == 0


class TestOpenAIAdapterCoverageGaps:
    def test_openai_legacy_functions_canonicalization(self):
        adapter = OpenAIAdapter()
        funcs = [
            {
                "name": "b_func",
                "parameters": {"properties": {"z": {"type": "string"}, "a": {"type": "number"}}},
            },
            {"name": "a_func", "parameters": {"properties": {"k": {"type": "boolean"}}}},
        ]
        res = adapter.optimize_request(
            {"messages": [{"role": "user", "content": "Hi"}], "functions": funcs}
        )
        canon_funcs = res.optimized_kwargs["functions"]
        assert canon_funcs[0]["name"] == "a_func"
        assert canon_funcs[1]["name"] == "b_func"

    def test_openai_disable_prefix_partitioning(self):
        config = CacheAlignConfig(enable_tail_migration=False, tenant_id="tenant-xyz")
        adapter = OpenAIAdapter(config=config)
        res = adapter.optimize_request(
            {
                "messages": [
                    {"role": "system", "content": "Do not partition.\nCurrent Time: 2026-10-03"},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )
        msgs = res.optimized_kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert "[CacheTenancy:" in msgs[0]["content"]
        assert "Current Time: 2026-10-03" in msgs[0]["content"]

    def test_openai_tenant_salt_without_system_message(self):
        config = CacheAlignConfig(enable_tail_migration=False, tenant_id="tenant-xyz")
        adapter = OpenAIAdapter(config=config)
        res = adapter.optimize_request({"messages": [{"role": "user", "content": "Hello"}]})
        msgs = res.optimized_kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert "[CacheTenancy:" in msgs[0]["content"]


class TestStreamingAndWrappersCoverageGaps:
    @pytest.mark.anyio
    async def test_wrapped_async_stream_getattr(self):
        mock_raw = AsyncMock()
        mock_raw.custom_attr = "stream_val"
        wrapper = WrappedAsyncStream(mock_raw, MagicMock(), MagicMock(), "test-model", "openai")
        assert wrapper.custom_attr == "stream_val"

    @pytest.mark.anyio
    async def test_wrapped_message_stream_manager_async_context(self):
        mock_stream = AsyncMock()
        mock_stream.get_final_message = AsyncMock(
            return_value=SimpleNamespace(usage=SimpleNamespace(input_tokens=100, output_tokens=20))
        )
        mock_stream.custom_method = lambda: "ok"

        mock_mgr = MagicMock()
        mock_mgr.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_mgr.__aexit__ = AsyncMock(return_value=None)
        mock_mgr.extra_attr = 42

        adapter = AnthropicAdapter()
        reporter = MagicMock()
        wrapper = WrappedMessageStreamManager(mock_mgr, adapter, reporter, "claude-3-5-sonnet")

        assert wrapper.extra_attr == 42
        async with wrapper as stream:
            assert stream.custom_method() == "ok"
        reporter.report_turn.assert_called_once()

    @pytest.mark.anyio
    async def test_async_gemini_models_wrapper_stream_and_fail_open(self):
        mock_orig = AsyncMock()
        mock_orig.generate_content_stream = AsyncMock(return_value=AsyncMock())
        mock_orig.some_gemini_method = lambda: "val"

        reporter = MagicMock()
        config = CacheAlignConfig(fail_open=True)
        wrapper = WrappedAsyncGeminiModelsResource(mock_orig, reporter, config)

        assert wrapper.some_gemini_method() == "val"

        s = await wrapper.generate_content_stream(model="gemini-2.5-flash", contents="Hi")
        assert isinstance(s, WrappedAsyncStream)

        bad_adapter = MagicMock()
        bad_adapter.optimize_request.side_effect = RuntimeError("Partitioner broke!")
        wrapper._adapter = bad_adapter
        res_kwargs = wrapper._safe_optimize({"contents": "Fallback content"})
        assert res_kwargs == {"contents": "Fallback content"}

    def test_client_wrapper_properties_and_getattr(self):
        mock_client = MagicMock()
        mock_client.special_client_id = "client-007"
        wrapped = wrap(mock_client, config=CacheAlignConfig(fail_open=True))
        assert wrapped.special_client_id == "client-007"
        assert wrapped.config.fail_open is True


class TestCLICoverageGaps:
    def test_cli_canonicalize_tools_dict_and_output_file(self, tmp_path):
        in_file = tmp_path / "tools.json"
        out_file = tmp_path / "canonical.json"

        data = {"tools": [{"name": "fetch", "parameters": {"b": 2, "a": 1}}]}
        in_file.write_text(json.dumps(data), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(canonicalize_tools_cli, [str(in_file), "--output", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()
        out_data = json.loads(out_file.read_text(encoding="utf-8"))
        assert "tools" in out_data
        assert out_data["tools"][0]["name"] == "fetch"
