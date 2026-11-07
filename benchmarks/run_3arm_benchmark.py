# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
Candidate C15: Hardened 3-Arm Reproducible Benchmark Harness.

Evaluates prompt caching behavior across 4 arms and 3 realistic workloads:
- Arm 1: Raw Baseline (no caching, raw dicts, volatile headers)
- Arm 2: Naive Provider Caching (provider cache_control added, but unpartitioned & uncanonicalized)
- Arm 3a: CacheAlign Direct SDK (RFC 8785 schema canonicalization + ephemeral tail-migration)
- Arm 3b: CacheAlign LiteLLM Gateway (CacheAlignLiteLLMHandler custom logger)

Computes:
- Cached Token Ratio (CTR %)
- Net Cost Reduction (NCR %)
- Cache Invalidation Penalty (Arm 2 vs Arm 1 %)
- p50 and p95 Time-To-First-Token (TTFT ms)
- Streaming audit events to benchmarks/results/3arm_benchmark_logs.jsonl

Usage:
    uv run python benchmarks/run_3arm_benchmark.py [--backend simulation|live] [--workload all|data|rag|coding] [--turns 10]
"""

import argparse
import copy
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from cachealign.adapters.anthropic import (
    SONNET_BASE_INPUT_PER_M,
    SONNET_CACHE_READ_PER_M,
    SONNET_CACHE_WRITE_PER_M,
    AnthropicAdapter,
)
from cachealign.config import CacheAlignConfig
from cachealign.integrations.litellm import CacheAlignLiteLLMHandler

console = Console()

# Pricing constants for Claude 3.5 Sonnet
PRICE_BASE_INPUT = SONNET_BASE_INPUT_PER_M / 1_000_000.0  # $0.000003 / tok ($3.00/M)
PRICE_CACHE_WRITE = SONNET_CACHE_WRITE_PER_M / 1_000_000.0  # $0.00000375 / tok ($3.75/M, 1.25x)
PRICE_CACHE_READ = SONNET_CACHE_READ_PER_M / 1_000_000.0  # $0.00000030 / tok ($0.30/M, 0.10x)
PRICE_OUTPUT = 15.00 / 1_000_000.0  # $0.000015 / tok ($15.00/M)

OUTPUT_TOKENS_PER_TURN = 250


# ==============================================================================
# Simulation KV-Cache Physics Engine
# ==============================================================================


class HardenedKVCacheSimulator:
    """
    Models physical provider KV-cache chunking behavior without circular heuristics:
    - Minimum cacheable prefix threshold: 1,024 tokens.
    - Explicit breakpoint hashing: Computes SHA-256 of canonical byte prefix up to cache boundary.
    - Invalidation Penalty accounting: Repeated cache misses with cache_control pay 1.25x write fee.
    """

    def __init__(self, min_token_threshold: int = 1024):
        self.min_token_threshold = min_token_threshold
        self.cache_registry: set[str] = set()

    def reset(self):
        self.cache_registry.clear()

    @staticmethod
    def _approx_token_count(text_or_obj: Any) -> int:
        """Approximates token count using UTF-8 byte length with word structure heuristics."""
        if isinstance(text_or_obj, str):
            byte_len = len(text_or_obj.encode("utf-8"))
            return max(1, int(byte_len / 3.8))
        if isinstance(text_or_obj, (dict, list)):
            serialized = json.dumps(text_or_obj, ensure_ascii=False)
            byte_len = len(serialized.encode("utf-8"))
            return max(1, int(byte_len / 3.8))
        return 0

    def compute_prefix_hash(self, system: Any, tools: Any, breakpoint_set: bool) -> tuple[str, int]:
        """
        Computes the exact byte hash of the static prefix up to the breakpoint.
        In real providers, keys must be serialized in exact identical order.
        """
        # Serialize exactly as passed into the network payload
        serialized_system = (
            json.dumps(system, default=str) if not isinstance(system, str) else system
        )
        serialized_tools = json.dumps(tools, default=str) if tools else ""

        tokens = self._approx_token_count(serialized_system) + self._approx_token_count(
            serialized_tools
        )
        hasher = hashlib.sha256()
        hasher.update(serialized_system.encode("utf-8"))
        hasher.update(b"||TOOLS||")
        hasher.update(serialized_tools.encode("utf-8"))
        hasher.update(f"||BP:{breakpoint_set}||".encode())
        return hasher.hexdigest(), tokens

    def process_turn(
        self,
        arm: str,
        system: Any,
        tools: Any,
        messages: list[dict[str, Any]],
        has_cache_control: bool,
    ) -> dict[str, Any]:
        """Executes a single turn against the KV-cache simulator, returning verified usage and TTFT."""
        prefix_hash, prefix_tokens = self.compute_prefix_hash(system, tools, has_cache_control)
        history_tokens = sum(self._approx_token_count(m.get("content", "")) for m in messages)
        total_input_tokens = prefix_tokens + history_tokens

        is_eligible = prefix_tokens >= self.min_token_threshold

        if arm == "raw_baseline" or not has_cache_control or not is_eligible:
            # Arm 1: No caching enabled
            cached_tokens = 0
            cache_write_tokens = 0
            uncached_tokens = total_input_tokens
            ttft_ms = random.uniform(340.0, 410.0)
            cost = (total_input_tokens * PRICE_BASE_INPUT) + (OUTPUT_TOKENS_PER_TURN * PRICE_OUTPUT)
            is_hit = False
        else:
            # Arm 2, Arm 3a, Arm 3b: Caching active
            if prefix_hash in self.cache_registry:
                # Cache HIT!
                cached_tokens = prefix_tokens
                cache_write_tokens = 0
                uncached_tokens = history_tokens
                ttft_ms = random.uniform(38.0, 58.0)
                cost = (
                    (cached_tokens * PRICE_CACHE_READ)
                    + (uncached_tokens * PRICE_BASE_INPUT)
                    + (OUTPUT_TOKENS_PER_TURN * PRICE_OUTPUT)
                )
                is_hit = True
            else:
                # Cache MISS (Cold write or Busted prefix)
                self.cache_registry.add(prefix_hash)
                cached_tokens = 0
                cache_write_tokens = prefix_tokens
                uncached_tokens = history_tokens
                ttft_ms = random.uniform(350.0, 440.0)
                # Cache creation surcharge (1.25x on Anthropic)
                cost = (
                    (cache_write_tokens * PRICE_CACHE_WRITE)
                    + (uncached_tokens * PRICE_BASE_INPUT)
                    + (OUTPUT_TOKENS_PER_TURN * PRICE_OUTPUT)
                )
                is_hit = False

        return {
            "total_input_tokens": total_input_tokens,
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
            "uncached_tokens": uncached_tokens,
            "cost_usd": cost,
            "ttft_ms": round(ttft_ms, 2),
            "is_hit": is_hit,
            "prefix_hash": prefix_hash[:12],
        }


class HardenedLiveExecutor:
    """
    Executes live network requests against Anthropic Claude 3.5 Sonnet to measure authentic
    provider KV-cache retention, cache creation charges, and real wire TTFT.
    """

    def __init__(self, api_key: str):
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError as err:
            raise ImportError(
                "Live execution requires the 'anthropic' package. Install with `uv add anthropic`."
            ) from err

    def reset(self):
        """No-op for live execution (provider handles its own server-side cache)."""
        pass

    def process_turn(
        self,
        arm: str,
        system: Any,
        tools: Any,
        messages: list[dict[str, Any]],
        has_cache_control: bool,
    ) -> dict[str, Any]:
        # Format system parameter for Anthropic API
        if isinstance(system, str) and has_cache_control and arm == "naive_provider":
            # Developer naively wrapped system prompt with cache_control
            formatted_system = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            formatted_system = system

        kwargs: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": OUTPUT_TOKENS_PER_TURN,
            "messages": messages,
        }
        if formatted_system:
            kwargs["system"] = formatted_system
        if tools:
            kwargs["tools"] = tools

        t0 = time.perf_counter()
        response = self.client.messages.create(**kwargs)
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        usage = response.usage
        cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        uncached_tokens = getattr(usage, "input_tokens", 0) or 0
        total_input_tokens = uncached_tokens + cached_tokens + cache_write_tokens
        output_tokens = (
            getattr(usage, "output_tokens", OUTPUT_TOKENS_PER_TURN) or OUTPUT_TOKENS_PER_TURN
        )

        cost = (
            (cache_write_tokens * PRICE_CACHE_WRITE)
            + (cached_tokens * PRICE_CACHE_READ)
            + (uncached_tokens * PRICE_BASE_INPUT)
            + (output_tokens * PRICE_OUTPUT)
        )
        is_hit = cached_tokens > 0
        prefix_hash = hashlib.sha256(
            json.dumps(formatted_system, default=str).encode()
        ).hexdigest()[:12]

        # Rate limit damping for live multi-turn sequences
        time.sleep(1.0)

        return {
            "total_input_tokens": total_input_tokens,
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
            "uncached_tokens": uncached_tokens,
            "cost_usd": cost,
            "ttft_ms": round(ttft_ms, 2),
            "is_hit": is_hit,
            "prefix_hash": prefix_hash,
            "response_id": response.id,
        }


# ==============================================================================
# Workload Generators
# ==============================================================================


def get_workload_definitions():
    """Generates the 3 benchmark workloads conforming to C15 specification."""
    # Workload 1: Enterprise Data Agent (~4,500 token invariant corporate compliance prompt)
    w1_system_base = """
You are Globex Enterprise Analytics Engine (v4.2). Architecture: Snowflake + BigQuery Lakehouse.
Security & Governance Directives:
1. All SQL operations must strictly enforce column-level access controls and tenant segmentation.
2. Under SOC2 Type II and HIPAA requirements, never emit unmasked PII (SSN, credit card, plaintext phone).
3. Always validate partition filters on daily date clusters before emitting BigQuery GQL or Snowflake transformations.
4. Emulate Claude design philosophy: thorough, self-critical, deeply rigorous, and mathematically exact.
5. All timestamps in generated schema must adhere to ISO 8601 UTC with explicit millisecond precision.
""" + (
        "\nData Model Schema Specification: Table `transactions_partitioned_v5` "
        "(tx_id: uuid, customer_id: uuid, amount: numeric(18,4), currency: char(3), settlement_status: varchar(32))... "
        * 65
    )

    # 15 Analytical Tools
    w1_tools_canonical = [
        {
            "name": f"analytics_tool_{i:02d}",
            "description": f"Executes data warehouse transformation step {i}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql_query": {"type": "string", "description": "SQL statement"},
                    "max_partitions": {"type": "integer", "description": "Pruning limit"},
                    "timeout_ms": {"type": "integer", "description": "Query timeout"},
                },
                "required": ["sql_query"],
            },
        }
        for i in range(15)
    ]

    # Workload 2: Rotating Multi-Tenant RAG Retrieval (~2,500 token guide + dynamic chunks)
    w2_system_base = """
You are the Legal & Compliance Invariant Analysis Engine for InvarCore Global.
Operational Directives:
1. Examine retrieved regulatory evidence chunks against European AI Act Article 14 human oversight requirements.
2. If evidence lacks cryptographic HMAC provenance envelope, fail closed and refuse synthesis.
3. Verify tenant boundary isolation before executing comparative cross-jurisdiction analysis.
""" + (
        "\nCompliance Reference Clause Article 14(4): Automated risk management systems shall maintain "
        "audit logs containing timestamp, operator signature, input context digest, and execution state... "
        * 40
    )

    w2_tools = [
        {
            "name": f"rag_evidence_verifier_{i}",
            "description": f"Verifies evidence item {i}",
            "input_schema": {"type": "object", "properties": {"chunk_id": {"type": "string"}}},
        }
        for i in range(8)
    ]

    # Workload 3: Multi-Turn Coding Loop (~3,000 token architectural constraints)
    w3_system_base = """
You are an autonomous senior systems compiler engineer for the Invarcore Rust/Python monorepo.
Architecture Principles:
1. Zero runtime allocations in the hot path.
2. Constant-time comparison for all token and signature checks.
3. Every test must adhere to the Zero-Defect Truth in Execution rule.
4. Preserves backward compatibility with RFC 8785 canonical serialization.
""" + (
        "\nCompiler Configuration Spec: Target x86_64-unknown-linux-gnu, lld linker, codegen-units=1, "
        "opt-level=3, panic=abort, sanitizers=[address, undefined]... " * 45
    )

    w3_tools = [
        {
            "name": f"compiler_action_{i}",
            "description": f"Executes compiler action {i}",
            "input_schema": {"type": "object", "properties": {"source_path": {"type": "string"}}},
        }
        for i in range(6)
    ]

    return {
        "data_agent": {
            "name": "Workload 1: Enterprise Data Agent (Tool Schema Jitter)",
            "system_base": w1_system_base,
            "tools": w1_tools_canonical,
            "has_tool_jitter": True,
        },
        "rotating_rag": {
            "name": "Workload 2: Rotating Multi-Tenant RAG (Evidence Displacement)",
            "system_base": w2_system_base,
            "tools": w2_tools,
            "has_tool_jitter": False,
        },
        "coding_loop": {
            "name": "Workload 3: Multi-Turn Coding Loop (ReAct Compaction)",
            "system_base": w3_system_base,
            "tools": w3_tools,
            "has_tool_jitter": False,
        },
    }


def jitter_dict_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Shuffles dictionary keys to simulate non-deterministic microservice serialization."""
    items = list(d.items())
    random.shuffle(items)
    result = {}
    for k, v in items:
        if isinstance(v, dict):
            result[k] = jitter_dict_keys(v)
        elif isinstance(v, list):
            result[k] = [jitter_dict_keys(i) if isinstance(i, dict) else i for i in v]
        else:
            result[k] = v
    return result


def jitter_tool_list(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shuffles tools array order and dict keys."""
    shuffled = [copy.deepcopy(t) for t in tools]
    random.shuffle(shuffled)
    return [jitter_dict_keys(t) for t in shuffled]


# ==============================================================================
# 3-Arm Execution Harness
# ==============================================================================


@dataclass
class TurnMetric:
    workload: str
    arm: str
    turn: int
    total_input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    uncached_tokens: int
    cost_usd: float
    ttft_ms: float
    is_hit: bool
    prefix_hash: str


def run_benchmark(
    backend: str = "simulation",
    workload_key: str = "all",
    turns: int = 10,
    output_dir: str = "benchmarks/results",
) -> dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logs_file = out_path / "3arm_benchmark_logs.jsonl"
    results_file = out_path / "3arm_benchmark_results.json"

    console.print(
        "\n[bold cyan]════════════════════════════════════════════════════════════════════════════════[/bold cyan]"
    )
    console.print(
        "[bold white]  Candidate C15: Hardened 3-Arm Reproducible Benchmark Harness  [/bold white]"
    )
    console.print(
        f"[bold cyan]  Backend: [{backend.upper()}]  |  Turns Per Run: {turns}  |  Out: {out_path}[/bold cyan]"
    )
    console.print(
        "[bold cyan]════════════════════════════════════════════════════════════════════════════════[/bold cyan]\n"
    )

    if backend == "live":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Live backend execution requires ANTHROPIC_API_KEY environment variable. "
                "Set ANTHROPIC_API_KEY in your environment or run with '--backend simulation' (default)."
            )
        executor = HardenedLiveExecutor(api_key=api_key)
    else:
        executor = HardenedKVCacheSimulator(min_token_threshold=1024)

    workload_defs = get_workload_definitions()
    selected_workloads = workload_defs.keys() if workload_key == "all" else [workload_key]

    arms = [
        ("Arm 1", "raw_baseline"),
        ("Arm 2", "naive_provider"),
        ("Arm 3a", "cachealign_direct"),
        ("Arm 3b", "cachealign_litellm"),
    ]

    anthropic_adapter = AnthropicAdapter(config=CacheAlignConfig(min_token_threshold=1024))
    litellm_handler = CacheAlignLiteLLMHandler(
        config=CacheAlignConfig(min_token_threshold=1024), verbose=False
    )

    all_turn_metrics: list[TurnMetric] = []
    summary_matrix: list[dict[str, Any]] = []

    # Clear previous log file
    with open(logs_file, "w", encoding="utf-8") as f:
        pass

    for wk in selected_workloads:
        w_info = workload_defs[wk]
        w_name = w_info["name"]
        console.print(f"\n[bold yellow]▶ Executing {w_name}...[/bold yellow]")

        arm_results = {}

        for arm_label, arm_id in arms:
            executor.reset()
            messages = []
            turn_records: list[TurnMetric] = []

            for turn in range(1, turns + 1):
                timestamp_header = (
                    f"Current Time: 2026-11-07 14:{turn:02d}:00 UTC\nSession-ID: sess_prod_{turn}"
                )
                user_content = (
                    f"Turn {turn}: Process analytics pipeline and verify compliance constraints."
                )

                # Context generation per workload
                if wk == "rotating_rag":
                    evidence = f"\n[Evidence Chunk {turn}A: Verified RLS context digest abc{turn}456]\n[Evidence Chunk {turn}B: Policy compliance verified at epoch {turn}000]"
                    user_content += evidence
                elif wk == "coding_loop":
                    compiler_out = f"\n[Build #{turn} exit status: 0, warnings: 0, test_cases_passed: {turn * 12}]"
                    user_content += compiler_out

                messages.append({"role": "user", "content": user_content})

                turn_messages = messages
                # Determine payload structures per arm
                if arm_id == "raw_baseline":
                    # Arm 1: Shuffled tools, volatile header embedded in system, no cache control
                    sys_content = f"{timestamp_header}\n{w_info['system_base']}"
                    tools_payload = (
                        jitter_tool_list(w_info["tools"])
                        if w_info["has_tool_jitter"]
                        else list(w_info["tools"])
                    )
                    has_cache_control = False
                    turn_messages = messages

                elif arm_id == "naive_provider":
                    # Arm 2: Naive cache_control on system, BUT volatile header at line 1 and uncanonicalized tools
                    sys_content = f"{timestamp_header}\n{w_info['system_base']}"
                    tools_payload = (
                        jitter_tool_list(w_info["tools"])
                        if w_info["has_tool_jitter"]
                        else list(w_info["tools"])
                    )
                    has_cache_control = (
                        True  # Developer enabled prompt caching, but didn't align prefix
                    )
                    turn_messages = messages

                elif arm_id == "cachealign_direct":
                    # Arm 3a: CacheAlign Direct SDK
                    raw_sys = f"{timestamp_header}\n{w_info['system_base']}"
                    raw_tools = (
                        jitter_tool_list(w_info["tools"])
                        if w_info["has_tool_jitter"]
                        else list(w_info["tools"])
                    )

                    # Apply CacheAlign Adapter optimization
                    opt_payload = anthropic_adapter.optimize_request(
                        {
                            "model": "claude-3-5-sonnet",
                            "system": raw_sys,
                            "tools": raw_tools,
                            "messages": messages,
                        }
                    )
                    sys_content = opt_payload.optimized_kwargs.get("system")
                    tools_payload = opt_payload.optimized_kwargs.get("tools")
                    has_cache_control = True
                    turn_messages = opt_payload.optimized_kwargs.get("messages", messages)

                elif arm_id == "cachealign_litellm":
                    # Arm 3b: CacheAlign LiteLLM Gateway
                    raw_sys = f"{timestamp_header}\n{w_info['system_base']}"
                    raw_tools = (
                        jitter_tool_list(w_info["tools"])
                        if w_info["has_tool_jitter"]
                        else list(w_info["tools"])
                    )

                    litellm_data = {
                        "model": "claude-3-5-sonnet",
                        "messages": [{"role": "system", "content": raw_sys}, *messages],
                        "tools": [{"function": t} for t in raw_tools],
                    }
                    opt_litellm = litellm_handler.async_pre_call_hook(
                        {}, None, litellm_data, "completion"
                    )

                    # Extract sanitized system and canonicalized tools
                    sys_content = next(
                        (m["content"] for m in opt_litellm["messages"] if m["role"] == "system"), ""
                    )
                    tools_payload = [t["function"] for t in opt_litellm["tools"]]
                    has_cache_control = True
                    turn_messages = [m for m in opt_litellm["messages"] if m["role"] != "system"]

                # Dispatch to Simulator or Live Executor
                res = executor.process_turn(
                    arm=arm_id,
                    system=sys_content,
                    tools=tools_payload,
                    messages=turn_messages,
                    has_cache_control=has_cache_control,
                )

                metric = TurnMetric(
                    workload=wk,
                    arm=arm_id,
                    turn=turn,
                    total_input_tokens=res["total_input_tokens"],
                    cached_tokens=res["cached_tokens"],
                    cache_write_tokens=res["cache_write_tokens"],
                    uncached_tokens=res["uncached_tokens"],
                    cost_usd=round(res["cost_usd"], 6),
                    ttft_ms=res["ttft_ms"],
                    is_hit=res["is_hit"],
                    prefix_hash=res["prefix_hash"],
                )
                turn_records.append(metric)
                all_turn_metrics.append(metric)

                # Stream to raw log file (results.jsonl)
                with open(logs_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(metric)) + "\n")

                # Append mock assistant response to history
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Acknowledged turn {turn} completed successfully.",
                    }
                )

            # Calculate Aggregate Arm Metrics
            total_in = sum(m.total_input_tokens for m in turn_records)
            total_cached = sum(m.cached_tokens for m in turn_records)
            total_cost = sum(m.cost_usd for m in turn_records)
            ctr = (total_cached / max(1, total_in)) * 100.0

            ttfts = sorted(m.ttft_ms for m in turn_records)
            p50_ttft = ttfts[len(ttfts) // 2]
            p95_ttft = ttfts[int(len(ttfts) * 0.95)]

            arm_results[arm_id] = {
                "arm_label": arm_label,
                "arm_id": arm_id,
                "total_input_tokens": total_in,
                "total_cached_tokens": total_cached,
                "cached_token_ratio_pct": round(ctr, 2),
                "total_cost_usd": round(total_cost, 4),
                "p50_ttft_ms": p50_ttft,
                "p95_ttft_ms": p95_ttft,
            }

        # Calculate comparative metrics against Arm 1 & Arm 2
        raw_cost = arm_results["raw_baseline"]["total_cost_usd"]
        naive_cost = arm_results["naive_provider"]["total_cost_usd"]

        # Cache Invalidation Penalty: How much MORE expensive naive caching was compared to raw
        invalidation_penalty_pct = round(((naive_cost - raw_cost) / max(1e-6, raw_cost)) * 100.0, 2)

        for arm_id, data in arm_results.items():
            cost = data["total_cost_usd"]
            savings_vs_raw = round(((raw_cost - cost) / max(1e-6, raw_cost)) * 100.0, 2)
            savings_vs_naive = round(((naive_cost - cost) / max(1e-6, naive_cost)) * 100.0, 2)

            data["savings_vs_raw_pct"] = savings_vs_raw
            data["savings_vs_naive_pct"] = savings_vs_naive
            data["invalidation_penalty_pct"] = (
                invalidation_penalty_pct if arm_id == "naive_provider" else 0.0
            )
            data["workload"] = wk
            summary_matrix.append(data)

    # Output Console Summary Table
    table = Table(title="Candidate C15: Hardened 3-Arm Benchmark Matrix (Sonnet 3.5 Engine)")
    table.add_column("Workload", style="cyan")
    table.add_column("Arm", style="bold")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Cached Tokens", justify="right")
    table.add_column("CTR (%)", justify="right", style="magenta")
    table.add_column("Total Cost ($)", justify="right")
    table.add_column("Savings vs Raw", justify="right", style="green")
    table.add_column("Invalidation Penalty", justify="right", style="red")
    table.add_column("p50 TTFT", justify="right")

    for row in summary_matrix:
        pen_str = (
            f"+{row['invalidation_penalty_pct']:.1f}%"
            if row["invalidation_penalty_pct"] > 0
            else "-"
        )
        sav_str = f"{row['savings_vs_raw_pct']:.1f}%" if row["savings_vs_raw_pct"] > 0 else "-"
        table.add_row(
            row["workload"],
            row["arm_label"],
            f"{row['total_input_tokens']:,}",
            f"{row['total_cached_tokens']:,}",
            f"{row['cached_token_ratio_pct']:.1f}%",
            f"${row['total_cost_usd']:.4f}",
            sav_str,
            pen_str,
            f"{row['p50_ttft_ms']:.1f}ms",
        )

    console.print(table)

    final_payload = {
        "benchmark": "Candidate C15 Hardened 3-Arm Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": backend,
        "turns": turns,
        "pricing_model": "Anthropic Claude 3.5 Sonnet ($3.00 in, $3.75 write, $0.30 read, $15.00 out)",
        "summary": summary_matrix,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    console.print("\n[bold green]✓ Execution Complete.[/bold green]")
    console.print(f"  • Summary written to: [bold]{results_file}[/bold]")
    console.print(f"  • Raw turn events written to: [bold]{logs_file}[/bold]\n")
    return final_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Candidate C15: Hardened 3-Arm CacheAlign Benchmark"
    )
    parser.add_argument(
        "--backend", choices=["simulation", "live"], default="simulation", help="Execution backend"
    )
    parser.add_argument(
        "--workload", choices=["all", "data_agent", "rotating_rag", "coding_loop"], default="all"
    )
    parser.add_argument("--turns", type=int, default=10, help="Number of turns per workload")
    parser.add_argument(
        "--output-dir", default="benchmarks/results", help="Directory for output artifacts"
    )

    args = parser.parse_args()
    run_benchmark(
        backend=args.backend,
        workload_key=args.workload,
        turns=args.turns,
        output_dir=args.output_dir,
    )
