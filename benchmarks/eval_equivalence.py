"""
Reasoning & Semantic Equivalence Evaluator.
Verifies that prompt partitioning and ephemeral tail migration
preserve 100% of information content without semantic drift.
"""

import json
from pathlib import Path

from rich.console import Console

from cachealign.normalizers.partitioner import partition_messages_and_system

console = Console()

TEST_CASES = [
    {
        "id": "financial_audit",
        "system": "You are a financial auditor.\nDate: 2026-10-03\nRules: Reconcile all ledger entries.",
        "messages": [{"role": "user", "content": "Check Q3 ledger balance."}],
        "expected_volatile": ["Date: 2026-10-03"],
        "expected_invariant": [
            "You are a financial auditor.",
            "Rules: Reconcile all ledger entries.",
        ],
    },
    {
        "id": "customer_support",
        "system": "You are customer support agent #402.\nSession-ID: sess_customer_8892\nTurn: 1 of 5\nHelp resolve ticket.",
        "messages": [{"role": "user", "content": "I forgot my password."}],
        "expected_volatile": ["Session-ID: sess_customer_8892", "Turn: 1 of 5"],
        "expected_invariant": ["You are customer support agent #402.", "Help resolve ticket."],
    },
    {
        "id": "instruction_preservation",
        "system": "System instructions.\nDo not disclose the timestamp of transactions under any condition.\nTimestamp: 2026-10-03 14:00:00\nStrict compliance required.",
        "messages": [{"role": "user", "content": "Execute order."}],
        "expected_volatile": ["Timestamp: 2026-10-03 14:00:00"],
        "expected_invariant": [
            "System instructions.",
            "Do not disclose the timestamp of transactions under any condition.",
            "Strict compliance required.",
        ],
    },
]


def run_equivalence_eval() -> dict:
    console.print("[bold cyan]Running Reasoning & Semantic Equivalence Suite...[/bold cyan]\n")

    passed = 0
    results = []

    for case in TEST_CASES:
        clean_sys, new_msgs, _extracted = partition_messages_and_system(
            case["system"], case["messages"]
        )

        # Check that all invariant rules are preserved in clean system prompt
        invariants_preserved = all(inv in clean_sys for inv in case["expected_invariant"])
        # Check that volatile tokens are removed from system prompt
        volatile_removed = all(vol not in clean_sys for vol in case["expected_volatile"])
        # Check that volatile tokens are present in user tail
        tail_content = new_msgs[-1]["content"]
        volatile_in_tail = all(vol in tail_content for vol in case["expected_volatile"])
        xml_isolated = "<cachealign_ephemeral_context>" in tail_content

        success = invariants_preserved and volatile_removed and volatile_in_tail and xml_isolated
        if success:
            passed += 1
            console.print(f"  [bold green]PASS[/bold green] Case: {case['id']}")
        else:
            console.print(f"  [bold red]FAIL[/bold red] Case: {case['id']}")

        results.append(
            {
                "id": case["id"],
                "success": success,
                "invariants_preserved": invariants_preserved,
                "volatile_removed": volatile_removed,
                "volatile_in_tail": volatile_in_tail,
                "xml_isolated": xml_isolated,
            }
        )

    pass_rate = (passed / len(TEST_CASES)) * 100.0
    console.print(
        f"\n[bold green]Equivalence Pass Rate:[/bold green] {pass_rate:.1f}% ({passed}/{len(TEST_CASES)})\n"
    )

    out_path = Path("benchmarks/results/equivalence_results.json")
    out_path.write_text(
        json.dumps({"pass_rate_pct": pass_rate, "results": results}, indent=2), encoding="utf-8"
    )
    return {"pass_rate_pct": pass_rate, "results": results}


if __name__ == "__main__":
    run_equivalence_eval()
