"""
CacheAlign FinOps Telemetry & Local Terminal Reporter.
Tracks session token velocity, cumulative dollars saved, and prints clean terminal metrics.
"""

from dataclasses import dataclass, field

from rich.console import Console

from cachealign.adapters.base import UsageStats

console = Console(stderr=True)


@dataclass
class SessionTelemetry:
    turns_count: int = 0
    total_tokens: int = 0
    total_cached_tokens: int = 0
    total_saved_usd: float = 0.0
    history: list[UsageStats] = field(default_factory=list)

    def record_turn(self, stats: UsageStats) -> None:
        self.turns_count += 1
        self.total_tokens += stats.total_tokens
        self.total_cached_tokens += stats.cached_tokens
        self.total_saved_usd += stats.estimated_cost_saved_usd
        self.history.append(stats)

    @property
    def overall_hit_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return round((self.total_cached_tokens / max(1, self.total_tokens)) * 100.0, 2)


class FinOpsReporter:
    """Manages session telemetry and terminal printing."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.session = SessionTelemetry()

    def report_turn(self, stats: UsageStats, model: str) -> None:
        self.session.record_turn(stats)
        if not self.verbose:
            return

        hit_color = (
            "green"
            if stats.cache_hit_rate >= 50.0
            else "yellow"
            if stats.cache_hit_rate > 0
            else "red"
        )

        console.print(
            f"[bold cyan][CacheAlign][/bold cyan] Turn {self.session.turns_count} ({model}) | "
            f"Hit: [{hit_color}]{stats.cache_hit_rate:.1f}%[/{hit_color}] "
            f"({stats.cached_tokens:,} cached) | "
            f"[bold green]Saved: ${stats.estimated_cost_saved_usd:.4f}[/bold green] "
            f"(Session: ${self.session.total_saved_usd:.3f})"
        )
