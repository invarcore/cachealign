"""
CacheAlign Command Line Interface.
"""

import json
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cachealign import __version__
from cachealign.normalizers.partitioner import VOLATILE_PATTERNS
from cachealign.normalizers.schema import canonicalize_tool_schemas

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="cachealign")
def cli():
    """CacheAlign: Autonomous prompt cache optimizer & prefix alignment CLI."""
    pass


@cli.command("lint")
@click.argument("file_path", type=click.Path(exists=True))
def lint_file(file_path: str):
    """Lints a system prompt text file or JSON payload for cache-busting antipatterns."""
    console.print(f"[bold cyan]Scanning {file_path} for cache-busting antipatterns...[/bold cyan]")

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    issues = []
    for line_num, line in enumerate(content.splitlines(), start=1):
        for pattern in VOLATILE_PATTERNS:
            if pattern.search(line):
                issues.append((line_num, line.strip(), pattern.pattern))

    if not issues:
        console.print(
            "[bold green]✓ No volatile prompt patterns detected! Prefix is cache-friendly.[/bold green]"
        )
        return

    table = Table(title="Detected Volatile Anti-Patterns (Cache-Busters)")
    table.add_column("Line", style="cyan", width=8)
    table.add_column("Snippet", style="yellow")
    table.add_column("Risk", style="red")

    for line_num, snippet, _ in issues:
        table.add_row(str(line_num), snippet, "High (Invalidates KV Cache)")

    console.print(table)
    console.print(
        Panel(
            "[bold yellow]Recommendation:[/bold yellow] Wrap your client with [bold cyan]cachealign.wrap()[/bold cyan] "
            "to automatically relocate these tokens to the dynamic tail of your prompt without code changes.",
            border_style="yellow",
        )
    )


@cli.command("canonicalize")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(),
    help="Output file path (prints to stdout if omitted).",
)
def canonicalize_tools_cli(input_file: str, output_file: str | None):
    """Sorts JSON tool schemas according to RFC 8785 for byte-identical determinism."""
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        canonical = canonicalize_tool_schemas(data)
    elif isinstance(data, dict) and "tools" in data:
        data["tools"] = canonicalize_tool_schemas(data["tools"])
        canonical = data
    else:
        canonical = canonicalize_tool_schemas([data])

    formatted = json.dumps(canonical, indent=2)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted)
        console.print(f"[bold green]✓ Canonicalized tools written to {output_file}[/bold green]")
    else:
        sys.stdout.write(formatted + "\n")


@cli.command("version")
def show_version():
    """Displays CacheAlign version and status."""
    console.print(
        f"[bold cyan]CacheAlign[/bold cyan] v{__version__} — Autonomous Prompt Cache Optimizer"
    )


if __name__ == "__main__":
    cli()
