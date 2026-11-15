# Copyright 2026 Invarcore Organization
# SPDX-License-Identifier: MIT

"""
CacheAlign Command Line Interface.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

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


@cli.command("probe")
@click.option(
    "-p",
    "--provider",
    type=click.Choice(["anthropic", "openai", "gemini"], case_sensitive=False),
    default="anthropic",
    help="Target LLM provider to probe.",
)
@click.option(
    "-r",
    "--route",
    type=click.Choice(["direct", "bedrock", "vertex", "openrouter"], case_sensitive=False),
    default="direct",
    help="Routing architecture to evaluate.",
)
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(),
    help="Output JSON report file path.",
)
@click.option(
    "--live",
    is_flag=True,
    default=False,
    help="Execute live network roundtrips against provider API (requires provider API key).",
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["full", "daily-smoke"], case_sensitive=False),
    default="full",
    help="Probe mode: 'full' (all 4 normative vectors) or 'daily-smoke' (P1 threshold + P4 accounting).",
)
def probe_command(
    provider: str,
    route: str,
    output_file: str | None,
    live: bool = False,
    mode: str = "full",
):
    """Probes provider prompt cache semantics across normative regression vectors."""
    from cachealign.cli.probe import execute_probe_cli

    exit_code = execute_probe_cli(
        provider=provider, route=route, output_file=output_file, live=live, mode=mode
    )
    if exit_code != 0:
        sys.exit(exit_code)


@cli.command("version")
def show_version():
    """Displays CacheAlign version and status."""
    console.print(
        f"[bold cyan]CacheAlign[/bold cyan] v{__version__} — Autonomous Prompt Cache Optimizer"
    )


def _default_ca_cert_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else (Path.home() / ".config")
    return base / "cachealign" / "ca" / "ca.crt"


def _build_exec_env(
    base_env: dict[str, str],
    proxy_url: str,
    no_proxy: str,
    ca_cert_path: Path | None,
) -> dict[str, str]:
    env = dict(base_env)
    env["HTTP_PROXY"] = proxy_url
    env["http_proxy"] = proxy_url
    env["HTTPS_PROXY"] = proxy_url
    env["https_proxy"] = proxy_url
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    env["NODE_USE_ENV_PROXY"] = "1"
    if ca_cert_path is not None:
        env["NODE_EXTRA_CA_CERTS"] = str(ca_cert_path)
    return env


@cli.command(
    "exec",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.option(
    "--proxy-url",
    envvar="CACHEALIGN_PROXY_URL",
    default="http://127.0.0.1:8080",
    show_default=True,
    help="CacheAlign HTTP/CONNECT proxy URL.",
)
@click.option(
    "--ca-cert",
    envvar="NODE_EXTRA_CA_CERTS",
    type=click.Path(dir_okay=False, resolve_path=True),
    default=None,
    help="Path to CacheAlign root CA PEM exported as NODE_EXTRA_CA_CERTS.",
)
@click.option(
    "--no-proxy",
    "no_proxy",
    default="localhost,127.0.0.1,::1",
    show_default=True,
    help="Comma-separated hosts bypassing the proxy.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the resolved proxy environment and command without executing.",
)
@click.argument("command", nargs=-1, type=click.UNPROCESSED, required=True)
def exec_command(
    proxy_url: str,
    ca_cert: str | None,
    no_proxy: str,
    dry_run: bool,
    command: tuple[str, ...],
) -> None:
    """Execute a CLI agent with CacheAlign proxy and Node TLS env vars configured."""
    cmd_args = list(command[1:] if command and command[0] == "--" else command)
    if not cmd_args:
        raise click.UsageError("No command specified. Example: cachealign exec -- copilot")

    if any(ch in proxy_url for ch in ("\r", "\n", "\x00")):
        raise click.ClickException("Invalid proxy URL: control characters are not allowed.")
    parsed = urlparse(proxy_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise click.ClickException(
            f"Invalid --proxy-url '{proxy_url}': expected http(s)://host:port."
        )

    resolved_ca: Path | None = None
    if ca_cert is not None:
        candidate = Path(ca_cert)
        if not candidate.is_file():
            raise click.ClickException(f"Specified --ca-cert file does not exist: {candidate}")
        resolved_ca = candidate
    else:
        default_ca = _default_ca_cert_path()
        if default_ca.is_file():
            resolved_ca = default_ca
        else:
            console.print(
                f"[yellow]Note:[/yellow] CA cert not found at {default_ca}; "
                "NODE_EXTRA_CA_CERTS will not be set until a CA is initialized."
            )

    child_env = _build_exec_env(os.environ, proxy_url, no_proxy, resolved_ca)

    if dry_run:
        click.echo(f"PROXY={proxy_url}")
        click.echo(f"NODE_USE_ENV_PROXY={child_env['NODE_USE_ENV_PROXY']}")
        if resolved_ca is not None:
            click.echo(f"NODE_EXTRA_CA_CERTS={resolved_ca}")
        click.echo(f"COMMAND={' '.join(cmd_args)}")
        return

    try:
        completed = subprocess.run(cmd_args, env=child_env, check=False)
    except FileNotFoundError as exc:
        raise click.ClickException(f"Executable not found: {cmd_args[0]}") from exc
    if completed.returncode != 0:
        sys.exit(completed.returncode)


if __name__ == "__main__":
    cli()
