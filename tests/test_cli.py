import json

from click.testing import CliRunner

from cachealign.cli.main import cli


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert "CacheAlign" in result.output


def test_cli_lint_clean(tmp_path):
    p = tmp_path / "clean_prompt.txt"
    p.write_text("You are an assistant. Follow the user instructions carefully.", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["lint", str(p)])
    assert result.exit_code == 0
    assert "No volatile prompt patterns detected" in result.output


def test_cli_lint_volatile(tmp_path):
    p = tmp_path / "bad_prompt.txt"
    p.write_text(
        "You are an assistant.\nCurrent Time: 2026-10-03 10:00:00\nExecute commands.",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["lint", str(p)])
    assert result.exit_code == 0
    assert "Detected Volatile Anti-Patterns" in result.output
    assert "Current Time:" in result.output


def test_cli_canonicalize(tmp_path):
    tools = [
        {"name": "tool_b", "parameters": {"y": 1, "x": 2}},
        {"name": "tool_a", "parameters": {"b": 1, "a": 2}},
    ]
    in_file = tmp_path / "tools.json"
    out_file = tmp_path / "tools.out.json"
    in_file.write_text(json.dumps(tools), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["canonicalize", str(in_file), "-o", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()

    with open(out_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data[0]["name"] == "tool_a"
    assert data[1]["name"] == "tool_b"
