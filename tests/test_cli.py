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


def test_cli_exec_exports_proxy_and_node_tls_env(tmp_path):
    import sys

    ca_file = tmp_path / "ca.crt"
    ca_file.write_text(
        "-----BEGIN CERTIFICATE-----\nTEST\n-----END CERTIFICATE-----\n", encoding="utf-8"
    )
    out_file = tmp_path / "captured_env.json"

    py_snippet = (
        "import json, os, sys; "
        f"open(r'{out_file}', 'w', encoding='utf-8').write(json.dumps({{"
        "'HTTPS_PROXY': os.environ.get('HTTPS_PROXY'), "
        "'HTTP_PROXY': os.environ.get('HTTP_PROXY'), "
        "'NO_PROXY': os.environ.get('NO_PROXY'), "
        "'NODE_USE_ENV_PROXY': os.environ.get('NODE_USE_ENV_PROXY'), "
        "'NODE_EXTRA_CA_CERTS': os.environ.get('NODE_EXTRA_CA_CERTS'), "
        "'argv': sys.argv[1:]"
        "}))"
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "exec",
            "--proxy-url",
            "http://127.0.0.1:9090",
            "--ca-cert",
            str(ca_file),
            "--",
            sys.executable,
            "-c",
            py_snippet,
            "--model",
            "gpt-4o",
        ],
    )
    assert result.exit_code == 0, result.output
    captured = json.loads(out_file.read_text(encoding="utf-8"))
    assert captured["HTTPS_PROXY"] == "http://127.0.0.1:9090"
    assert captured["HTTP_PROXY"] == "http://127.0.0.1:9090"
    assert captured["NO_PROXY"] == "localhost,127.0.0.1,::1"
    assert captured["NODE_USE_ENV_PROXY"] == "1"
    assert captured["NODE_EXTRA_CA_CERTS"] == str(ca_file.resolve())
    assert captured["argv"] == ["--model", "gpt-4o"]


def test_cli_exec_discovers_xdg_ca_dry_run_and_error_paths(tmp_path, monkeypatch):
    import sys

    xdg_ca_dir = tmp_path / "xdg" / "cachealign" / "ca"
    xdg_ca_dir.mkdir(parents=True)
    ca_file = xdg_ca_dir / "ca.crt"
    ca_file.write_text("DUMMY CA", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("NODE_EXTRA_CA_CERTS", raising=False)

    runner = CliRunner()
    dry = runner.invoke(cli, ["exec", "--dry-run", "--", "copilot", "--help"])
    assert dry.exit_code == 0
    assert "PROXY=http://127.0.0.1:8080" in dry.output
    assert "NODE_USE_ENV_PROXY=1" in dry.output
    assert f"NODE_EXTRA_CA_CERTS={ca_file}" in dry.output
    assert "COMMAND=copilot --help" in dry.output

    # Exit code propagation
    rc_res = runner.invoke(cli, ["exec", "--", sys.executable, "-c", "import sys; sys.exit(7)"])
    assert rc_res.exit_code == 7

    # Error paths
    bad_url = runner.invoke(cli, ["exec", "--proxy-url", "ftp://invalid", "--", "echo", "hi"])
    assert bad_url.exit_code != 0
    assert "Invalid --proxy-url" in bad_url.output

    bad_ctrl = runner.invoke(cli, ["exec", "--proxy-url", "http://127.0.0.1:8080\n", "--", "echo"])
    assert bad_ctrl.exit_code != 0

    missing_ca = runner.invoke(
        cli, ["exec", "--ca-cert", str(tmp_path / "missing.crt"), "--", "echo", "hi"]
    )
    assert missing_ca.exit_code != 0
    assert "does not exist" in missing_ca.output

    empty_cmd = runner.invoke(cli, ["exec", "--"])
    assert empty_cmd.exit_code != 0

    missing_bin = runner.invoke(cli, ["exec", "--", "nonexistent_cachealign_binary_xyz"])
    assert missing_bin.exit_code != 0
    assert "Executable not found" in missing_bin.output

    # Missing default CA warns gracefully in dry-run without failing
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty_xdg"))
    warn_res = runner.invoke(cli, ["exec", "--dry-run", "--", "copilot"])
    assert warn_res.exit_code == 0
    assert "CA cert not found" in warn_res.output
