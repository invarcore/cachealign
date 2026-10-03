# CLI Reference Guide

CacheAlign includes a powerful CLI utility to lint prompts for cache-busting antipatterns and canonicalize JSON tool definitions.

---

## 1. Installation

The CLI is included with the `cachealign` package:

```bash
pip install cachealign
cachealign --help
```

---

## 2. Commands

### `cachealign lint <FILE_PATH>`
Scans a system prompt text file or JSON request payload for volatile patterns (timestamps, turn counters, session IDs) that invalidate prompt caches.

```bash
cachealign lint system_prompt.txt
```

**Output:**
```
Scanning system_prompt.txt for cache-busting antipatterns...

               Detected Volatile Anti-Patterns (Cache-Busters)               
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Line ┃ Snippet                                ┃ Risk                     ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 2    │ Current Time: 2026-10-03 10:14:02      │ High (Invalidates Cache) │
│ 5    │ Session-ID: sess_948192                │ High (Invalidates Cache) │
└──────┴────────────────────────────────────────┴──────────────────────────┘

Recommendation: Wrap your client with cachealign.wrap() to automatically relocate
these tokens to the dynamic tail of your prompt without code changes.
```

---

### `cachealign canonicalize <INPUT_FILE> [-o <OUTPUT_FILE>]`
Reads a JSON file containing tool definitions and sorts dictionary keys lexicographically according to RFC 8785 (JSON Canonicalization Scheme).

```bash
# Output to terminal
cachealign canonicalize tools.json

# Save to canonicalized file
cachealign canonicalize tools.json -o tools.canonical.json
```

---

### `cachealign version`
Displays the installed version of CacheAlign:

```bash
cachealign version
# CacheAlign v0.1.0 — Autonomous Prompt Cache Optimizer
```

---

## 3. GitHub Actions CI Integration

You can add `cachealign lint` as a pre-commit or CI check to ensure engineering teams do not accidentally commit cache-busting system prompts:

```yaml
# .github/workflows/lint_prompts.yml
name: Lint System Prompts for Cache Stability

on: [push, pull_request]

jobs:
  lint-prompts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv pip install cachealign
      - name: Lint Prompts
        run: |
          uv run cachealign lint prompts/agent_system_prompt.txt
```