# LLM Prompt-Cache Longitudinal Observatory (cache-probe / Candidate C1)

## Referee Independence & Relationship Disclosure

This dataset records append-only, cryptographically anchored wire observations of LLM provider prompt-cache semantics (`P1_ENGAGEMENT_THRESHOLD`, `P2_TTL_BOUNDARY_SURVIVAL`, `P3_RETENTION_PARAM_HONORING`, `P4_USAGE_FIELD_PARITY`, and `P5_ROUTE_AFFINITY_DRIFT`).

**Disclosure:** Fixed-length deterministic 1,200-token prefix blocks used in these probes are generated using `invarcore/cachealign` (`build_aligned_prefix`). Probe definitions are committed prior to execution, and daily JSONL ledger SHA-256 digests are anchored externally via signed Git tags and GitHub Actions CI artifacts.

## Cadence

* **Daily Smoke Probes (`--mode daily-smoke`):** Executes `cold_seed` + `warm_10s` daily to pinpoint the exact calendar date any provider or gateway alters minimum token thresholds or usage-field reporting.
* **Monthly Full Survival Curves (`--mode full`):** Executes multi-delay TTL boundary (`5m` / `1h`) and load-balancer worker-affinity sweeps on the 1st of each month.

## Automated CI Verification

Daily smoke probes and physical invariant tests run automatically in GitHub Actions via [.github/workflows/live-eval.yml](../.github/workflows/live-eval.yml).
Official run ledger anchors are archived in [observatory/ANCHORS.md](ANCHORS.md).
