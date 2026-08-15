# 2. The run ledger is authoritative; trackers are sinks

Status: accepted (2026-08-15)

## Context

FORGE tracked experiments in Weights & Biases and never modelled run identity. The
consequence is visible in `scripts/analysis/aggregate_ssl_results.py`, which recovers
which run produced which published number using **four fallback log-filename globs per
experiment**, with hand-written "canonical run" comments. Three parallel comparison
mechanisms existed — the W&B API, grepping stdout for a `METRICS_JSON:` line, and a
bespoke 847-line offline evaluator — and they did not agree.

`algua`, by contrast, stamps `config_hash`, `snapshot_id`, `code_hash` and
`dependency_hash` onto everything, and can answer the question directly.

Large labs resolve this the same way: DeepMind, Meta and OpenAI each run their own
experiment database, and treat vendor tools as visualization over it. They can afford to
because they have platform teams; the pattern only transfers to a solo operator if
"own the ledger" means a directory of JSON files rather than a service.

## Decision

dsio owns an append-only, file-based run ledger. Each run is a directory containing
`run.json`, `config.resolved.yaml`, `metrics.jsonl`, `artifacts/`, `reproduce.sh`, and —
when the tree was dirty — `git.patch`.

MLflow and W&B are optional sinks behind the `ExperimentTracker` protocol. A sink failure
is logged loudly and never terminates a run, because the authoritative record is already
safe on disk.

The record is written *before* any work begins, so a run that crashes still has an
identity.

## Consequences

Run identity survives a change of tracking vendor, works offline, and works inside a
Kaggle kernel with no network. Comparison and the job-matrix done-ledger read the same
store.

The cost is that we own the query surface: there is no web UI, and `dsio runs list` and
`dsio runs compare` have to be good enough to be worth using. Anyone wanting a UI attaches
a sink.
