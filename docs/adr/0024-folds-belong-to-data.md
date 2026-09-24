# 24. Folds belong to data; evaluation does not own promotion

Date: 2026-09-25

## Decision

`Fold` lives in `dsio.data.splits.folds`, beside the manifests and resolution code that
produce it. It raises the existing `SplitError` used by that boundary.

Delete the unused fold-pooling artifact contract (`eval.contract` and `eval.pool`) and the
paired-run verdict mechanism (`eval.verdict`). Evaluation remains the small operation that
runs an immutable MLflow Logged Model against an immutable MLflow dataset input, logs its
predictions and named metrics to the caller-owned attempt Run, and returns the metric
values.

## Why

The project-owned Prefect flows are now the only training entrypoints. They do not write or
consume the historical per-fold `predictions.npz` schema; native MLflow model, dataset,
artifact, and metric evidence replaced it. Keeping a second unused evidence protocol would
create two answers to where evaluation evidence lives.

Promotion is a project lifecycle decision. A generic library may provide metrics, but it
cannot decide whether a metric delta is sufficient for a project's operational risk,
deployment target, or product objective. DSIO therefore does not expose a promotion gate.

## Consequences

- Split construction and its row-position invariants have one owner and one error type.
- Projects aggregate folds through their own DAG and query native MLflow evidence.
- Projects own comparison policy and promotion decisions.
- `dsio.eval.evaluate` and reusable metric implementations remain generic library code.
