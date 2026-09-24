---
status: accepted
date: 2026-09-24
amends: ADR-0019
---

# Prefect owns flow hierarchy; MLflow exposes evidence runs

Real Kaggle training exposed a usability failure in ADR-0019's original hierarchy: MLflow collapsed every metric-bearing task Run beneath one empty flow parent, so the Training Runs view showed a single Run with no metrics. Copying metrics to that parent would duplicate evidence and make multi-stage training ambiguous. Prefect already owns the flow execution, topology, and status, so DSio resolves one native MLflow Experiment for the workstream and records each Prefect task Attempt as a visible top-level MLflow Run. Attempts are related through native Prefect flow-run, task-run, dynamic-key, and retry identities; immutable MLflow inputs and source-Run references continue to express evidence lineage.

## Consequences

There is no flow-level MLflow Run and no duplicated flow status. Training, evaluation, and other evidence Runs appear directly in MLflow with their own metrics and artifacts. Retries remain append-only Runs sharing the same Prefect task-run identity and carrying distinct attempt numbers. Consumers pass an MLflow Experiment ID to `dsio.tracking.attempt(...)`; `dsio.tracking.resolve_experiment(...)` resolves the native MLflow Experiment without creating or activating a Run.
