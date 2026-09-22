---
baseline_commit: 0572114
---

# Story 4.4: Evaluate Immutable Evidence in a Prefect Task

Status: in-progress

## Story

As an experiment author,
I want evaluation to consume immutable model and dataset evidence in an ordinary Prefect task,
so that I can recompute metrics independently of training and deployment decisions.

## Acceptance Criteria

1. A project-owned Prefect task validates immutable compatible model and dataset references, runs the predictor, and computes reusable native metrics without invoking training.
2. Within a tracked flow, the evaluation child Run records native MLflow metrics, dataset/model inputs, a prediction artifact, and lineage to the source model and dataset Runs.
3. Changing only metric configuration or evaluation data creates new evaluation evidence under a new parent Run while reusing the unchanged predictor without retraining.
4. Incompatible prediction schema, target schema, metric inputs, or source evidence fails before any successful metric is logged and identifies the incompatible boundary.
5. Evaluation makes no promotion, approval, Registry alias, serving, or deployment decision.

## Tasks / Subtasks

- [ ] Add one framework-light evaluation function under `dsio.eval` (AC: 1-5)
  - [ ] Accept an evaluation child Run id, immutable Logged Model URI, immutable dataset source Run id, native prediction inputs/targets, and metric names.
  - [ ] Reuse `dsio.inference.predict` and existing native metric implementations; do not invoke a trainer.
  - [ ] Return a plain metric mapping rather than an evaluation result model.
- [ ] Validate evidence and evaluation contracts before metric writes (AC: 1, 4)
  - [ ] Require a writable child Run, READY DSIO model evidence with a FINISHED source Run, and one compatible native dataset input from a FINISHED source Run.
  - [ ] Validate target/prediction/score fields, cardinality, shapes, configured metrics, and finite results.
  - [ ] Revalidate source evidence before committing successful metrics.
- [ ] Record native MLflow evaluation evidence (AC: 2-3)
  - [ ] Link the Logged Model and dataset through native MLflow inputs and explicit source-Run tags.
  - [ ] Log a native NPZ prediction artifact and metric configuration before logging native metrics last.
  - [ ] Keep parent Run creation, Prefect decoration, retries, caching, and downstream lifecycle project-owned.
- [ ] Prove tracked Prefect execution, downstream-only reruns, failure-before-metrics, immutable lineage, and absence of policy/training through tests and release gates (AC: 1-5)

## Dev Notes

### Minimal shape

- Add one `evaluate` function beside existing metrics, plus one boundary error if needed. Do not add an evaluator class, evaluation result dataclass, DAG, task decorator, metric registry wrapper, promotion gate, or model alias.
- A project task composes `dsio.tracking.attempt` with `dsio.eval.evaluate`; DSIO does not decorate or schedule the task.
- Reuse MLflow `DatasetInput`, `LoggedModelInput`, metrics, tags, params, and artifacts directly.

### Evidence and metric contract

- The model URI is the immutable MLflow 3 Logged Model form already accepted by `dsio.inference.predict`. The model source Run and dataset source Run must be active, FINISHED evidence.
- The dataset source Run supplies exactly one native dataset input. Evaluation copies that dataset identity into the child Run with evaluation/source lineage tags.
- Existing `dsio.eval.metrics.compute` is the metric seam. Prediction and target arrays must align exactly; optional continuous score fields are explicit.
- Validate and compute everything before logging. Log predictions and lineage first, then successful metrics last; MLflow remains the evidence system rather than a DSIO result model.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 4.4, FR30]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Metrics, evaluation, and inference]
- [Source: `src/dsio/eval/metrics.py`]
- [Source: `src/dsio/inference/loading.py`]
- [Source: `src/dsio/tracking/attempt.py`]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 4.3 at `0572114`; selected a plain evaluation function called from a project-owned Prefect task over any DSIO task/DAG abstraction.

### Completion Notes List

### File List

- `_bmad-output/implementation-artifacts/4-4-evaluate-immutable-evidence-in-a-prefect-task.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`

### Change Log

- 2026-09-22: Created Story 4.4 and started implementation.
