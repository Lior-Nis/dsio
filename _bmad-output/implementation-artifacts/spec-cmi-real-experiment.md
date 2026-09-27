---
title: 'Child Mind real multimodal experiment'
type: 'feature'
created: '2026-09-27'
status: 'done'
baseline_commit: 'db1fd81c398159e8043643888fb0b8dcfa7a577e'
context:
  - '{project-root}/CLAUDE.md'
  - '{project-root}/docs/research/2026-09-25-kaggle-gap-portfolio.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** DSIO has not yet been proven on a real experiment that combines tabular data with an optional, very large sensor modality. The CMI corpus provides 2,736 labelled participants, 996 sensor partitions, severe class imbalance, and 1,740 labelled participants without sensor data.

**Approach:** Add a project-owned CMI Prefect flow that composes the existing DSIO storage, split, Lightning, MLflow, evaluation, and inference blocks. Stream deterministic participant-level sensor summaries into one packed model input, retain explicit missingness and modality-presence evidence, and admit QWK to DSIO's metric registry now that two independent consumers require it.

## Boundaries & Constraints

**Always:** Use only the train/test schema intersection as model features and exclude every `PCIAT-*` field. Train all labelled participants, including those without sensor data. Preserve missingness explicitly; fit imputation/scaling state from the training fold only. Use exact `DsioModule` and `DsioDataModule`, a project-owned DAG, DSIO-governed stratified group splitting, immutable inference export, and MLflow-backed lineage and metrics. Stream each Parquet partition with bounded memory and retain participant identity through submission.

**Ask First:** Widening DSIO's single-tensor Predictor contract; adding a generic multimodal abstraction; training directly on raw sensor sequences; or changing a public DSIO interface beyond the metric registry.

**Never:** Leak `sii` or target-derived PCIAT fields into inputs; represent absent sensor data as unmarked zeros; copy the raw 6.3 GB corpus into another store; add a generic Kaggle runner, project registry, result dataclass, deployment concern, or hidden-test integration.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Mixed modalities | Labelled rows with and without sensor partitions | Both train through one model; sensor presence is explicit | Missing partitions are valid, not fabricated |
| Partial tabular data | Missing safe feature values | Missing indicators survive; fold-fitted replacement is finite | Non-numeric or inconsistent schema fails at ingestion |
| Leakage boundary | Train-only target/PCIAT columns | None enter staged input or model schema | Unexpected safe-schema drift fails before staging |
| Sensor ingestion | Valid, empty, or malformed participant Parquet | Deterministic bounded summaries for valid data | Empty/malformed files identify the participant and fail |
| Ordinal evaluation | Integer ratings on any finite ordered scale | Registered QWK matches scikit-learn | Shape, non-integer, or undefined cases raise `MetricError` |
| Replay | Same data, seed, and configuration | Split, predictions, metrics, and submission bytes agree | Evidence exposes any mismatch |

</frozen-after-approval>

## Code Map

- `src/dsio/eval/metrics.py` -- reusable metric registry and QWK implementation.
- `reference_projects/kaggle/essay_scoring/` -- first QWK consumer to migrate from duplicate logic.
- `reference_projects/kaggle/child_mind/` -- new project-owned boundary, components, tasks, and flow.
- `tests/kaggle_portfolio/` -- synthetic boundary and end-to-end portfolio coverage.
- `reference_projects/kaggle/README.md` -- dataset contract and invocation.

## Tasks & Acceptance

**Execution:**
- [x] `src/dsio/eval/metrics.py`, `tests/eval/test_metrics.py` -- register dependency-free generic QWK, pinned to scikit-learn across CMI and Essay scales.
- [x] `reference_projects/kaggle/essay_scoring/` -- consume registered QWK and remove the project duplicate.
- [x] `reference_projects/kaggle/child_mind/data.py`, `components.py` -- validate safe schemas; stream participant summaries; define packed samples, fusion model, objective, normalizer, and validator.
- [x] `reference_projects/kaggle/child_mind/tasks/`, `flow.py` -- compose ingest, split, train, export, overall/subgroup/ablation evaluation, and submission with MLflow evidence.
- [x] `tests/kaggle_portfolio/conftest.py`, `test_child_mind.py` -- cover the matrix with a source-shaped fixture and deterministic end-to-end replay.
- [x] `reference_projects/kaggle/README.md` -- document the CMI layout, optional `pyarrow` requirement, and command.
- [x] Real CMI corpus -- run ingestion over all 996 partitions and train/evaluate both tabular-only and fused configurations against the same split.

**Acceptance Criteria:**
- Given the real labelled corpus, when tabular-only and fused configurations run against the same split, then both finish through the exact DSIO spine and appear in the existing MLflow UI with training curves, modality counts, QWK, sensor-availability subgroup QWK, sensor ablation delta, model signature, and submission artifact.
- Given validation participants with sensor data, when ablation is evaluated, then it uses the same samples and checkpoint while masking only the sensor branch.
- Given the public test bundle, when inference runs, then it preserves original row order and emits exactly `id,sii` with integer predictions in `[0, 3]`.
- Given the completed change, when its public surface is inspected, then no new generic framework layer or DSIO input contract exists.

## Spec Change Log

## Design Notes

One participant becomes one fixed-width packed tensor: safe tabular values and masks, deterministic sensor summary values, and modality-presence flags. The model owns fold-fitted preprocessing buffers, separate tabular/sensor encoders, branch masking, and a four-class head. This deliberately proves full-corpus ingestion and missing-modality fusion; it does not claim raw long-sequence throughput.

## Verification

**Commands:**
- `pytest tests/eval/test_metrics.py tests/kaggle_portfolio/test_essay_scoring.py tests/kaggle_portfolio/test_child_mind.py` -- focused behavior and replay pass.
- `ruff check . && mypy src` -- repository quality gates pass.
- `pytest` -- full suite passes.
- Run `child_mind_flow` twice against `/home/liornisimov/Datasets/child-mind-institute-problematic-internet-use` -- real runs complete and MLflow evidence is inspectable.

## Suggested Review Order

**Experiment composition**

- Start with the project-owned DAG composing two modes through one DSIO spine.
  [`flow.py:22`](../../reference_projects/kaggle/child_mind/flow.py#L22)

- Ingestion retains identity and records bounded-source evidence before governed splitting.
  [`data.py:37`](../../reference_projects/kaggle/child_mind/tasks/data.py#L37)

**Data boundary**

- CSV loading excludes target-derived fields and makes missing modalities explicit.
  [`data.py:124`](../../reference_projects/kaggle/child_mind/data.py#L124)

- Parquet summarization enforces exact schemas and stable bounded-memory statistics.
  [`data.py:166`](../../reference_projects/kaggle/child_mind/data.py#L166)

**Model and training**

- One packed input supports explicit masks, separate branches, and sensor ablation.
  [`components.py:54`](../../reference_projects/kaggle/child_mind/components.py#L54)

- Exact DSIO Lightning classes train fold-fitted tabular and fused configurations.
  [`training.py:65`](../../reference_projects/kaggle/child_mind/tasks/training.py#L65)

**Evaluation and inference**

- Overall, subgroup, and same-participant ablation metrics share immutable predictions.
  [`downstream.py:106`](../../reference_projects/kaggle/child_mind/tasks/downstream.py#L106)

- Submission generation restores official identity and ordering with retained evidence.
  [`downstream.py:187`](../../reference_projects/kaggle/child_mind/tasks/downstream.py#L187)

**Reusable component and proof**

- Generic QWK preserves ordinal distances without introducing a new metric API.
  [`metrics.py:100`](../../src/dsio/eval/metrics.py#L100)

- Synthetic replay proves boundary failures, exact classes, evidence, and determinism.
  [`test_child_mind.py:355`](../../tests/kaggle_portfolio/test_child_mind.py#L355)

- Official replay IDs and browser-visible MLflow results make acceptance auditable.
  [`README.md:81`](../../reference_projects/kaggle/README.md#L81)
