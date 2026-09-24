---
title: 'Validate DSIO with a Kaggle consumer portfolio'
type: 'feature'
created: '2026-09-24'
status: 'done'
baseline_commit: '42874efa9bf4994a78b75e97d9fd53cb6b79d849'
context:
  - 'docs/adr/0015-lightning-is-the-only-training-path.md'
  - 'docs/adr/0019-versioned-library-with-project-owned-prefect-flows.md'
  - 'docs/superpowers/specs/2026-09-18-generic-experiment-spine.md'
  - '_bmad-output/implementation-artifacts/spec-canonical-supervised-reference-cleanup.md'
  - '_bmad-output/implementation-artifacts/spec-harden-self-supervised-reference-flow.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Synthetic references prove the intended architecture but do not show that unrelated, source-shaped data-science problems can use DSIO without project modes, hidden leakage, or a second training path.

**Approach:** Add three small consumer projects for Kaggle Titanic, Bike Sharing Demand, and Digit Recognizer. Each accepts an already-downloaded official CSV directory, owns its Prefect DAG and competition logic, and runs through public DSIO storage, splitting, Lightning, MLflow, evaluation, inference, and immutable evidence. Offline tests use generated schema-faithful fixtures; real downloads remain opt-in because Kaggle credentials and rules acceptance are human-owned.

## Boundaries & Constraints

**Always:** Use exact `DsioModule` and `DsioDataModule` classes; keep competition code outside `src/dsio`; preserve source identifiers and submission order; exclude official test rows from fitting and validation; record complete dataset, split, execution, checkpoint, model, and transferred-encoder lineage; keep fixtures synthetic, tiny, deterministic, and shaped like official schemas.

**Ask First:** Any `src/dsio` change, competition substitution, new runtime dependency, network access, Kaggle credential use, official-data redistribution, split-semantics change, or public API change.

**Never:** Add a generic Kaggle runner, project registry, DAG abstraction, result dataclass, submission API, project mode flag, or competition dependency to DSIO; import one competition project from another; call legacy training runners; commit Kaggle rows; submit to Kaggle; claim leaderboard or full-data validation from fixture runs.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Competition run | Official-schema `train.csv` and `test.csv` under an explicit directory | Project-owned flow logs immutable evidence and writes a correctly ordered submission | Missing columns, duplicate IDs, invalid values, or shape drift fail before training |
| Grouped classification | Titanic passengers sharing `Ticket` | Ticket groups remain disjoint; unlabeled test passengers never enter a split | Leakage or incomplete labeled coverage is rejected |
| Temporal regression | Ordered Bike Sharing hourly rows | One final purged holdout is causal; label horizon and discarded band are explicit | Future training rows or overlap fail validation |
| Representation reuse | Digit labels withheld during pretraining | Verified encoder artifact feeds a separate frozen downstream classifier run | Failed, deleted, corrupted, mismatched, or label-tainted evidence is rejected |

</frozen-after-approval>

## Code Map

- `reference_projects/kaggle/README.md` -- official slugs, data placement, opt-in download commands, and honest validation scope.
- `reference_projects/kaggle/{titanic,bike_sharing,digit_recognizer}/` -- independent importable components, task packages, and project-owned Prefect flows.
- `tests/kaggle_portfolio/conftest.py` -- isolated MLflow/Prefect services and generated source-shaped CSV fixtures.
- `tests/kaggle_portfolio/test_{titanic,bike_sharing,digit_recognizer}.py` -- public end-to-end, leakage, replay, lineage, and submission contracts.
- `tests/test_built_distribution.py` -- copied consumers run against only the installed DSIO wheel.
- `_bmad-output/implementation-artifacts/deferred-work.md` -- record causal multi-fold temporal semantics as separate core work.

## Tasks & Acceptance

**Execution:**
- [x] `reference_projects/kaggle/titanic/`, `tests/kaggle_portfolio/test_titanic.py` -- ingest mixed CSV fields into `SignalStore`, split labeled passengers by ticket, train/export/evaluate/infer, and produce `PassengerId,Survived` deterministically.
- [x] `reference_projects/kaggle/bike_sharing/`, `tests/kaggle_portfolio/test_bike_sharing.py` -- encode hourly features, generate one final `purged_walk_forward` holdout with nonzero horizon/embargo, fit train-only preprocessing, and produce ordered non-negative `datetime,count` predictions.
- [x] `reference_projects/kaggle/digit_recognizer/`, `tests/kaggle_portfolio/test_digit_recognizer.py` -- pretrain without labels, persist and verify encoder evidence, train a separate frozen classifier, and produce ordered `ImageId,Label` predictions.
- [x] `reference_projects/kaggle/README.md`, `tests/test_built_distribution.py` -- document credentialed real-data commands and prove all fixture flows remain unshipped consumer code runnable against the built wheel.
- [x] Run focused portfolio tests, broad tests, installed-distribution consumers, lint, typing, import contracts, and diff hygiene.

**Acceptance Criteria:**
- Given the same fixture inputs and seed, when each flow runs twice, then split digests, node identities, predictions, metrics, encoder digest where applicable, and submission bytes match while MLflow run IDs differ.
- Given each training task, when Lightning executes, then runtime types are exactly `DsioModule` and `DsioDataModule`, loader membership equals recorded split assignments, and requested/resolved execution settings agree.
- Given a copied portfolio and installed DSIO wheel outside the checkout, when offline probes run, then all three flows finish without repository-source imports, Kaggle credentials, network access, or packaged consumer code.

## Spec Change Log

## Design Notes

Titanic and Bike Sharing are deliberately smaller than more elaborate alternatives: the portfolio validates DSIO seams, not leaderboard ambition. Digit Recognizer is larger but CSV-native and gives the cleanest label-free encoder-to-classifier handoff. Temporal validation uses only the final single fold because the current multi-fold algorithm admits post-test training rows; changing that stable semantic is core split work, not a competition workaround.

## Verification

**Commands:**
- `uv run pytest -q tests/kaggle_portfolio` -- all fixture flows and boundary failures pass.
- `uv build && uv run pytest -q tests/test_distribution.py tests/test_built_distribution.py tests/test_project_flow.py` -- installed-wheel consumers pass.
- `uv run --group benchmark pytest -q --ignore=tests/test_distribution.py --ignore=tests/test_built_distribution.py --ignore=tests/test_project_flow.py` -- broad suite passes.
- `uv run ruff check . && uv run mypy && uv run lint-imports && git diff --check` -- static and architecture gates pass.

## Suggested Review Order

**Portfolio shape**

- Start with the independent consumer boundary, official schemas, and honest validation scope.
  [`README.md:1`](../../reference_projects/kaggle/README.md#L1)

- Titanic keeps orchestration visible while delegating cohesive task responsibilities.
  [`titanic/flow.py:21`](../../reference_projects/kaggle/titanic/flow.py#L21)

- Bike exposes the same public spine without a shared competition framework.
  [`bike_sharing/flow.py:22`](../../reference_projects/kaggle/bike_sharing/flow.py#L22)

- Digit makes pretraining and downstream transfer explicit in one project-owned DAG.
  [`digit_recognizer/flow.py:22`](../../reference_projects/kaggle/digit_recognizer/flow.py#L22)

**Governed data and splits**

- Titanic validates source shape and isolates ticket groups before canonical loading.
  [`titanic/tasks/data.py:52`](../../reference_projects/kaggle/titanic/tasks/data.py#L52)

- Bike records UTC-stable data and one deliberately causal final purged holdout.
  [`bike_sharing/tasks/data.py:65`](../../reference_projects/kaggle/bike_sharing/tasks/data.py#L65)

- Digit creates label-bearing storage while exposing a label-free pretraining dataset.
  [`digit_recognizer/tasks/data.py:34`](../../reference_projects/kaggle/digit_recognizer/tasks/data.py#L34)

**Canonical training and transfer**

- Titanic records every output-changing Lightning, batching, and optimizer input.
  [`titanic/tasks/training.py:53`](../../reference_projects/kaggle/titanic/tasks/training.py#L53)

- Bike fits preprocessing only from governed training assignments.
  [`bike_sharing/tasks/training.py:64`](../../reference_projects/kaggle/bike_sharing/tasks/training.py#L64)

- Digit pretraining persists label-free, content-addressed encoder evidence.
  [`digit_recognizer/tasks/training.py:97`](../../reference_projects/kaggle/digit_recognizer/tasks/training.py#L97)

- Downstream training verifies source provenance before freezing the transferred encoder.
  [`digit_recognizer/tasks/training.py:193`](../../reference_projects/kaggle/digit_recognizer/tasks/training.py#L193)

**Model and submission evidence**

- Export identity binds evaluation and inference to the complete Predictor definition.
  [`digit_recognizer/tasks/downstream.py:37`](../../reference_projects/kaggle/digit_recognizer/tasks/downstream.py#L37)

- Submissions remain local while MLflow receives content-addressed immutable evidence.
  [`titanic/tasks/downstream.py:124`](../../reference_projects/kaggle/titanic/tasks/downstream.py#L124)

**Acceptance proof**

- End-to-end tests prove replay, exact training types, membership, and lineage failures.
  [`test_digit_recognizer.py:148`](../../tests/kaggle_portfolio/test_digit_recognizer.py#L148)

- Built artifacts prove copied consumers run only against the installed DSIO wheel.
  [`test_built_distribution.py:256`](../../tests/test_built_distribution.py#L256)

- Deferred work preserves cross-cutting improvements without portfolio-specific abstractions.
  [`deferred-work.md:12`](deferred-work.md#L12)
