---
title: 'Make the supervised reference canonical'
type: 'refactor'
created: '2026-09-23'
status: 'done'
baseline_commit: 'b4686cd6ee202d719e7e305213abdbe0ab661de2'
context:
  - 'docs/adr/0015-lightning-is-the-only-training-path.md'
  - 'docs/adr/0019-versioned-library-with-project-owned-prefect-flows.md'
  - 'docs/superpowers/specs/2026-09-18-generic-experiment-spine.md'
  - '_bmad-output/implementation-artifacts/spec-harden-self-supervised-reference-flow.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The supervised reference reaches the exact DSIO Lightning classes but bypasses the shared Trainer construction, leaves loader and execution behavior implicit, records incomplete export lineage, and silently trains on time-major tensors only because its one-channel model flattens either layout. This makes it an unsafe template for real Kaggle consumers.

**Approach:** Make the supervised project follow the same canonical composition as the hardened SSL reference: one recorded Trainer configuration, explicit loading policy, requested/resolved capability evidence, channel-first training with deterministic export preprocessing, and complete MLflow provenance.

## Boundaries & Constraints

**Always:** Keep the project-owned Prefect DAG and the exact `DsioModule`/`DsioDataModule` classes; preserve deterministic replay, reevaluation without retraining, flow return contracts, validation coverage, and all assigned samples; keep raw Predictor input time-major and model input channel-first; make runtime behavior and recorded configuration agree.

**Ask First:** Any change to existing flow outputs, split membership, metrics, checkpoint semantics, public DSIO defaults, or the external Predictor schema.

**Never:** Add a runner, result model, orchestration abstraction, project registry, or supervised-only configuration type; migrate or delete legacy `torch_task`/`ssl_task`/`make_loader` compatibility paths; put Kaggle-specific code in this change; promote the reference adapter into `src/dsio`; modify the four repository-root review documents.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Supervised training | Recorded trainer and four-phase loader mappings | Shared builders create the exact runtime; declared and resolved values enter provenance | Unsupported requested capability fails before Trainer construction |
| Training sample | Store-shaped `[time, channels]` sample | Dataset returns contiguous `[channels, time]`; model validates the declared shape | Wrong rank or extent fails at the model boundary |
| Export/inference | Predictor receives `[batch, time, channels]` | Recorded deterministic adapter supplies the same values/layout used in training | Wrong external rank or extent fails clearly through Predictor |
| Validation loading | Train/test split mapped to train/validate | Every assigned identity is retained; no phase drops a remainder | Any observation-phase truncation remains rejected |

</frozen-after-approval>

## Code Map

- `reference_projects/supervised/components.py` -- supervised dataset, model shape contract, and shared signal-layout adapter.
- `reference_projects/supervised/tasks/training.py` -- canonical training composition and provenance.
- `reference_projects/supervised/tasks/export.py` -- extracted Predictor export and lineage.
- `reference_projects/supervised/tasks/__init__.py` -- stable task re-export.
- `reference_projects/self_supervised/{components.py,tasks/export.py}` -- consume the shared project-owned adapter without changing SSL behavior.
- `src/dsio/train/trainer.py` -- express and forward the existing sanity-validation setting.
- `tests/train/test_trainer.py` -- exact native Trainer mapping.
- `tests/reference_flows/{test_supervised_flow.py,test_self_supervised_flow.py}` -- public layout, runtime/provenance, replay, and inference contracts.
- `_bmad-output/implementation-artifacts/deferred-work.md` -- immediate Kaggle portfolio follow-up.

## Tasks & Acceptance

**Execution:**
- [x] `tests/train/test_trainer.py`, `src/dsio/train/trainer.py` -- specify and add a non-negative `num_sanity_val_steps` setting, preserving Lightning's default for existing callers and forwarding an explicit supervised value exactly.
- [x] `tests/reference_flows/test_supervised_flow.py`, `reference_projects/supervised/{components.py,tasks/training.py}` -- test-first declare one `_TRAINER`, explicit shuffle/drop-last mappings, requested/resolved capability evidence, and exact runtime/provenance parity; preserve three CPU epochs, one device, all validation batches, disabled automatic checkpoints, zero sanity steps, and every sample identity.
- [x] `tests/reference_flows/test_{supervised,self_supervised}_flow.py`, `reference_projects/{supervised,self_supervised}` -- prove non-square multichannel layout values, centralize the project-owned adapter, make supervised datasets/model channel-first, and attach the configured adapter to both exported Predictors without changing their external schema.
- [x] `reference_projects/supervised/tasks/{training.py,export.py,__init__.py}` -- separate training from export once responsibilities are canonical; record split, model, preprocessor, full configuration, and effective execution evidence without changing imports or flow outputs.
- [x] Run focused reference/trainer tests, the broad suite, installed-distribution consumers, lint, typing, import contracts, and diff hygiene.

**Acceptance Criteria:**
- Given a supervised flow run, when training is assembled, then one serializable configuration drives Trainer, callbacks, DataModule, Execution Identity, and MLflow parameters with no inline second builder.
- Given raw store values, when they pass through training and exported inference, then both routes provide identical channel-first tensors to the model and reject ambiguous shapes.
- Given the same seed and inputs, when the full flow is replayed or downstream evaluation is rerun, then assignments, identities, predictions, lineage, and no-retraining behavior remain deterministic.
- Given an installed DSIO wheel plus copied consumer projects, when both reference flows execute, then they use only public DSIO APIs and complete successfully.

## Spec Change Log

## Design Notes

The signal-layout adapter remains project-owned: a second synthetic reference use is not independent downstream evidence for admission into DSIO. It lives with the supervised synthetic signal components already reused by the SSL flow. Legacy registered runners remain compatibility code and are a separate deletion/migration decision; this cleanup makes the supported consumer-owned path unambiguous without mixing in that blast radius.

## Verification

**Commands:**
- `uv run pytest -q tests/train/test_trainer.py tests/reference_flows/test_supervised_flow.py tests/reference_flows/test_self_supervised_flow.py` -- focused public contracts pass.
- `uv run --group benchmark pytest -q --ignore=tests/test_distribution.py --ignore=tests/test_built_distribution.py --ignore=tests/test_project_flow.py` -- broad suite passes.
- `uv build && uv run pytest -q tests/test_distribution.py tests/test_built_distribution.py tests/test_project_flow.py` -- built consumers pass.
- `uv run ruff check . && uv run mypy && uv run lint-imports && git diff --check` -- static and architecture gates pass.

## Suggested Review Order

**Canonical training composition**

- Start here: one explicit configuration governs supervised runtime, loading, capabilities, and provenance.
  [`training.py:35`](../../reference_projects/supervised/tasks/training.py#L35)

- Shared construction replaces the reference project's inline Lightning assembly.
  [`training.py:68`](../../reference_projects/supervised/tasks/training.py#L68)

- The shared schema exposes sanity validation without changing Lightning's default.
  [`trainer.py:16`](../../src/dsio/train/trainer.py#L16)

**Signal layout and export**

- Training samples now cross the model boundary in contiguous channel-first layout.
  [`components.py:18`](../../reference_projects/supervised/components.py#L18)

- One strict adapter preserves time-major Predictor inputs while validating extents.
  [`components.py:45`](../../reference_projects/supervised/components.py#L45)

- Model shape validation makes ambiguous layout errors immediate.
  [`components.py:74`](../../reference_projects/supervised/components.py#L74)

- Export is isolated and records checkpoint, data, split, preprocessor, and component lineage.
  [`export.py:26`](../../reference_projects/supervised/tasks/export.py#L26)

- SSL consumes the canonical adapter while retaining its own export behavior.
  [`self_supervised/export.py:23`](../../reference_projects/self_supervised/tasks/export.py#L23)

- A compatibility re-export keeps historical SSL component references resolvable.
  [`self_supervised/components.py:17`](../../reference_projects/self_supervised/components.py#L17)

**Contract evidence and follow-up**

- Non-square multichannel values prove training and inference layouts match exactly.
  [`test_supervised_flow.py:20`](../../tests/reference_flows/test_supervised_flow.py#L20)

- End-to-end evidence covers replay, effective runtime, MLflow parameters, and export lineage.
  [`test_supervised_flow.py:64`](../../tests/reference_flows/test_supervised_flow.py#L64)

- Shared Trainer tests pin exact sanity-step forwarding and validation.
  [`test_trainer.py:86`](../../tests/train/test_trainer.py#L86)

- The review-caught historical component path now has a regression contract.
  [`test_self_supervised_flow.py:21`](../../tests/reference_flows/test_self_supervised_flow.py#L21)

- Cross-project lineage hardening and Kaggle validation remain explicit follow-up work.
  [`deferred-work.md:9`](deferred-work.md#L9)
