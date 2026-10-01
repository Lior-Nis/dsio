# Story 6.5: Place every component where its evidence puts it

Status: done

## Story

As a consumer reading the catalog,
I want a component's location to tell me its maturity truthfully,
so that "stable" always means proven by two unrelated real uses.

## Acceptance Criteria

1. Given `docs/component-admission.md`, when the pre-1.0 legacy clause is added, then it states that pre-existing components with no real use move to `dsio.experimental`, carry no compatibility promise, and are deleted at 1.0 if still unproven. It also states that uses are counted per public callable and that two consumers of one competition count as one use.
2. Given the experimental package, when the layout is created, then `dsio.experimental.data`, `.model`, `.train`, `.inference` and `.eval` exist, mirroring the stable domains. An import-linter contract forbids stable packages from importing `dsio.experimental`.
3. Given each component in the cohort's legacy list, when it is reclassified, then it moves to its `dsio.experimental.<domain>` module. Tests and fixtures import the new path with no compatibility shim (pre-1.0), and the self-supervised reference flow still passes.
4. Given every public object under `dsio.experimental`, when the test suite runs, then `require_admissible_component` passes for each one against every reference consumer package name, including `calibrate_training_execution`, `measure_phase` and `log_phase_evidence`.
5. *(Moved to Story 6.4, since 6.5 is implemented first: the CI check that stable-located components list two unrelated real uses needs the evidence register 6.4 introduces.)*

## Tasks / Subtasks

- [x] Task 1: Amend `docs/component-admission.md` with "Counting uses" and "Legacy components (pre-1.0)" sections (AC 1). `CONTEXT.md` gains the **Legacy Experimental Component** term.
- [x] Task 2: Create the experimental domain packages and add the import-linter contract "Stable packages never import experimental components". It is pinned in `tests/test_import_contracts.py` (AC 2).
- [x] Task 3: Move every component with fewer than two unrelated real uses out of the stable packages (AC 3), with `git mv` to preserve history.

  | From | To |
  |---|---|
  | `dsio/model/{components,chain,masking}.py` | `dsio/experimental/model/` |
  | `dsio/train/{augmentation,callbacks}.py` | `dsio/experimental/train/` |
  | `dsio/data/loading/windows.py`, `dsio/data/labels.py` | `dsio/experimental/data/` |
  | `StoredSamples`/`stored_samples`, split from `dsio/data/loading/datasets.py` | `dsio/experimental/data/samples.py` |
  | `TensorOutput`/`validate_tensor_prediction`, split from `dsio/inference/predictor.py` | `dsio/experimental/inference/outputs.py` |
  | `dsio/eval/ess.py` | `dsio/experimental/eval/` |

  - Stable re-exports are removed: `dsio.model.module.export_encoder`, `dsio.data.loading.{StoredSamples,stored_samples,WindowDataset}`, `dsio.inference.{TensorOutput,validate_tensor_prediction}`.
  - Every import site and `module:qualname` string in tests, the two synthetic fixtures, the README and live docs is updated.
- [x] Task 4: `tests/experimental/test_admission_coverage.py` (AC 4) walks every experimental module's `__all__` and audits each defining module against the 8 Kaggle consumer package names.
  - The auditor itself is excluded, along with type aliases.
  - The source audit covers the whole module, so one object per module is audited, and the rest get the cheap namespace check. This takes 24 s instead of 183 s for per-object auditing, with identical coverage.

- [x] Task 5: Review follow-ups (no critical findings).
  - [x] 5.1 `dsio.experimental.telemetry` declares `__all__`, so `measure_phase` and `log_phase_evidence` are audited. A test asserts the AC 4 components appear in the audited set.
  - [x] 5.2 Consumer names derive from `reference_projects/kaggle/*`, not a hardcoded list.
  - [x] 5.3 "Unrelated" means a different consumer on different data (the stricter "and a different task" wording contradicted the spec's counting). The legacy clause states its exemption from the Compatibility section's review requirement.
  - [x] 5.4 A test asserts every non-experimental `dsio` package is in the stable-never-imports-experimental contract. The wheel test asserts the five experimental domain packages ship.
  - [x] 5.5 Stale docstrings fixed:
    - the `mae_decoder_head` export path;
    - the "Canonical factory" wording;
    - two test comments;
    - the `dsio.model` package docstring.

    The moved modules carry the legacy note.

## Dev Notes

- **Data infrastructure is spine, not warehouse.** The store, views, examples/adapters (`SignalExamples`, `build_index`, `WindowSpec`, `entity_examples`), staging, splits and the `dsio.testing` contract suites stay in stable packages even with fewer than two consumer uses. They are the data layer every block builds on, not injectable Components (`CONTEXT.md`). The review flagged `dsio.data.staging.stage` (tests only) and single-use views/adapters; they are recorded here as spine. Story 6.4's location-vs-evidence check applies to catalogued components only.

- **What stays stable.** Only the spine and the closed dispatchers:
  - `DsioModule`/`DsioDataModule`;
  - identity dataset, collation and loader;
  - Predictor, export, predict and lineage;
  - `evaluate`/`METRICS`;
  - split algorithms, store and tracking;
  - trainer and capabilities;
  - calibration and telemetry, already in `dsio.experimental`.

  No model-side component had a real (Kaggle) use, so warehouse blocks are reshaped from `dsio.experimental.*` in Epics 7–9 and promoted in Epic 11.
- **Why not delete.** Owner decision 2026-09-30: legacy clause instead of strict deletion. `bce_loss`/`mse_loss` are superseded in Story 7.4, and `TensorOutput`/`validate_tensor_prediction` in Story 7.5; they are deleted then.
- **Why the fixture names are not audited.** `supervised`/`self_supervised` are generic ML terms, and the audit matches snake_case identifier segments, so every SSL identifier would false-positive. They are fixtures, not real uses (spec counting rule).
- **Why 6.5 comes before 6.4.** The catalog's docstring sections should be written once, at final locations.

## Dev Agent Record

### Agent Model Used

Claude Opus 5.5 (claude-opus-5-5)

### Completion Notes List

- The adversarial review found nothing critical. It verified:
  - no Kaggle consumer imports anything moved;
  - no stale paths remain;
  - there are no shims and no circular imports;
  - the wheel ships all new files;
  - one-object-per-module auditing is complete.

  Its important and minor findings are fixed (Task 5), apart from the spine note above.
- Gates: full suite 1043 passed, 1 deselected (405 s). `ruff check`, `ruff format --check`, `mypy` (104 files) and `lint-imports` (3 contracts kept) are clean.

### File List

- `docs/component-admission.md`, `CONTEXT.md`, `README.md`, `pyproject.toml`
- `src/dsio/experimental/{data,model,train,inference,eval}/` (new packages; moved modules)
- `src/dsio/data/loading/{__init__,datasets}.py`, `src/dsio/inference/{__init__,predictor}.py`, `src/dsio/model/module.py`
- `tests/experimental/test_admission_coverage.py` (new), `tests/test_import_contracts.py`, plus import updates across `tests/` and `reference_projects/{supervised,self_supervised}`
- `_bmad-output/planning-artifacts/epics-component-warehouse.md` (6.4/6.5 order note), `_bmad-output/implementation-artifacts/sprint-status.yaml`
