---
stepsCompleted: [1, 2, 3, 4]
inputDocuments:
  - _bmad-output/specs/spec-component-warehouse/SPEC.md
  - _bmad-output/specs/spec-component-warehouse/component-cohort.md
  - _bmad-output/specs/spec-component-warehouse/conventions.md
  - _bmad-output/specs/spec-component-warehouse/brownfield.md
  - _bmad-output/specs/spec-component-warehouse/roadmap.md
  - docs/component-admission.md
  - docs/superpowers/specs/2026-09-18-generic-experiment-spine.md
  - CONTEXT.md
continues: _bmad-output/planning-artifacts/epics.md  # Epics 1-5, all done
---

# DSio Component Warehouse v1 - Epic Breakdown

## Overview

This document breaks the Component Warehouse v1 spec (`SPEC-component-warehouse`) into
implementable epics and stories. It continues the completed DSio epic breakdown in
`epics.md`, so numbering starts at Epic 6. Capability IDs (CAP-n) and cohort block numbers
(#n) refer to the spec folder. "Kaggle consumers" means the 9 Kaggle reference consumers.
The `supervised` and `self_supervised` references are fixtures.

## Requirements Inventory

### Functional Requirements

FR1 (CAP-1): A consumer builds models for flat/tabular, token-sequence, signal-sequence (whole and dense per-timestep) and multimodal missing-modality inputs by composing DSio backbones, heads, layout adapters, standardization slots and named compositions, with no local `nn.Module`.

FR2 (CAP-2): A consumer selects DSio objectives covering:
- binary, multiclass and ordinal classification, with class weights (native semantics) and sample weights (mean-1 rule);
- regression, including transformed-target space;
- boolean-masked dense losses;
- reconstruction.

Each honors `(model, batch, stage) -> {loss, ...}` with configured auxiliary metrics.

FR3 (CAP-3): A consumer maps stored samples and windows into items. The mapping covers:
- field selection into `x`/`y`/`mask`/`sample_weight`/`group`;
- dtype and layout;
- target transform;
- optional labels.

Datasets are supplied as ComponentConfigs whose parameters enter provenance, and the store digest check stays in force.

FR4 (CAP-3): A consumer pads variable-length batches with a DSio collator. It stacks fixed fields, optionally emits a True-is-valid `mask`, and has an interface smaller than the implementations it replaces.

FR5: *Withdrawn.* Labelled-example selection fails the depth test (cohort #6).

FR6 (CAP-4): A consumer derives class-frequency and group-frequency weights on training-role samples. Group weights are normalized to mean 1 over the training role, so weighted losses are invariant to micro-batch partitioning.

FR7 (CAP-4): A consumer fits standardization statistics (NaN-aware, observed-only option) on training-role samples only. The fitted values are logged as evidence and injected as runtime arguments. A config/runtime collision raises.

FR8 (CAP-5): A consumer declares binary, multiclass, ordinal, regression (inverse transform, non-negative) and dense per-timestep outputs. Each comes with a parameterized validator and an MLflow signature.

FR9 (CAP-6): A consumer builds evaluation and inference arrays in `dsio.data` with the same dataset and collation used in training.

FR10 (CAP-6): A consumer evaluates through `dsio.eval` with subgroup metrics and modality-ablation deltas over consumer-supplied arrays.

FR11 (CAP-7): A consumer's own task trains through one DSio fit function. The function receives project-constructed roots (or factories, for calibration) and a `TrainerConfig`, and calls native `Trainer.fit` once. It adds only these invariants:
- capability evidence;
- provenance;
- optional calibration;
- a digest-verified checkpoint.

It takes no task or mode argument.

FR12 (CAP-7): A consumer's own task exports a Predictor through one DSio export function, from checkpoint lineage to a Logged Model.

FR13 (CAP-6, CAP-7): A consumer's downstream tasks compose FR9 arrays with the existing `evaluate`/`predict`. No new DSio evaluate function is added.

FR14 (CAP-8): A consumer discovers every warehouse component in a catalog generated from docstring sections plus the evidence and candidates registers. CI rejects missing sections, missing evidence, nonexistent consumer or test paths, and a stale catalog.

FR15 (CAP-9): Warehouse maturity follows recorded per-callable evidence, under an admission doc amended with a pre-1.0 legacy clause:
- legacy unproven components move to `dsio.experimental`;
- the audit covers every experimental component;
- promotions record human approval.

FR16 (CAP-10): Each migrated consumer records metric parity against committed golden contract-tier metrics and recorded representative-tier baselines. Metrics must match exactly unless a story declares its computation change and tolerance in advance. The replay-identity flake is resolved first.

FR17 (CAP-11): DSio 0.3.0 is released with a single-sourced version, a `v0.3.0` tag, green distribution/consumer CI, and release notes listing each component's maturity and evidence.

FR18 (Success signal): A new reference consumer on a dataset not yet in the portfolio is written outside the repository, given only the installed 0.3.0 wheel, README and catalog. It contains only ingestion, feature/label mapping, configuration, flow topology and submission formatting.

### NonFunctional Requirements

NFR1: Components are or return native PyTorch/Lightning/TorchMetrics/MLflow objects. There is no DSio component base class, and no result, evaluation or batch dataclass.

NFR2: Compositions are separate named factories. There are no task-mode flags, no dispatch on task or project names, and no universal collator or model.

NFR3: Selection is by `module:qualname` ComponentConfig only. Factories return module-level classes, never `<locals>` classes, lambdas or closures. There is no runtime registry or entry point.

NFR4: `DsioModule`/`DsioDataModule` remain the only training classes. Projects construct them and never subclass them. DSio functions never define Prefect flows or tasks, and never decide topology, retries or scheduling.

NFR5: The model input stays a single `x`, and the Predictor/export API is not widened. Model-side validity and modality presence are declared channels of `x`.

NFR6: Import contracts hold, including `dsio.eval` importing no `data`/`model`/`train`/`tracking`. A new contract forbids stable packages from importing `dsio.experimental`.

NFR7: Maturity is decided by per-callable evidence. Kaggle consumers are real uses. Fixtures are not. The CMI pair counts as one use. Stable status requires an unrelated second use plus human approval.

NFR8: Depth test: every admitted block removes meaningful consumer implementation. Thin renames and one-line filters are rejected.

NFR9: Code is placed by pipeline responsibility (`data`, `model`, `train`, `eval`, `inference`, `tracking`). A concern starts as a file and becomes a package when justified, with re-exported public imports.

NFR10: Determinism under seed. Accelerator augmentation runs only in `training_step`. Deterministic preprocessing lives in the Predictor. There is no silent device fallback.

NFR11: Calibration changes execution knobs only, and every candidate and selection is logged.

NFR12: Fitted statistics and weights use training-role samples only. The ordered `sample_id` guard and the digest check stay intact.

NFR13: Consumers own ingestion, domain semantics, mapping, topology, configuration, submission, promotion and deployment. DSio source (including docstrings) names no consumer or competition.

NFR14: Components depend only on the admission allowlist. Package metadata stays accelerator-neutral.

NFR15: The full gate stays green on every story: `pytest`, `ruff check`, `ruff format --check`, `mypy`, `lint-imports`, and distribution/consumer CI.

### Additional Requirements

- Conventions (`conventions.md`) are normative:
  - batch fields `sample_id`, `x`, `y`, `mask` (True = valid, objectives and evaluation only), `sample_weight`, `group`, and `hidden` for SSL corruption;
  - the weighting rule;
  - layouts, with no silent squeeze;
  - zero-based class targets;
  - catalog docstring sections Consumes, Produces, Parameters, Devices, Limitations, Example;
  - the evidence register `docs/component-warehouse/evidence.yaml` and the candidates register `docs/component-warehouse/candidates.yaml`;
  - catalog generator under `tools/`;
  - the export round-trip test.
- The admission process (`docs/component-admission.md`) governs every new block. Each needs unit tests, an integration/property test, deterministic replay evidence, provenance assertions, one real downstream confirmation, limitations, an adversarial review, and explicit human approval of entry.
- **Supersession:** the 2026-09-25 portfolio guardrails (consumers own model choice/loss; do not generalize collators yet) are superseded. The collator caution becomes the depth constraint on #3.
- **Spine amendment:** an ADR defines "universal trainer" as a DSio-owned loop or task-dispatching entry point, and shows the CAP-7 functions are neither.
- **Brownfield:** there is no starter template. All work is in-place refactoring of `src/dsio/` and `reference_projects/`.
- **Baseline:** Story 6.6 records the authoritative non-ingestion line baseline and golden metrics. `brownfield.md` gives the initial figures.
- CUDA scale runs need a consumer-side CUDA torch override and a run-isolated `PREFECT_HOME`.

### UX Design Requirements

Not applicable (library; no UI).

### FR Coverage Map

FR1: Epics 7, 8, 9 - flat/tabular (7), token and signal sequence (8), multimodal (9) model composition
FR2: Epics 7, 8 - supervised and reconstruction objectives (7); masked dense objectives (8)
FR3: Epics 7, 9 - stored-sample field mapping (7); window items (9)
FR4: Epic 8 - pad collator
FR5: Withdrawn (fails depth test)
FR6: Epic 9 - class- and group-frequency weights
FR7: Epics 7, 9 - fitted standardization (7); observed-only statistics (9)
FR8: Epics 7, 8 - binary, multiclass, ordinal and regression outputs (7); dense per-timestep outputs (8)
FR9: Epics 7, 8 - evaluation arrays from training collation
FR10: Epic 9 - subgroup metrics and ablation deltas
FR11: Epic 10 - fit function
FR12: Epic 10 - export function
FR13: Epic 10 - downstream tasks over DSio arrays
FR14: Epics 6, 11 - catalog, registers and CI checks (6); complete catalog at release (11)
FR15: Epics 6, 11 - legacy clause, reclassification and audit (6); evidence-based promotion (11)
FR16: Epics 6, 7, 8, 9, 10 - flake and baseline (6); parity per migrated consumer (7-10)
FR17: Epics 6, 11 - single-sourced version (6); 0.3.0 release (11)
FR18: Epic 11 - warehouse-first acceptance consumer

## Epic List

### Epic 6: Trust and discover what DSio ships
A consumer can see every DSio component, with its contract, maturity and evidence, in a CI-checked catalog. They can rely on a single-sourced version, a consistent batch-field vocabulary, and replay identity that does not flake. Every component sits where its evidence places it. Golden baselines exist for every consumer.
**FRs covered:** FR14, FR15 (legacy clause, audit, reclassification), FR16 (flake, baseline), FR17 (version)

### Epic 7: Compose tabular classification and regression experiments without model code
Titanic, Bike Sharing, Store Sales and Digit Recognizer train, export and evaluate using DSio datasets, fitted standardization, MLP/head compositions, objectives and prediction outputs, with no local model, objective, dataset, normalizer or validator. Parity is recorded against the baseline.
**FRs covered:** FR1 (flat), FR2 (supervised, reconstruction), FR3 (stored items), FR7 (fixed), FR8 (non-dense outputs), FR9, FR16

### Epic 8: Compose variable-length and dense sequence experiments without model code
Essay Scoring, Parkinson FoG and ROGII train, export and evaluate using DSio padding collation, layout adapters, sequence encoders, masked dense objectives and dense outputs, with no local collator, model, objective or validator. Parity is recorded.
**FRs covered:** FR1 (sequence), FR2 (masked dense), FR4, FR8 (dense and ordinal), FR9, FR16

### Epic 9: Compose multimodal missing-modality experiments with balanced training
Both Child Mind consumers train using DSio window items, a modality-presence fusion composition, observed-only statistics, and class- and group-frequency weights. `child_mind` evaluates through `dsio.eval` with subgroup and ablation metrics. Parity is recorded.
**FRs covered:** FR1 (multimodal), FR3 (windows), FR6, FR7 (observed-only), FR10, FR16

### Epic 10: Run consumer tasks through DSio task functions
Every reference consumer's own Prefect tasks train and export by calling DSio spine functions, and downstream tasks use DSio arrays with `evaluate`/`predict`. Task bodies shrink to configuration plus DSio calls. Flow topology stays project-owned, and wiring falls by at least 50%. Parity is recorded.
**FRs covered:** FR11, FR12, FR13, FR16

### Epic 11: Release Warehouse 0.3.0 and prove a warehouse-first consumer
A consumer pins DSio 0.3.0. The release promotes evidence-backed blocks to stable with human approval, and lists every component's maturity and the remaining admission candidates. A new consumer built outside the repository from the wheel, README and catalog passes acceptance, and the success signal is measured.
**FRs covered:** FR14 (complete), FR15 (promotion), FR17, FR18

## Delivery rules for every story

These apply to every story below. They are the per-story form of NFR1–NFR15 and the admission process.

- **Test-first.** Consumer contract-tier tests (`tests/reference_flows/`, `tests/kaggle_portfolio/`) pass **without editing their expected values**, and the golden metrics from Story 6.6 hold. That is the per-story parity guard.
- **Declared computation changes.** A story that changes a consumer's computation declares the change and a tolerance **before** running. Examples include initialization order, random draws, normalization placement or loss reduction.
- **Home for new or reshaped blocks.** They live in `dsio.experimental.<domain>` until Epic 11 promotes them. Entry of each new block into `dsio.experimental` needs explicit human approval, per the admission doc.
- **Each block that gains a consumer:**
  - adds its evidence-register entry;
  - passes `require_admissible_component` with every consumer package name;
  - is reviewed adversarially before merge.
- **Measured deletion.** Report the non-ingestion line delta and the syntax-tree check result (Story 6.6 tooling) in the PR.

## Epic 6: Trust and discover what DSio ships

A consumer can see every DSio component, with its contract, maturity and evidence, in a CI-checked catalog. They can rely on a single-sourced version, a consistent batch-field vocabulary, and replay identity that does not flake. Every component sits where its evidence places it. Golden baselines exist for every consumer.

### Story 6.1: Single-source the version and clean the package surface

**Implements:** FR17

As a consumer pinning DSio,
I want the version I install, the version DSio reports and the version recorded in provenance to be one value,
So that evidence identifies exactly what produced it.

**Acceptance Criteria:**

**Given** the built wheel
**When** `dsio.__version__`, `importlib.metadata.version("dsio")` and the provenance-recorded DSio version are compared
**Then** all three are equal, and a test fails if they ever diverge
**And** `pyproject.toml` is the only place the version is written

**Given** the duplicate `ComponentError` in `config/components.py:32` and `model/chain.py:12`
**When** the package is inspected
**Then** it is defined once, and every former import site uses that definition

**Given** the repository
**When** `uv run ruff format --check .` runs in CI
**Then** it passes, with the 71 pre-existing differences normalized in one formatting-only commit

### Story 6.2: Make replay identity deterministic in CI

**Implements:** FR16

As a maintainer about to migrate every consumer,
I want the replay-identity flake in the aggregate CMI flow diagnosed and eliminated,
So that migration parity rests on trustworthy replays.

**Acceptance Criteria:**

**Given** a replay-identity mismatch in any replay test
**When** the test fails
**Then** the failure message includes a field-level diff of both provenance records, covering git commit, dirty patch digest, lockfile, DSio version, environment and component identities

**Given** the hypothesis that the first export dirties a clean checkout
**When** export writes are traced in a clean worktree
**Then** it is confirmed or refuted, with evidence recorded in the story
**And** if confirmed, no DSio export, evaluation or inference path writes inside the consumer's git worktree, and a test proves `git status --porcelain` is unchanged after a full flow

**Given** the definition of "resolved"
**When** the story closes
**Then** either the root cause is fixed with a regression test, or the aggregate CMI replay test has passed 20 consecutive times in a clean CI-equivalent worktree with provenance-diff capture in place

### Story 6.3: One batch-field vocabulary with consistent mask polarity

**Implements:** Additional Requirements (conventions); enables FR1–FR10

As a consumer composing blocks,
I want every DSio component to use the same batch-field names, one mask polarity and one weighting rule,
So that datasets, collators, objectives and evaluation compose without adapters.

**Acceptance Criteria:**

**Given** the published conventions (`docs/component-warehouse/conventions.md`, sourced from the spec's `conventions.md`)
**When** a consumer reads them
**Then** they define:
- the batch fields (with `x` as the only model input);
- `mask` (True = valid, objectives and evaluation only);
- `hidden`;
- declared validity and presence channels of `x`;
- the weighting rule;
- layouts and target conventions.

**Given** the masking strategies and `MaskedReconstruction`
**When** they emit a corruption tensor into a batch
**Then** it is named `hidden` (True = hidden), and no batch field named `mask` ever carries True = hidden
**And** a test guards that polarity at every DSio producer and consumer of `mask`

**Given** the self-supervised reference flow
**When** it replays after the rename
**Then** its contract-tier metrics are unchanged

### Story 6.4: Generate a CI-checked component catalog

**Implements:** FR14

As a consumer looking for a lego block,
I want a catalog generated from source and the evidence register that lists every component with its contract, maturity and evidence,
So that I can compose from what exists instead of re-implementing it.

**Acceptance Criteria:**

**Given** every name in `__all__` of `dsio.data.loading`, `dsio.model`, `dsio.train`, `dsio.inference` and the experimental domain packages, plus closed-dispatcher entries (split algorithms, `METRICS`)
**When** the generator (`tools/`, not shipped in the wheel) runs
**Then** it writes `docs/component-warehouse/catalog.md`, grouped by pipeline package, with:
- import path;
- maturity (from location);
- the Consumes, Produces, Parameters, Devices, Limitations and Example docstring sections;
- evidence joined from `docs/component-warehouse/evidence.yaml`.

**Given** the evidence register
**When** CI validates it
**Then** each entry names consumer and test paths that exist, whether the use is real or a fixture, its unrelated-use group, and optional MLflow run URIs (format-checked, not dereferenced)
**And** DSio source contains no consumer names (the admission audit passes over every changed module)

**Given** a public component missing a docstring section or evidence entry, or a committed catalog that differs from the regenerated one
**When** CI runs
**Then** it fails and names the component or the drifted entry

**Given** each Example section
**When** the test suite runs
**Then** every example executes as a doctest

**Given** `docs/component-warehouse/candidates.yaml`
**When** the catalog is generated
**Then** candidates render with consumer path, reason and use count
**And** stale docstrings found while writing sections are corrected. These include the `model/components.py:175-190` references, the `MaskedMSE` NaN-writer claim and the `masking.py` registry claim.

### Story 6.5: Place every component where its evidence puts it

**Implements:** FR15

As a consumer reading the catalog,
I want a component's location to tell me its maturity truthfully,
So that "stable" always means proven by two unrelated real uses.

**Acceptance Criteria:**

**Given** `docs/component-admission.md`
**When** the pre-1.0 legacy clause is added
**Then** it states that pre-existing components with no real use move to `dsio.experimental`, carry no compatibility promise, and are deleted at 1.0 if still unproven
**And** it states that uses are counted per public callable, and that the two consumers of one competition count as one use

**Given** the experimental package
**When** the layout is created
**Then** `dsio.experimental.data`, `.model`, `.train`, `.inference` and `.eval` exist, mirroring the stable domains
**And** an import-linter contract forbids stable packages from importing `dsio.experimental`

**Given** each component in the cohort's legacy list
**When** it is reclassified
**Then** it moves to its `dsio.experimental.<domain>` module, and tests and fixtures import the new path with no compatibility shim (pre-1.0)
**And** the self-supervised reference flow still passes

**Given** every public object under `dsio.experimental`
**When** the test suite runs
**Then** `require_admissible_component` passes for each one against every reference consumer package name, including `calibrate_training_execution`, `measure_phase` and `log_phase_evidence`

**Given** the catalog
**When** CI runs
**Then** it fails if any importable component in a stable package lists fewer than two unrelated real uses in the evidence register. Closed-dispatcher entries and spine functions are exempt.

### Story 6.6: Record golden baselines and the deletion measure

**Implements:** FR16

As a maintainer migrating consumers,
I want every consumer's pre-migration metrics and code measures recorded against a fixed commit,
So that each migration proves it preserved behavior and reduced consumer code.

**Acceptance Criteria:**

**Given** the baseline commit (the head of Epic 6 before this story's measurements)
**When** each of the 11 consumers runs its contract tier
**Then** its evaluation metrics are committed as golden values that its contract tests assert
**And** contract tests fail on any unexpected change

**Given** each Kaggle consumer whose dataset is under `~/Datasets`
**When** it runs its representative tier on the live MLflow server with a run-isolated `PREFECT_HOME`
**Then** its evaluation run URI, metrics, commit and hardware are recorded in `docs/component-warehouse/parity-baseline.md`

**Given** the consumer tree
**When** the measurement tool runs
**Then** it reports non-ingestion lines per consumer (every line outside `data.py`, `tasks/data.py` and ingestion-only modules such as `parkinsons_fog/scale/`), and the baseline is recorded
**And** a syntax-tree check lists every `nn.Module`/`Dataset` subclass and every locally defined value passed as an objective, collate function, normalizer or validator. This is the violation baseline that Story 11.3 must reduce to the candidates register.

## Epic 7: Compose tabular classification and regression experiments without model code

Titanic, Bike Sharing, Store Sales and Digit Recognizer train, export and evaluate using DSio datasets, fitted standardization, MLP/head compositions, objectives and prediction outputs, with no local model, objective, dataset, normalizer or validator. Parity is recorded against the baseline.

### Story 7.1: Map stored samples into training items

**Implements:** FR3

As a consumer with a staged store,
I want a DSio dataset that maps stored columns into `x`/`y`/`mask`/`sample_weight`, with dtype, layout and target transforms, configured and recorded like any other component,
So that I stop writing a Dataset class per project (cohort #1).

**Acceptance Criteria:**

**Given** a field-mapping ComponentConfig with:
- source columns per field;
- dtype;
- time-major or channel-first layout;
- optional target transform (offset, log1p or scale);
- optional labels

**When** the dataset reads a sample
**Then** it yields exactly the configured fields and verifies that sample's payload digest
**And** a tampered payload fails, and a declared field missing from the store fails with the field named

**Given** `DsioDataModule` and the dataset's ComponentConfig
**When** a training attempt records provenance
**Then** the dataset's full configuration is recorded, and a config parameter colliding with a runtime argument raises instead of overriding it

**Given** titanic, bike_sharing, store_sales and digit_recognizer, plus the supervised and self_supervised fixtures where the mapping fits
**When** they are migrated
**Then** their local sample Dataset classes and factories are deleted, and contract tiers match the golden metrics

### Story 7.2: Fit standardization on training-role samples

**Implements:** FR7

As a consumer normalizing inputs,
I want DSio to fit standardization statistics on the training role and inject them into a standardization slot, with the fitted values logged as evidence,
So that I cannot leak validation data into normalization (cohort #4a, #9a).

**Acceptance Criteria:**

**Given** a split manifest, fold and store
**When** the statistic fitter runs
**Then** it computes per-feature mean and standard deviation over training-role samples only (NaN-aware)
**And** it logs the values as an evidence artifact referenced from provenance

**Given** a validation-role sample whose values change
**When** the fitter reruns
**Then** the fitted statistics are identical (leakage test)

**Given** bike_sharing and digit_recognizer
**When** they are migrated
**Then** bike_sharing's `_scaler` and model buffers, and digit_recognizer's `ScalePixels`, are replaced by the fixed standardization slot (fitted or constant)
**And** contract tiers match the golden metrics

### Story 7.3: Compose MLP models from backbones and heads

**Implements:** FR1

As a consumer building a tabular or flat model,
I want to compose a preprocessor, an MLP backbone and a head (including bounded activations) into one model exposing `encode()`,
So that I never define an `nn.Module` for flat inputs (cohort #7, #8a, #8b, #14, #17).

**Acceptance Criteria:**

**Given** preprocessor, backbone and head component configurations
**When** the composition factory builds the model
**Then** it returns a native `nn.Module` whose `forward` and `encode` match the configured chain, with shape errors naming the offending stage

**Given** a non-negative regression target
**When** a softplus-bounded head is configured
**Then** outputs are non-negative without a consumer activation

**Given** titanic (linear head), bike_sharing (standardization, linear and bounded head), store_sales (MLP and bounded head), and digit_recognizer (classifier, plus the experimental autoencoder composition #17)
**When** they are migrated
**Then** their local model classes are deleted, and contract tiers match the golden metrics or a pre-declared tolerance where initialization order changes
**And** digit_recognizer keeps only its verified encoder handoff, listed in the candidates register

### Story 7.4: Train with DSio objectives

**Implements:** FR2

As a consumer choosing a loss,
I want DSio objectives over any native loss, with auxiliary metrics, target adaptation, stage restriction, native class weights and mean-1 sample weights,
So that I never write an `Objective` (cohort #18, #20).

**Acceptance Criteria:**

**Given** a supervised objective configured with a native loss and its native parameters (e.g. `weight=` class weights), target dtype/shape adaptation, and named auxiliary metrics
**When** it runs on a batch
**Then** it returns `loss` and each metric as a scalar
**And** a `[B,1]` target is never silently squeezed against a `[B]` prediction

**Given** sample weights with training-role mean 1
**When** the same samples are split into different micro-batch partitions
**Then** the optimized loss and logged weighted metrics are identical (micro-batch invariance test)
**And** per-micro-batch `/sum(w)` normalization is absent

**Given** a reconstruction objective
**When** the target is declared as the input
**Then** the loss is computed against `x` without a consumer objective

**Given** titanic, bike_sharing, store_sales and digit_recognizer (classification and reconstruction), and the supervised fixture
**When** they are migrated
**Then** every local objective is deleted, logged metric names are unchanged, and contract tiers match the golden metrics
**And** the self_supervised fixture may use the objective over legacy `NTXent`, or keep fixture code

### Story 7.5: Declare prediction outputs without writing validators

**Implements:** FR8

As a consumer exporting a predictor,
I want DSio binary, multiclass/ordinal and regression outputs with parameterized validators and MLflow signatures,
So that I never write a normalizer or validator (cohort #24, #25, #26).

**Acceptance Criteria:**

**Given** each output:
- binary threshold and probability;
- multiclass argmax and probability, with an optional ordinal label offset;
- regression with a declared inverse target transform and optional non-negativity

**When** its validator receives a violating prediction
**Then** it rejects each violation class with a named error: shape, finiteness, sign, range, simplex, threshold consistency and argmax consistency

**Given** each output inside a predictor
**When** exported, loaded through MLflow and used in `predict`
**Then** outputs are identical before and after the round trip, and the Logged Model signature declares the output fields

**Given** titanic, bike_sharing, store_sales and digit_recognizer, and the supervised fixture
**When** they are migrated
**Then** their local normalizers and validators are deleted
**And** `TensorOutput` and `validate_tensor_prediction` are removed once no consumer or fixture uses them
**And** the self_supervised fixture keeps its embedding-norm pair as fixture code

### Story 7.6: Build evaluation arrays from the training collation

**Implements:** FR9

As a consumer evaluating an exported predictor,
I want `dsio.data` to assemble evaluation inputs, targets and masks through the same dataset and collation used for training,
So that I stop re-implementing batching in NumPy (cohort #5).

**Acceptance Criteria:**

**Given** a dataset configuration, collation and role
**When** evaluation arrays are assembled
**Then** they equal the training collation's output for the same sample IDs, in order, with `sample_id` preserved
**And** `dsio.eval` still imports no pipeline layer (import contract unchanged)

**Given** titanic, bike_sharing, store_sales and digit_recognizer, and the supervised/self_supervised fixtures
**When** they are migrated
**Then** every `_arrays` and `evaluation_arrays` helper is deleted, and evaluation metrics match the golden metrics

### Story 7.7: Record tabular migration parity

**Implements:** FR16

As a maintainer,
I want representative-tier parity for the migrated tabular consumers recorded in MLflow,
So that Epic 7 proves the warehouse preserved behavior at real data scale.

**Acceptance Criteria:**

**Given** the parity baseline (Story 6.6)
**When** the migrated consumers rerun their representative tier on the recorded hardware class
**Then** each records a parity run tagged with its baseline run ID. Deltas are exact, or within each story's pre-declared tolerance, and are logged.
**And** each block these consumers use has the parity run URI in its evidence entry

## Epic 8: Compose variable-length and dense sequence experiments without model code

Essay Scoring, Parkinson FoG and ROGII train, export and evaluate using DSio padding collation, layout adapters, sequence encoders, masked dense objectives and dense outputs, with no local collator, model, objective or validator.

### Story 8.1: Pad variable-length batches with a DSio collator

**Implements:** FR4, FR9

As a consumer with ragged sequences,
I want a DSio collator that pads declared variable-length fields, stacks fixed fields and optionally emits a True-is-valid `mask`,
So that I never write a collate function (cohort #3).

**Acceptance Criteria:**

**Given** a collator ComponentConfig declaring variable-length fields with padding values, fixed fields, and whether to emit `mask`
**When** items of different lengths are collated
**Then** variable fields are padded to the batch maximum, `mask` (when emitted) is True exactly on real positions, fixed fields are stacked, and `sample_id` order is preserved

**Given** fields that must align (e.g. `x`, `y`, `mask`) but whose lengths differ
**When** they are collated
**Then** collation fails, naming the sample and fields

**Given** the depth constraint on #3
**When** the collator is reviewed
**Then** its configuration surface is shown to be materially smaller than the three implementations it replaces, or the story stops and records why the patterns do not share one block

**Given** essay_scoring, parkinsons_fog and rogii
**When** they are migrated
**Then** their local collate functions and `_targets_and_mask`/`_arrays` builders are deleted, with evaluation arrays built via Story 7.6 using this collator
**And** contract tiers match the golden metrics

### Story 8.2: Train dense sequence models with masked objectives

**Implements:** FR2

As a consumer predicting per timestep,
I want a DSio objective applying any elementwise loss only where the batch `mask` is True,
So that padded and ignored positions never affect training (cohort #19).

**Acceptance Criteria:**

**Given** a masked objective over BCE or MSE
**When** masked-out positions change value
**Then** the loss and metrics are unchanged
**And** an all-False mask raises a named error instead of producing NaN

**Given** parkinsons_fog (BCE over `[B,T,K]`) and rogii (MSE; its dataset maps the validity column to both a declared `x` channel and the `mask` field)
**When** they are migrated
**Then** their local objectives are deleted, and contract tiers match the golden metrics

### Story 8.3: Compose dense signal models with layout adapters and per-instance standardization

**Implements:** FR1

As a consumer of multichannel signals,
I want a dense per-timestep Conv1d encoder, a time-major ↔ channel-first layout adapter and per-instance standardization as composable blocks,
So that my signal model is configuration only (cohort #9b, #11, #13).

**Acceptance Criteria:**

**Given** a time-major `[B,T,C]` input
**When** it flows through the layout adapter and dense encoder
**Then** the output is `[B,T,K]`, with extent checks naming mismatched axes

**Given** per-instance standardization as deterministic Predictor preprocessing
**When** the predictor is exported and reloaded
**Then** training and inference apply identical standardization (round-trip test)
**And** the story pre-declares the parity tolerance for moving parkinsons_fog's NumPy normalization into the Predictor

**Given** parkinsons_fog
**When** it is migrated
**Then** `FogDetector` and both copies of `normalize_signal` (dataset and downstream) are deleted, and its contract tier is within the declared tolerance

### Story 8.4: Compose token-sequence models with masked pooling

**Implements:** FR1

As a consumer of token sequences,
I want a token embedding encoder with masked mean pooling that reads validity from a declared `x` channel,
So that padded tokens never influence the representation (cohort #12).

**Acceptance Criteria:**

**Given** padded token batches whose declared validity channel marks padding
**When** padding token values change
**Then** the pooled representation is unchanged

**Given** essay_scoring
**When** it is migrated
**Then** `EssayRegressor` is deleted and composed from the encoder plus a linear head, and its contract tier matches the golden metrics
**And** its tokenizer is listed in the candidates register

### Story 8.5: Compose residual models around an explicit baseline

**Implements:** FR1

As a consumer with a strong baseline,
I want a composition that adds a zero-initialized bounded residual to a declared baseline channel of `x`,
So that a new model can never start worse than the baseline (cohort #16).

**Acceptance Criteria:**

**Given** the residual composition at initialization
**When** it predicts
**Then** its output equals the baseline channel exactly

**Given** rogii
**When** it is migrated
**Then** `TvtRegressor` is deleted, its contract tier matches the golden metrics, and it still does not regress the last-value baseline

### Story 8.6: Declare dense and ordinal prediction outputs

**Implements:** FR8

As a consumer exporting sequence predictions,
I want the DSio outputs to cover dense per-timestep binary and regression shapes and ordinal label offsets,
So that no sequence consumer writes a validator (cohort #24, #25, #26).

**Acceptance Criteria:**

**Given** dense binary `[B,T,K]` and dense regression `[B,T]` outputs, and an ordinal output with a label offset
**When** validators receive violating predictions
**Then** each violation class is rejected by name, and the export round trip is identical

**Given** essay_scoring, parkinsons_fog and rogii
**When** they are migrated
**Then** their local normalizers and validators are deleted, and submissions are byte-identical at the contract tier

### Story 8.7: Record sequence migration parity

**Implements:** FR16

As a maintainer,
I want representative-tier parity for the migrated sequence consumers recorded in MLflow,
So that Epic 8 proves behavior preservation.

**Acceptance Criteria:**

**Given** the parity baseline
**When** essay_scoring, parkinsons_fog and rogii rerun their representative tier
**Then** each records a parity run meeting the exact or pre-declared tolerance rule, with deltas logged
**And** each block used has these parity runs in its evidence entry

## Epic 9: Compose multimodal missing-modality experiments with balanced training

Both Child Mind consumers train using DSio window items, a modality-presence fusion composition, observed-only statistics, and class- and group-frequency weights. `child_mind` evaluates through `dsio.eval` with subgroup and ablation metrics.

### Story 9.1: Map windows into training items with weights and groups

**Implements:** FR3

As a consumer training on windows,
I want the DSio window dataset to use the same field mapping as stored items, including per-window `sample_weight` and `group`,
So that I never write a window Dataset (cohort #2).

**Acceptance Criteria:**

**Given** a window index and field-mapping ComponentConfig
**When** windows are read
**Then** the configured fields are yielded with their `group` identity and digest verification, and the configuration is recorded in provenance

**Given** child_mind/sequence
**When** it is migrated
**Then** `CmiSequenceWindows` and `sequence_windows` are deleted, and its contract tier matches the golden metrics

### Story 9.2: Fit observed-only statistics for missing data

**Implements:** FR7

As a consumer with missing values,
I want DSio to fit and apply standardization over observed values only, both fitted and per-window,
So that missingness never corrupts normalization (cohort #4a, #9a, #9b).

**Acceptance Criteria:**

**Given** features with NaNs or a declared observed channel
**When** observed-only statistics are fitted on the training role
**Then** unobserved values are excluded, all-missing features raise a named error, and the leakage test holds

**Given** child_mind and child_mind/sequence
**When** they are migrated
**Then** `_statistics`, `_observed_statistics`, `_normalize_observed`, `_normalize_window` and the parameter-validation helpers (`_vector`, `_scale`, `_binary`/`_binary_mask`) are deleted, and contract tiers match the golden metrics

### Story 9.3: Balance training by class and group

**Implements:** FR6

As a consumer with imbalanced classes or participants,
I want DSio to fit class-frequency and group-frequency weights on the training role and apply them through the weighting rule,
So that balancing needs no local code (cohort #4b, #4c).

**Acceptance Criteria:**

**Given** training-role labels and group identities
**When** weights are fitted
**Then** class weights pass as native loss parameters, and group weights are normalized to mean 1 over the training role
**And** both are logged as evidence and depend only on training-role samples

**Given** child_mind (native class weights) and child_mind/sequence (class and participant weights)
**When** they are migrated
**Then** `_class_weights` and the dataset-side participant weighting are deleted
**And** both consumers' contract-tier metrics match the golden metrics exactly, and cmi-seq's micro-batch invariance test still passes

### Story 9.4: Compose multimodal fusion with modality-presence gating

**Implements:** FR1

As a consumer with optional modalities,
I want a fusion composition that encodes declared slices of `x` with configured encoders, gates by declared presence channels, and fuses into a head,
So that missing-modality models are configuration only (cohort #10, #15).

**Acceptance Criteria:**

**Given** a slice layout for modality inputs and presence channels
**When** a modality is absent for a sample
**Then** its contribution is gated out, and the prediction depends only on present modalities (test)

**Given** a pooled temporal Conv1d encoder for the sequence modality, behind a layout adapter
**When** it is configured in the fusion composition
**Then** `CmiSequenceClassifier`'s architecture is reproduced

**Given** child_mind and child_mind/sequence
**When** they are migrated
**Then** `CmiFusionClassifier` and `CmiSequenceClassifier` are deleted, and contract tiers match the golden metrics or a pre-declared tolerance where initialization order changes

### Story 9.5: Evaluate subgroups and modality ablation

**Implements:** FR10

As a consumer evaluating a multimodal model,
I want `dsio.eval` to report subgroup metrics and modality-ablation deltas over the arrays I supply,
So that my evaluation is configuration, not an 80-line task (cohort #29).

**Acceptance Criteria:**

**Given** consumer-supplied arrays, a subgroup definition (e.g. modality present or missing) and a declared modality slice to ablate
**When** evaluation runs
**Then** subgroup metrics and the ablated-minus-full delta are logged as native MLflow evidence, with ablated predictions persisted
**And** `dsio.eval` still imports no pipeline layer

**Given** child_mind
**When** it is migrated
**Then** its evaluation runs through `dsio.eval.evaluate`, `ablate_sensor` is deleted, and QWK and ablation deltas match the golden metrics

**Given** child_mind/sequence
**When** Epic 9 closes
**Then** its bounded-memory streaming evaluation is listed in the candidates register (roadmap: streaming evaluation)

### Story 9.6: Record multimodal migration parity

**Implements:** FR16

As a maintainer,
I want representative-tier parity for both Child Mind consumers recorded against the baseline,
So that Epic 9 proves behavior preservation.

**Acceptance Criteria:**

**Given** the parity baseline
**When** both consumers rerun their representative tier
**Then** each records a parity run meeting the exact or pre-declared tolerance rule, and each block used has these runs in its evidence entry

## Epic 10: Run consumer tasks through DSio task functions

Every reference consumer's own Prefect tasks train and export by calling DSio spine functions, and downstream tasks use DSio arrays with `evaluate`/`predict`. Task bodies shrink to configuration plus DSio calls, and flow topology stays project-owned.

### Story 10.1: Train inside a project task with one DSio fit function

**Implements:** FR11

As a consumer writing a training task,
I want one DSio function that runs the native Lightning fit for my constructed `DsioModule` and `DsioDataModule` and records its evidence,
So that my task body is configuration plus one call (cohort #22).

**Acceptance Criteria:**

**Given** a project-constructed `DsioModule`, `DsioDataModule`, a `TrainerConfig` and an attempt run
**When** the fit function runs inside a consumer `@task`
**Then** it calls native `Trainer.fit` once with the MLflow logger and capability logging, records provenance (including dataset and collator configurations), and stores the checkpoint as a digest-verified artifact whose reference it returns
**And** it takes no task, mode or model-family argument, chooses no hyperparameter, and owns no loop, callback policy, flow, task or retry

**Given** the spine's "universal trainer" non-goal
**When** the function is merged
**Then** an ADR defines a universal trainer as a DSio-owned loop or task-dispatching entry point, shows this function is neither, and amends the generic-spine spec and `CONTEXT.md` accordingly

**Given** titanic and bike_sharing
**When** their training tasks are migrated
**Then** each task body is configuration plus one call, and contract tiers match the golden metrics

### Story 10.2: Opt into execution calibration through the fit function

**Implements:** FR11, NFR11

As a consumer on new hardware,
I want the fit function to optionally run execution calibration before training, from my module and data-module factories,
So that hardware adaptation needs no consumer wiring.

**Acceptance Criteria:**

**Given** calibration enabled with a target effective batch and memory budget
**When** the fit function runs
**Then** only execution knobs are selected, every candidate and the selection are logged, and the effective batch is unchanged

**Given** parkinsons_fog and child_mind/sequence
**When** their training tasks are migrated
**Then** their duplicated calibration factories are deleted, and the selection is reproducible across two replays

### Story 10.3: Migrate the remaining training tasks

**Implements:** FR11

As a maintainer,
I want every remaining consumer's training task to use the fit function,
So that training wiring is removed everywhere.

**Acceptance Criteria:**

**Given** store_sales, digit_recognizer, essay_scoring, rogii, child_mind and the supervised/self_supervised fixtures
**When** their training tasks are migrated
**Then** each body is configuration plus DSio calls, contract tiers match the golden metrics, and digit_recognizer's verified encoder handoff remains a listed candidate

### Story 10.4: Export predictors with one DSio function

**Implements:** FR12

As a consumer exporting a model,
I want one DSio function that verifies checkpoint lineage, builds the Predictor and logs the Logged Model,
So that export tasks are configuration plus one call (cohort #23).

**Acceptance Criteria:**

**Given** a checkpoint reference and predictor configuration
**When** the export function runs
**Then** it enforces checkpoint lineage, builds the predictor, logs the Logged Model with signature and example, and returns its immutable URI

**Given** all 11 consumers' export tasks
**When** they are migrated
**Then** each export body is configuration plus one call, and exported model signatures are unchanged

### Story 10.5: Run downstream tasks over DSio arrays

**Implements:** FR13

As a consumer scoring a predictor,
I want my downstream tasks to call `dsio.data` array assembly with the existing `evaluate` and `predict`,
So that no consumer re-implements the evaluation glue.

**Acceptance Criteria:**

**Given** a model URI, dataset configuration, role and metrics
**When** a downstream task runs
**Then** it assembles arrays via Story 7.6 and calls `evaluate` or `predict`, preserving `sample_id`
**And** no new DSio evaluate function is added and the `dsio.eval` import contract holds

**Given** every consumer with downstream tasks, except child_mind/sequence's streaming evaluation (a listed candidate)
**When** migrated
**Then** downstream bodies are configuration plus DSio calls (submission formatting stays in the consumer), and outputs match the golden metrics

### Story 10.6: Record wiring parity and reduction

**Implements:** FR16 (CAP-7 success)

As a maintainer,
I want parity and the wiring reduction measured for all consumers,
So that CAP-7's at-least-50% target is proven.

**Acceptance Criteria:**

**Given** the parity baseline
**When** all consumers rerun their representative tier
**Then** parity runs meet the exact or pre-declared tolerance rule

**Given** the Story 6.6 measurement tool
**When** it runs
**Then** train/export wiring lines fall at least 50% from the baseline, and the per-consumer table is recorded in `docs/component-warehouse/parity-baseline.md`

## Epic 11: Release Warehouse 0.3.0 and prove a warehouse-first consumer

A consumer pins DSio 0.3.0. The release promotes evidence-backed blocks to stable with human approval and lists every component's maturity. A new consumer built outside the repository passes acceptance.

### Story 11.1: Promote evidence-backed blocks to stable

**Implements:** FR15

As a consumer depending on DSio,
I want every block with two unrelated real uses promoted to its stable package under the admission process,
So that the components I rely on carry a compatibility promise.

**Acceptance Criteria:**

**Given** each stable-candidate whose evidence register lists two unrelated real uses, including the calibration and telemetry functions (resolving the spec's open question)
**When** it is proposed for promotion
**Then** the PR provides compatibility tests (public import path, configuration, determinism, provenance), migration notes, a semver classification and a fresh adversarial review
**And** explicit human approval is recorded in the evidence register before merge, and no component is promoted without it

**Given** the promoted blocks
**When** consumers are updated
**Then** they import stable paths, contract tiers match the golden metrics, and the location-vs-evidence CI check from Story 6.5 passes

### Story 11.2: Prove a warehouse-first consumer

**Implements:** FR18

As a new DSio user,
I want to build a consumer for a new dataset outside the repository from the installed wheel, README and catalog,
So that the warehouse is proven sufficient, not merely reused.

**Acceptance Criteria:**

**Given** a public dataset that needs no competition-rule acceptance and is not in the portfolio, a clean directory outside the repository, and the built 0.3.0 wheel
**When** a new consumer is written using only the wheel, README and catalog
**Then** the Story 6.6 syntax-tree check finds no banned local definitions
**And** its code is limited to ingestion, feature/label mapping, configuration, flow topology and submission formatting

**Given** that consumer
**When** it runs a contract tier and a representative tier
**Then** it trains, exports, evaluates and infers with complete MLflow lineage, and its line counts per category are reported
**And** it is then added to `reference_projects/` as a regression consumer

### Story 11.3: Release DSio 0.3.0

**Implements:** FR14, FR17

As a consumer,
I want to pin DSio 0.3.0 with release notes that tell me every component's maturity and evidence,
So that I can adopt the warehouse knowingly.

**Acceptance Criteria:**

**Given** the release commit
**When** it is tagged `v0.3.0`
**Then** the version test passes, the catalog is current, and the distribution and consumer-flow contract CI is green on the tag

**Given** `docs/releases/0.3.0.md`
**When** a consumer reads it
**Then** it lists:
- every catalogued component with its maturity and evidence;
- migration notes for moved or removed components, including the legacy clause;
- the remaining candidates.

**Given** the success signal
**When** it is measured at the tag
**Then** the syntax-tree check over the 9 Kaggle consumers finds only candidates-register entries, non-ingestion lines are down at least 50% from the Story 6.6 baseline, and every migrated consumer has recorded parity
**And** all figures are in the release notes

## Validation (2026-09-30)

- **FR coverage:** each active FR (FR1–FR4, FR6–FR18) is implemented by at least one story (see each story's **Implements** tag). FR5 is withdrawn.
- **Starter template:** none; this is brownfield refactoring of `src/dsio/` and `reference_projects/`.
- **Dependencies:** every story depends only on earlier stories and epics. Stories 6.6, 7.7, 8.7, 9.6 and 10.6 record parity. Epic 10 needs nothing from Epics 8–9, and Story 10.5 uses Story 7.6.
- **File churn:** Epics 7–9 all extend `dsio.experimental` dataset, objective and output modules. The split is kept deliberately: each sequence or multimodal epic feeds back requirements the tabular blocks cannot anticipate (dense shapes, validity channels, presence gating), and each epic ships a working consumer cohort. Consumer `tasks/` files are touched incidentally by Epics 7–9 and substantively by Epic 10.
- **Human gates:**
  - admission entry approval for each new experimental block;
  - the legacy-clause amendment (Story 6.5);
  - stable-promotion approval (Story 11.1).
- **Adversarial fact-check (2026-09-30):**
  - Critical findings were applied: evidence register, weighting rule, single `x`, CAP-7/ADR, and the `dsio.eval` import contract.
  - Important findings were applied: per-callable counts, #6 dropped, collator depth constraint, streaming evaluation deferred, parity definition, provenance, catalog definition, syntax-tree success measure.
  - See the spec's `.decision-log.md`.
