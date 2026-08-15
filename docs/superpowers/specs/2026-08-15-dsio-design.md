# dsio — Design Spec

## Context

`dsio` is to be the foundational ML/DL experimentation system for the user's company and
personal work. Every other repo (`algua`, `pdm`, Kaggle work, future ventures) will
structure itself *around* dsio's artifacts and contracts — dsio does not adapt to them.

Three existing repos were surveyed as evidence, not as integration targets:

- **`acc_base` / FORGE** (`~/Projects/acc_base`, 51k LOC) — a *finished* masters thesis:
  Freezing-of-Gait detection from lower-back accelerometry, with seven SSL pipelines
  (MAE ×3 masking modes, SimCLR, I-JEPA/LeJEPA, iBOT, VICReg, TFC, patient-contrastive).
  It is the proof of concept dsio generalizes. **42.2M unlabeled patches vs 197k labeled
  (~100:1 by hours).**
- **`algua`** (`~/Projects/algua`, 40k LOC src / 61k LOC tests, zero TODO/FIXME) — sets the
  engineering bar. Its correctness-is-structural patterns are the model for dsio's core.
- **`kaggler`** (`~/Projects/kaggler`) — a ~660 LOC library that is effectively a
  proto-dsio: fold source-of-truth, experiment ledger, noise-floor verdicts, diagnostics.

The problems dsio exists to solve, all observed directly in those repos:

1. **Run identity is unmodeled.** FORGE recovers which run produced a paper number via four
   fallback log-filename globs per experiment (`scripts/analysis/aggregate_ssl_results.py`).
2. **Windowing is materialized.** 29 Zarr stores, 229 GB, because every (window, stride,
   labeling-policy) combination is a physical copy. Two of the largest are manual backups.
3. **Entrypoint duplication.** 8 near-identical `scripts/train/*.py` differing only in the
   pipeline class and W&B tags.
4. **Orchestration in bash.** 70+ shell scripts including `probe_loop.sh`, a polling daemon
   with a `.probed_checkpoints` resume file — a job scheduler written in bash.
5. **Config explosion.** 340 YAMLs despite a *written* anti-explosion policy, including
   `adamw_differential_lr_1e{5,6,7}.yaml`. The policy needed a system, not a rule.
6. **Validation too late.** Every FORGE leaf schema was `extra="allow"` with a bare
   `_target_: str`, so errors surfaced inside `hydra.utils.instantiate` after data loading.

**Intended outcome:** a forkable foundation where a new project reaches its first tracked,
reproducible, leakage-checked experiment in under an hour, and where every run can be
reconstructed months later without archaeology.

---

## Scope

**In:** data ingestion → staged/cached processing → splits → training (supervised + SSL) →
evaluation → model selection under multiplicity → registered model artifact.

**Out:** online serving, endpoints, drift/production monitoring. dsio's railway ends at a
versioned, reproducible model artifact plus its evaluation report. Batch inference is
included (a scoring run is tracked exactly like a training run) because a backtest or PdM
scoring pass that can't be reproduced is worth nothing.

**Modalities v1:** time series / forecasting, tabular (incl. sensor→features), NLP/text.
**Computer vision is explicitly out of v1** — but the data layer stays format-pluggable so
it can be added without redesign.

**Compute:** local-first (workstation, single GPU). Every path goes through `fsspec`, so
`s3://` works the day it's needed. No cloud job launcher until there is a cluster to launch
onto. Ship a `Dockerfile` for parity.

---

## Locked decisions

| Area | Decision |
|---|---|
| Shape | Forkable git repo. Hard `src/dsio/` ↔ `src/<project>/` boundary; forks keep an `upstream` remote and merge from it. Core fixes go upstream as PRs. |
| Config | Typed Pydantic objects composed in **Python**. No Hydra. YAML is a *recorded output*, never an authored input. |
| Sweeps | Native resumable content-addressed **job matrix** + **Optuna** for adaptive search. |
| Tracking | **dsio owns the authoritative Run ledger** (local, append-only, file-based). MLflow / W&B are optional sinks behind a Protocol. |
| Data | **One canonical store per corpus + lazy windowed views.** Materialize only what is expensive *and* deterministic. |
| Repro | Never block on a dirty tree. Capture enough that any run is reconstructible. Registry promotion requires a clean tree. |
| Abstraction | Thin spine. dsio owns config/cache/splits/tracking/eval; models stay idiomatic (real sklearn `Pipeline`, real `LightningModule`, real Nixtla objects). No universal `Model` wrapper. |
| Rigor | `src/dsio/` held to algua's bar (import-linter walls, named test per invariant, no swallowed exceptions). `src/<project>/` gets ruff only. |
| Stack | Python ≥3.12, `uv`, torch + Lightning, sklearn + XGBoost, Nixtla, Optuna, Pydantic v2, Polars, PyArrow. |

---

## Architecture

```
src/dsio/
  config/      typed Pydantic config objects, registry, CLI override parsing
  data/        stores, views, stage cache, manifests, remotes
  splits/      leakage-safe splitters + split manifests
  runs/        Run ledger, provenance capture, reproduce.sh emission
  train/       per-modality runners (tabular / forecast / torch)
  ssl/         pretext tasks, online probe, encoder handoff
  eval/        metrics, diagnostics, selection-under-multiplicity
  registry/    model artifacts with pinned refs + provenance digests
  matrix/      resumable job matrix + Optuna search
  cli/         Typer app, JSON envelope, output projection
  tracking/    ExperimentTracker Protocol + MLflow/W&B sinks
```

Dependency direction is strictly downward; `config`, `runs`, and `eval` are leaves that
import nothing else from dsio. Enforced by import-linter, not convention.

---

## Subsystems

### 1. Config (`src/dsio/config/`)

Structure lives in Python; YAML/CLI override **leaf values only**.

```python
@register_backbone("vit1d")
class ViT1d(BackboneConfig):
    model_config = ConfigDict(frozen=True, extra="forbid")
    dim: int = 384
    depth: int = 12

def mae_pretrain(backbone: str = "vit1d", lr: float = 1e-4, ctx: int = 500) -> RunConfig:
    return RunConfig(
        data=WindowedView(store="defog", length=ctx, stride=200),
        model=MAE(backbone=BACKBONES[backbone](), mask_ratio=0.75),
        optim=AdamW(lr=lr),
    )
```

Non-negotiables:

- **`extra="forbid"` and `frozen=True` on every schema, leaves included.** FORGE's
  `extra="allow"` leaves are the specific mistake being corrected.
- **Component selection goes through a decorator registry**, not reflective import of a
  `_target_` string. Unknown key → parse-time error with a did-you-mean suggestion.
- **Pre-flight the whole run graph** — resolve, validate, and instantiate-check every
  component — *before reading a single byte of data*.
- CLI overrides via `tyro` (types → CLI, no hand-written parser).
- Every run writes `config.resolved.yaml` + its sha256 into the run record. Reproduction is
  `RunConfig.model_validate(yaml)`.
- **Variants are arguments, never files.** There must be no path by which `lr=1e-6` becomes
  a checked-in file.

Built-in test: every registered preset constructs and validates. This is the cheapest guard
against config rot and FORGE already proved its value
(`tests/configs/test_experiment_configs.py`).

### 2. Data layer (`src/dsio/data/`)

Three tiers, deliberately different mechanisms:

**Canonical stores** — each corpus stored **once** as continuous signal.

```
stores/kaggle_defog/
  signal/     [T, C] chunked            <- Zarr v3 (default for continuous signal)
  labels/     [T]
  sessions/   sessions.parquet          <- id, start, end, patient_id, protocol
  stats/      per-session/patient/protocol mean, std, median, mad
  manifest.yaml                          <- content hashes, as_of, provider, row counts
```

Backend is pluggable per stage: Zarr for continuous multi-channel signal, Parquet for
tabular and metadata, memory-mapped Arrow or flat binary for tokenized text. Default for
signal is Zarr — proven at 229 GB in FORGE. (See Open Questions on Lance.)

**Views** — a window spec produces an **index**, not a copy.

```
views/len500_stride200_anyfog.idx.parquet   # megabytes
  -> start_offset, session_id, label, purity, patient_id
```

Carry forward FORGE's `fog_stride_len` idea (`utils/data.py::get_blocks`): a denser stride
applied only over rare-positive regions, so class imbalance is addressed at index time.
Labeling policy (`any` / majority / continuous ratio) is a view parameter.

A window-config change costs seconds and megabytes. FORGE's 229 GB collapses to roughly
raw-corpus size.

**Stage cache** — content-addressed, disposable, for everything downstream.

```python
@stage("features", version=3)
def build_features(cfg, upstream): ...

key = sha256("features" | version | canonical_json(cfg.features) | upstream.key)[:12]
```

Rule: **materialize what is expensive and deterministic; index what is cheap and
combinatorial.** Tokenization and embedding extraction get cached; windowing and shuffling
never do.

Cache invalidation is by explicit `version=` bump, *plus* a stored source fingerprint that
emits a loud warning on drift without silently discarding work. This avoids both failure
modes: reformat-nukes-a-40-minute-stage, and edit-logic-get-stale-results. FORGE has three
separate hand-rolled caches (sampler weights with a manual `CACHE_VERSION = 4`, eval
predictions invalidated by checking whether a column exists, embeddings) — this replaces
all three.

**Versioning/remotes:** the manifest is the interface; remotes are pluggable via `fsspec` —
local dir, S3/GCS, and **HuggingFace Hub + Xet** (Xet's content-defined chunking gives real
dedup and is the best-funded content-addressed storage in ML). No DVC: it is in maintenance
mode under lakeFS (acquired 2025-11-18) and its pipeline half would sit unused beside the
stage cache.

Port from `algua/data/{store,manifest,files,staging,verify}.py`: atomic staging via
`os.replace`, fsync-to-directory durability, flock-serialized manifest append, torn-write
recovery, and full read-back `verify`.

### 3. Splits (`src/dsio/splits/`)

Splits are **derived from a declarative spec and cached as a manifest**, never hand-written.
FORGE has 145 committed split YAMLs (family × context × fold); that becomes:

```python
GroupedKFold(k=3, group_by="patient_id", stratify_by="fog_count", seed=42)
PurgedWalkForward(windows=5, holdout_frac=0.2, embargo="max(feature_lookback, decision_lag)")
```

Requirements:

- Group-wise splitting (by patient / machine / well / symbol) is the **default**, not an
  option. Sensor and clinical data are always grouped.
- **Assert the splits are mutually disjoint.** FORGE's `SplitsConfig` validates duplicates
  *within* each list but never across them — cross-split patient overlap would pass silently.
- Purge + embargo for temporal splits, with the embargo derived from a *declared* feature
  lookback (port `algua/backtest/walkforward.py::_segment_bounds`).
- The resolved assignment is hashed into the run record, and reproducible in a different
  execution environment. `kaggler/cv.py`'s doctrine holds: **the fold assignment is the
  single source of truth; comparing runs on different folds is meaningless.** `templates/train.py`
  documents this exact hazard for remote Kaggle kernels.
- Fit/transform split enforced structurally: `dsio.features.fit` is unreachable from serve
  lanes via import-linter (port algua's `features/scaling{,_fit}.py` pattern). Normalization
  stats come from the train fold only.

### 4. Runs & provenance (`src/dsio/runs/`)

```
runs/<run_id>/
  run.json              config_hash, git.sha, git.dirty, env.lock_sha,
                        data.snapshot_ids, split_manifest_sha, seeds{python,numpy,torch,cuda},
                        env{python,torch,cuda,driver,hostname,gpu}
  config.resolved.yaml
  git.patch             (artifact; present when tree was dirty)
  metrics.jsonl
  artifacts/            preds.parquet, oof.parquet, metrics.json, params.json
  reproduce.sh
```

- Dirty runs are **allowed and tagged**, never blocked. Because the diff is stored, a dirty
  run is still exactly reconstructible.
- Port `algua/backtest/stamps.py`: `code_hash` is git HEAD, or
  `HEAD-dirty-<sha256 of status+diff+untracked>`, and returns `None` rather than a
  confident-but-wrong stamp when git is unavailable.
- `dsio reproduce <run_id>` rebuilds env + data + rerun.
- **Registry promotion is the gate**: only clean-tree runs may be promoted.
- Tracking sinks sit behind `ExperimentTracker` (Protocol ported from
  `algua/tracking/mlflow_tracker.py`): lazy import, finite-numeric filtering, nested
  parent/child runs for sweeps.

### 5. Training runners (`src/dsio/train/`)

**One entrypoint.** `dsio run <preset>` dispatches on `cfg.task.kind`. This deletes FORGE's
8 near-duplicate `scripts/train/*.py`.

Adapters, not wrappers — each runner uses its framework idiomatically:

- `tabular` — real sklearn `Pipeline` as the leakage boundary; XGBoost default (the only
  GBDT still shipping regularly; LightGBM went 17 months between releases, CatBoost 10).
- `forecast` — Nixtla `cross_validation` (best rolling-origin story) / sktime for nested CV.
- `torch` — Lightning. Port FORGE's best pattern wholesale (`pipeline/base.py:81-106`):
  a fixed component chain `preprocessors → signal_augmentor → transform → spectral_augmentor
  → backbone → head`, loss delegated, with always-present/maybe-present invariants declared
  and enforced. Plus `_common_step(batch, stage, compute_metrics, include_x)` to kill
  train/val/test triplication, and typed `BatchLogData` schemas with a centralized
  `detach_cpu()`.

Fold loop is **framework code, not user code** — `for fold: fit → predict → accumulate →
score → log` with a fixed artifact contract. `kaggler` re-implements this by hand in four
scripts; that is the thing being deleted.

Hard rules learned from FORGE bugs:

- **No swallowed exceptions.** FORGE's `instantiate_callbacks` catches everything and only
  warns — a misconfigured `ModelCheckpoint` silently disables checkpointing for a multi-hour
  run.
- **Sanitize metric names in filename templates.** `ap{metrics/val_ap:.3f}` created six
  stray `checkpoints/checkpoint-epoch=NN-metrics/` directories.
- **Checkpoints must close over their lineage.** A FORGE classification checkpoint silently
  reloads its MAE encoder from a hardcoded path and fails on a fresh clone. Model refs are
  pinned (name + version + digest), never "latest".

### 6. SSL (`src/dsio/ssl/`)

First-class, not an add-on. The primary workflow is
**pretrain on a large unlabeled corpus → probe/finetune on a small labeled corpus →
evaluate on a third held-out cohort**, with the three corpora having different schemas and
split logic.

- Pretext tasks as composable losses + heads over the shared component chain: masked
  reconstruction (temporal / frequency / patch / causal), contrastive (SimCLR-style),
  JEPA-style latent prediction, VICReg. Adding one must not require touching 7 places
  (FORGE's documented change-amplification cost).
- **Online linear probe as a native concept**, not a subprocess. FORGE's
  `DownstreamTaskCallback` shells out to `train_classification.py` via subprocess/SBATCH
  with a `ThreadPoolExecutor`; `probe_loop.sh` polls a checkpoint dir every 15 minutes.
  Both become `dsio matrix` cells with a done-ledger.
- Embeddings are a first-class cached artifact — expensive, deterministic, reusable, and the
  ideal stage-cache candidate.
- **Label-budget curves with arm-invariant subsets.** Port FORGE's
  `data/dataset/base.py:249-311` — seeded, stratified subsampling such that the *same* K
  patients are chosen across probe/random/scratch arms. This is real experimental rigor and
  belongs in the framework.

### 7. Evaluation & selection (`src/dsio/eval/`)

- Fixed artifact contract per run: out-of-fold predictions, test predictions, metrics,
  params, fold scores. Both `kaggler` and FORGE state this contract in prose and then
  re-implement it by hand.
- **Baseline-relative verdicts filtered by the fold-spread noise floor** — port
  `kaggler/self_model.py::verdict`. This is the single most transferable piece of code in
  any of the surveyed repos.
- Diagnostics as one-call functions (from `kaggler/diagnostics.py`): metric noise floor vs
  evaluation-set size, OOF correlation/saturation, adversarial validation for drift,
  permutation-importance pruning validated on fixed folds.
- **Selection under multiplicity** (port `algua/research/`): Deflated Sharpe Ratio, PBO via
  CSCV, stationary block bootstrap, effective sample size under serial dependence. Applies
  directly to "I ran 200 SSL ablations, which wins are real?"
- **Pluggable evaluation protocol.** Some tasks are not `metric(y_true, y_pred)` over rows:
  Rogii replays "predict the masked suffix per well, pool RMSE"; FORGE has episode-level
  clinical metrics and ICC; pdm scores "did the ETA land before the real crossing". This
  needs a first-class task-replay seam.
- Imbalanced-classification defaults: AP/AUPRC over accuracy, and pooled-across-folds rather
  than averaged per-fold.

### 8. Registry (`src/dsio/registry/`)

Port `algua/models/registry.py` + `contracts/model_types.py` — deserialization-agnostic
(stores raw bytes, no framework dependency), append-only manifest authoritative for valid
versions, per-name flock, fail-closed on digest mismatch / duplicate versions / missing
artifact / torn writes, and version allocation reserved off *both* the directory listing and
the manifest so a crash can't reuse a number.

`provenance_digest` commits to `{digest, training_snapshot_id, training_as_of, code_hash,
hyperparameters, seed, eval_report}` canonically.

### 9. Matrix & search (`src/dsio/matrix/`)

```bash
dsio matrix mae_probe --fold=0,1,2 --ctx=200,500,1000 \
                      --ckpt=glob:checkpoints/mae/*/epoch*.ckpt
dsio search mae_pretrain --n-trials=50 --optuna lr=loguniform(1e-5,1e-3)
```

- Job identity = sha256 of the resolved config. Output path derives from it, so re-running a
  completed cell is a **no-op** and a crash resumes exactly where it stopped.
- Both matrix and search emit ordinary Runs into the same ledger.
- Optuna is driven **directly**. The `hydra-optuna-sweeper` plugin is unusable regardless:
  stable 1.2.0 is from 2022 and pins `optuna<3.0`.

### 10. CLI (`src/dsio/cli/`)

Typer app. Port from `algua/cli/`:

- **JSON envelope on every command** — `{ok, error, code, retryable}` — with even
  argument-parse errors rendered as JSON (Click in `standalone_mode=False`).
- **Output projection from day one.** algua had to retrofit `--summary` because full JSON
  overwhelmed agent context. Every command that can emit per-fold/per-combo detail ships
  both a full and a projected view.
- Command modules are mutually independent (no sibling imports); composition happens at the
  root.

---

## Repo layout

```
dsio/
  pyproject.toml          uv, py>=3.12, hatchling; [tool.uv.index] explicit=true for torch
  uv.lock
  Dockerfile              local/cloud parity; installs git (code_hash needs it)
  conf/                   optional YAML VALUE overrides only (no structure)
  src/dsio/               the spine — CI-enforced (see Rigor)
  src/project/            template project package — ruff only
  stores/  views/  runs/  cache/    gitignored; manifests committed
  tests/
  docs/adr/
  .github/workflows/ci.yml
```

Forks rename `src/project/` and keep `upstream` pointed at dsio. CI warns when `src/dsio/`
has local commits.

---

## Rigor (`src/dsio/` only)

CI: `uv sync --locked` → `pytest` → `ruff check` → `mypy` → `lint-imports`.

Import-linter contracts encoding *specific leaks*, each with a comment explaining why:

| Contract | Prevents |
|---|---|
| `dsio.data.future` unreachable from train/eval lanes | Lookahead / hindsight reads |
| `dsio.features.fit` unreachable from serve lanes | Train/serve skew |
| `config`, `runs`, `eval` import no other dsio module | Layering inversion |
| `cli.*_cmd` modules mutually independent | Command-surface drift |

Plus: `frozen=True` + fail-closed `__post_init__`/validators on value objects, a named test
per invariant, and **no bare `except`** — every swallowed exception in FORGE hid a real
failure.

`src/<project>/`: ruff only. Tests optional. Hack freely.

---

## Open questions (resolve during Phase 1)

1. **Canonical store format for continuous signal.** Default is Zarr v3 (proven at 229 GB
   in FORGE). **Lance** is worth a bake-off — it claims ~100× faster random access than
   Parquet *and* has zero-copy dataset versioning built into the format, which could
   collapse the store and versioning layers into one. The research pass on this died on a
   session API limit and was never completed. Decide with a benchmark on real FORGE data
   before committing, since the `StageIO` interface must not leak the choice either way.
2. **NLP scope.** Assumed: embeddings + small encoder finetunes, plus API-based LLM calls
   with prompt-as-config and response caching. Not assumed: 7B+ fine-tuning (which would
   force a cloud launcher into v1).
3. **Zarr v3 vs v2 locally.** An open upstream issue reports v3 5–7.5× slower than v2 for
   local small-slice indexing. `zarrs-python` (Rust codecs) is the likely fix; benchmark.

---

## Implementation phases

1. **Spine** — config + registry + CLI envelope + Run ledger + provenance stamps. Verifiable
   with a trivial sklearn model end to end.
2. **Data** — canonical stores, manifests, views, stage cache, fsspec remotes.
3. **Splits & eval** — grouped/purged splitters, split manifests, fold loop, artifact
   contract, noise-floor verdicts, diagnostics.
4. **Training runners** — tabular, forecast, torch/Lightning component chain.
5. **SSL** — pretext tasks, embedding cache, online probe, label-budget curves.
6. **Matrix & search** — resumable ledger, Optuna.
7. **Selection under multiplicity** — DSR, CSCV/PBO, block bootstrap.

Each phase ends with a working `dsio run` on a real dataset, not a stub.

---

## Verification

- **End-to-end smoke, every phase:** `dsio run <preset>` on a small real dataset produces a
  Run record whose `reproduce.sh` reruns to *identical* metrics. This is the headline test.
- **Config composition test (built-in):** every registered preset constructs, validates, and
  instantiate-checks. Runs in CI.
- **Determinism test:** same config + same seed → identical metrics; differing seed →
  differing metrics (guards against a seed that isn't actually wired through).
- **Cache correctness:** changing a stage's config invalidates exactly that stage and its
  descendants, and nothing else. Changing a *sibling* stage's config invalidates nothing.
- **Leakage tests, named per invariant:**
  - splits are mutually disjoint (the check FORGE lacks);
  - normalization stats fit on train fold only — assert a scaler's
    `fit_max_timestamp <= first decision index`;
  - `lint-imports` passes, proving the future/fit walls hold structurally.
- **Matrix resumability:** kill a matrix run mid-flight; re-invoking completes only the
  unfinished cells and produces byte-identical output for the finished ones.
- **Registry integrity:** corrupt an artifact on disk → load fails closed with a digest
  mismatch rather than returning a wrong model.
- **Fresh-clone reproduction:** clone into a clean directory, `uv sync --locked`,
  `dsio data pull`, `dsio reproduce <run_id>` — must work with no manual steps. This is the
  test FORGE failed (checkpoints depended on a hardcoded encoder path).
- **Round-trip config:** `RunConfig.model_validate(yaml.safe_load(config.resolved.yaml))`
  reconstructs an object equal to the original.
