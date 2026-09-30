# Brownfield state at `main` 59b7db6 (2026-09-30)

Facts downstream work relies on. They were verified by an adversarial fact-check. Line
references are to `src/dsio/` unless stated.

## Component mechanism (keep)

- `config/components.py`:
  - `ComponentConfig` (l.25) is `{"reference", "parameters"}`.
  - `resolve_component` (l.74) imports, constructs and type-checks. **Config parameters silently override runtime keyword arguments** (l.84); v1 turns that collision into an error.
  - `require_importable_component` (l.127) rejects lambdas, closures, `<locals>` and `__main__`.
- Closed dispatchers: `METRICS` (`eval/metrics.py:36`, a `config/registry.py` `Registry`) and the split algorithm set (`data/splits/models/manifest.py:16`). `generate()` refuses runtime registration.
- `DsioModule` (`model/module.py:57`) and `DsioDataModule` (`data/loading/module.py:33`) raise on subclassing.
- `Predictor` (`inference/predictor.py:57`) validates importability of the model, preprocessor, normalizer and validator.
- `DsioDataModule` takes **bare callables** for `dataset_factory` and `collate_fn` (`data/loading/module.py:46,55`). Consumers record only a reference string in provenance; parameters and fitted values are not recorded.
- `DsioDataModule` has no sampler hook. Shuffle uses a seeded `RandomSampler` (`data/loading/loaders.py:62`).
- **Single model input.** `predict_step` (`model/module.py:238`), Predictor input (`inference/predictor.py:223-232`) and PyFunc export (`inference/export.py:55-61`) accept only `x`. This is why essay and rogii pack validity into `x`, and cmi-seq flattens every modality into `x`. The CMI spec explicitly kept this ("no widened Predictor API").
- **Import contract.** `pyproject.toml` forbids `dsio.eval` from importing `dsio.data`, `.model`, `.train` or `.tracking` ("Project flows provide the arrays").

## Why the existing library goes unused

These components have zero Kaggle-consumer users:

- **Backbones, heads and compositions:** `MLP1d`, `Conv1dEncoder`, `EmbeddingEncoder`, the heads, `ComponentChain`, `LossObjective`, `export_encoder`.
- **Losses:** `CrossEntropy`, `bce_loss`, `mse_loss`, `MaskedMSE`, `VICReg`.
- **Transforms and augmentations:** the standardizers, `RandomScale`, the masks, `MaskedReconstruction`.
- **Datasets and callbacks:** `StoredSamples`, `WindowDataset`, `OnlineProbe`, `RankMeMonitor`.

The synthetic references use `NTXent`, `Jitter`, `TwoView`, `TensorOutput` and
`validate_tensor_prediction`. The concrete misfits:

- `bce_loss` and `mse_loss` return `<locals>` classes and `squeeze(-1)` predictions, which mismatches `[B,1]` targets.
- `LossObjective` cannot emit auxiliary metrics without a `diagnostics()` method. It also cannot use the input as the target, restrict to a stage, or adapt target dtype/shape.
- `StoredSamples` yields no `y` and has no dtype, layout, column mapping or target transform. Every non-CMI-sequence consumer wrote its own dataset and dropped `examples`, which skips the digest check (cmi-seq does use `examples`).
- `MaskedMSE` supports only a NaN-sentinel target, which `MaskedReconstruction` (an augmentation, `train/augmentation.py:33`) produces.
- `TensorOutput` and `validate_tensor_prediction` only rename a tensor and check finiteness.
- There is no pad collator, statistic fitter, weight fitter, or binary/multiclass/regression output.

## Weighting semantics in consumers today

- `child_mind` (`components.py:147`) passes `cross_entropy(weight=class_weights)` with the default mean reduction, which normalizes by the batch's target weights.
- `child_mind/sequence` (`sequence/components.py:189-201`) computes `(losses * weights).mean()` over participant weights that have a split-wide mean of 1 (`:41-45`). Its test proves micro-batch invariance.
- v1 keeps both behaviors reachable (conventions.md, weighting rule), so both consumers can reach exact parity.

## Consumer baseline

There are 11 consumers under `reference_projects/`: 9 Kaggle consumers plus the `supervised` and `self_supervised` fixtures. `examples/` holds only an untracked `__pycache__`.

| Area | Lines |
|---|---|
| Ingestion/domain (`data.py`, `tasks/data.py`, `parkinsons_fog/scale/`) | 3,577 |
| Model-side (`components.py`) | 1,529 |
| Train/export/evaluate wiring | 4,361 (60–70% templated) |
| Flow + init | ~1,100 |
| **Total** | **10,568** |

Story 6.6 re-records these at the baseline commit as the authoritative non-ingestion
baseline. Non-ingestion means every consumer line outside the ingestion files. It also adds
the syntax-tree check for banned local definitions.

Other facts behind the success signal:

- Each Kaggle consumer defines one prediction normalizer and one validator (9 each). The ssl fixture adds an embedding-norm pair.
- All 11 consumers have export task bodies.
- The duplication catalogue D1–D16 is mapped to v1 blocks in [component-cohort.md](component-cohort.md).

Cross-consumer imports show the gap:

- `self_supervised` imports `TimeMajorToChannelFirst` and `evaluation_arrays` from `supervised`.
- `child_mind/sequence` re-implements `child_mind`'s `_vector`/`_scale` helpers, and its `_binary` corresponds to `child_mind`'s `_binary_mask`.

## Parity baseline gap

Contract-tier tests log to a throwaway file-store MLflow (`tests/conftest.py:20-22`), so no
pre-migration runs persist. Story 6.6 commits golden contract-tier metrics per consumer and
records representative-tier runs, with their hardware, on the live MLflow server.

## Admission tooling

- Static audit: `experimental/admission/__init__.py:19` `audit_component`, `:56` `require_admissible_component`.
- The audit scans **every string in the source file, docstrings included**, for supplied consumer names (`admission/source.py:125` → `syntax/projects/consumers.py`). Evidence therefore lives outside `src/dsio`.
- Dependency allowlist: `admission/source.py:12-25`.
- Only `dsio.experimental.execution:calibrate_execution` is audited (`tests/experimental/test_execution_calibration.py:127`). `calibrate_training_execution` and the telemetry functions are not audited.
- Calibration is used by fog and cmi-seq. Telemetry is used by fog (`tasks/data.py`, `tasks/training.py`) and cmi-seq (`tasks/training.py`).
- No status or catalog file exists. Maturity is implied by package location only.

## Recorded decisions this spec supersedes

`docs/research/2026-09-25-kaggle-gap-portfolio.md` said consumers own "model choice, loss"
(l.360). It also said "Do not generalize the three project collators yet": their semantics
differ enough that a helper "would expose nearly as much interface as implementation"
(l.407). The owner's warehouse vision, stated later the same day, supersedes the ownership
guardrail. The collator caution becomes the depth constraint on block #3.

## Defects and hygiene to fix in v1

- The version is not single-sourced. `pyproject.toml:3` says `0.2.0`, while `src/dsio/__init__.py:3` says `0.1.0`. Provenance records the metadata version (`tracking/provenance.py:147`). No test checks agreement.
- Mask polarity conflicts. `model/masking.py` returns True = hidden, while batch `mask` and `dsio.eval` use True = valid.
- There is a replay-identity CI flake in the aggregate CMI flow. A post-merge `main` run of 59b7db6 showed the first exported model apparently dirtying a clean checkout, which changed downstream identities on replay. A clean-worktree reproduction and the rerun passed, and the root cause is unconfirmed.
- Stale docstrings:
  - `model/components.py:175-190` refers to `ssl_task.py`, `torch_task.py` and a misplaced `export_encoder`.
  - The `MaskedMSE` docstring names `WindowDataset` as the NaN writer.
  - `masking.py:5` mentions a decorator registry that does not exist.
- `ComponentError` is defined twice: `config/components.py:32` and `model/chain.py:12`.
- `ruff format --check .` reports 71 files to reformat.
- The `__pycache__`-only directories under `src/dsio/` are untracked. Cleaning them is local housekeeping, not a repository change.

## Operational notes

- CUDA runs require a local consumer-side `torch` CUDA override (e.g. `2.13.0+cu130`). `uv sync` restores the pinned CPU wheel.
- Use a run-isolated `PREFECT_HOME`, because the shared `~/.prefect` database belongs to an incompatible Prefect release.
- Real datasets live under `~/Datasets` (symlinked into consumers).
- The MLflow stack is `docker compose up -d` (Postgres + MLflow on :5000).
