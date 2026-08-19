# dsio — Lean Skeleton Design

Status: proposed (2026-08-20)
Supersedes: `2026-08-15-dsio-design.md` (the template-and-spine design)

## Context

The spine at `lib/dsio/` reached 13,345 lines of source across 14 packages in four days,
executing all seven phases of the original spec without cutting any of them. Every ADR
justifies itself by citing FORGE or algua. The result is a design generalised from a sample
size of one: correct in its parts, and shaped by one project's requirements throughout.

Two properties caused most of the mass:

**Framework-neutrality.** A fold loop that could drive sklearn, Nixtla and Lightning
uniformly needs a dispatch protocol, its own metrics, its own tracker abstraction and its own
seeding — one reimplementation of a Lightning primitive per subsystem.

**A template that could not run itself.** Copier's Jinja machinery existed to rename the
package per project, which forced the repo to have no root `pyproject.toml` and made
`lib/dsio` read-only by convention, which in turn justified a broad CLI and wide `__init__`
façades so consumers could reach inside a package they were not allowed to edit.

This document specifies the lean skeleton: one framework, one package, cloned and owned.

## Goal

A skeleton of roughly 5,400 lines that a new project clones, adds components to, and pulls
improvements into with `git merge upstream/main`. It owns the parts every project rebuilds
badly — typed configuration, a memory-mapped store with lazy windowed views, leakage-safe
splits, provenance, and honest comparison — and nothing that Lightning, torchmetrics or
MLflow already provide.

The criterion applied throughout, in preference to any line count:

> **Does a second, unrelated project need this on day one?**

Where the answer is no, the code is cut and its shape recorded, so re-entry is a known
extension rather than a redesign.

---

## Locked decisions

### 1. Lightning-only

Torch and Lightning are the single first-class training path. No sklearn runner, no Nixtla,
no RL, no agentic training. Committing to one framework is what makes the skeleton lean:
Lightning already provides the mix-and-match contract that the framework-neutral layer was
rebuilding.

Consequence — every abstraction that existed to be framework-neutral is deleted:

| Deleted | Replaced by |
|---|---|
| `eval.loop.FitPredict` dispatch | `Trainer` |
| `eval/metrics.py` implementations | `torchmetrics` |
| `tracking/` (`ExperimentTracker`, `MultiTracker`, `MlflowTracker`) | Lightning `Logger` / `MLFlowLogger` |
| most of `runs/seeding.py` | `lightning.seed_everything(seed, workers=True)` |

We do **not** adopt `LightningCLI`: it is YAML-config-driven, which ADR 0001 rejects.

### 2. One package, cloned and owned

No Copier, no Jinja, no `.copier-answers.yml`. The workspace collapses to a single
distribution rooted at `src/dsio/` with one `pyproject.toml`, so the repo runs itself:
`uv sync && uv run pytest` is green in a fresh clone.

Projects clone, add components in the same tree, and pull improvements with
`git remote add upstream … && git merge upstream/main`. The root README's claim that "a
fork can only merge" was never about git — it was about paths failing to line up because
every project renamed its package. Fixing the package name removes the reason for the
template.

Cost accepted: the `project → dsio` direction was enforced by packaging (the spine could not
name a project). It is now an import-linter rule instead of a structural impossibility.

### 3. Storage: flat binary, memory-mapped

ADR 0005 stands. Re-tested against a wider scope (multivariate time series, NLP, vision) and
the conclusion strengthens rather than weakens:

- **MTS and NLP are the same access pattern** — random offset, contiguous slice. A token
  stream is a flat `uint16` array read in fixed blocks; this is what nanoGPT and Megatron-LM
  do, and `data/format.py` already uses Megatron's `.bin`/`.idx` split.
- **Vision** wants variable-length compressed blobs decoded in workers, which is memmap plus
  an offset table (the FFCV shape), not a dense array.
- **Zarr and Lance lose both ways** — 65–74× slower than memmap when data fits locally, and
  worse than sharded sequential streaming when it does not.

`DenseStore` (fixed dtype, fixed trailing shape, entity offset index) ships now and covers
MTS and NLP. `BlobStore` (variable-length records: `data.bin` + `offsets.npy`) is a known
~150-line extension for vision, added when a vision project exists — same memmap+offsets
concept, no change to `views.py` or the DataModule.

Zarr remains a read-only ingest path for corpora that already exist in it.

**Locality contract:** materialize-then-memmap, always. If a dataset ever exceeds local NVMe,
the answer is sharded sequential streaming behind the existing `backend=` seam
(`store.py`'s `__init__(self, path, *, backend="mmap")`), not a chunked random-access format.

### 4. The dataset owns the paradigm; the loss stays narrow

`loss(pred, target)` — the loss never sees the batch dict. What differs between paradigms is
what the dataset returns:

| Task | dataset returns |
|---|---|
| Classification / regression | `(x, label)` |
| MAE / denoising | `(x_masked, x_orig)` |
| MLM | `(tokens_masked, tokens_orig)`, `-100` at unmasked positions |
| Next-step | `(x[:-1], x[1:])` |
| Contrastive / joint-embedding | collate stacks V views into the batch dim; `target` is the pair index |

Rejected alternative: `loss(pred, batch)`. It lets every loss reach anything, so nothing
constrains what a loss depends on and no loss is testable without constructing a batch.

**Consequences.** There is exactly one `LightningModule` and no paradigm subclasses.
Masking moves from a model slot to the dataset, which deletes the augmentor slots and their
`self.training` guard — the "never augment during validation" property now holds *by
construction*, since the DataModule supplies a different dataset per stage. Augmentation runs
on CPU in DataLoader workers; revisit only if heavy vision augmentation makes that a
bottleneck.

### 5. Directories are technical kinds, never paradigms

`ssl/`, `agents/`, and any future `rl/` are category errors in a repo organised by technical
kind, and they are how a junk drawer regrows. `ssl/`'s 814 lines dissolve:

| Was | Lines | Becomes |
|---|---|---|
| `ssl/methods.py` | 214 | `model/heads.py` + `model/losses.py` (registry entries) |
| `ssl/masking.py` | 157 | `data`-side transform, applied by the dataset |
| `ssl/probe.py` | 194 | `train/callbacks.py` — a linear probe on frozen features is general |
| `ssl/budget.py` | 191 | cut (a sweep over one config axis) |
| `ssl/module.py` | 58 | cut (decision 4 removes the need) |

`model/` ships a deliberately small set of components; projects register their own. It is not
a model zoo.

### 6. The process boundary is the fold boundary

There is no `cross_validate`, no `CVReport`, no in-process fold loop. **`dsio run` trains one
config against one fold** and is linear top to bottom: build config → build data → build
module → `Trainer.fit` → predict → write artifacts → stamp provenance.

Cross-validation is running the entry point N times — from a shell loop, or an agent. This
gives three properties for free: resume (rerun the fold that died), parallelism (N folds
across N GPUs is N invocations), and native MLflow grouping (one experiment, N runs).

`RunConfig` gains `split` (which committed file) and `fold` (which index). That is the entire
interface between the loop and the run.

Preserved guarantees:
- **Cross-fold disjointness** moves from run time to load time — one split file holds all
  folds as an ordered list, so `SplitFile`'s validator checks it (strictly earlier than before).
- **The paired noise floor** still fires: each run records the split digest plus its fold
  index, and comparison checks correspondence across the two sets. This additionally catches
  "fold 2 of split A compared against fold 2 of split B".
- **Pooled out-of-fold metrics** become a ~30-line function reading N `predictions.npz` files.

### 7. MLflow is the source of truth; runs fail without it

**This supersedes ADR 0002** ("the ledger is authoritative; trackers are sinks"). Keeping
generated run data inside the source tree mixes code with artifacts, and a component a run
cannot write to is not a sink — it is a broken dependency. If MLflow is unavailable, the run
fails.

Everything ADR 0002 said MLflow could not hold is solved by logging it as an *artifact*
rather than a param:

| Gap | Fix |
|---|---|
| Uncommitted working-tree diff | `log_artifact(diff.patch)` |
| `reproduce.sh` | `log_artifact(reproduce.sh)` |
| Full nested config (params cap at 250 chars) | `log_artifact(config.resolved.yaml)`; flat params for search only |
| Config-hash identity | `set_tag("config_hash", …)` |

`runs/` is no longer a ledger; it is the provenance stamper — git rev, **dirty-diff capture**
(the thing that makes a dirty run reproducible, per ADR 0003), the reproduce script, and the
seed recorder. `artifacts/` becomes a thin policy layer over MLflow's Model Registry: compute
a digest on save, verify on load, **fail closed on mismatch** (which MLflow does not do), plus
`promotion_blockers` for ADR 0003's clean-tree gate.

Accepted risk: the Postgres volume is a single point of failure for experimental history.
Mitigation is decision 8's backup job, set up on day one rather than after the first loss.

### 8. Infrastructure: local, two services

Local only. Cloud training is explicitly deferred to its own design discussion.

```yaml
services:
  postgres:
    image: postgres:16
    environment: {POSTGRES_USER: mlflow, POSTGRES_PASSWORD: mlflow, POSTGRES_DB: mlflow}
    volumes: [pgdata:/var/lib/postgresql/data]
    healthcheck: {test: ["CMD-SHELL", "pg_isready -U mlflow"], interval: 5s, retries: 10}
    restart: unless-stopped
  mlflow:
    build: ./docker/mlflow          # ghcr.io/mlflow/mlflow + psycopg2-binary baked in
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri postgresql+psycopg2://mlflow:mlflow@postgres/mlflow
      --default-artifact-root /artifacts
    volumes: [mlartifacts:/artifacts]
    ports: ["5000:5000"]
    depends_on: {postgres: {condition: service_healthy}}
    restart: unless-stopped
volumes: {pgdata: , mlartifacts: }
```

MLflow runs as a daemon on port 5000 and is the only thing training talks to. The backend
store is invisible from the training side — `MLFlowLogger(tracking_uri="http://localhost:5000")`
is the entire integration, metrics stream from `self.log(...)` automatically, and
`logger.experiment` is an `MlflowClient` for artifacts.

Postgres rather than SQLite: decision 6 makes concurrent runs normal, and the intended
workload is **multiple agents launching experiments in parallel**, not one person running
folds in sequence. SQLite is single-writer and the MLflow server's own gunicorn workers
contend on it. Named volumes rather than bind mounts, so artifacts never land in the source
tree. The official `ghcr.io/mlflow/mlflow` image ships without `psycopg2` (mlflow#9513), so
it is baked into a small Dockerfile rather than pip-installed at container start.

### Backup

MLflow being the source of truth (decision 7) makes backup a correctness concern, not
hygiene. **Both halves must be captured:** Postgres holds run metadata and artifact *URIs*,
while the files live in the `mlartifacts` volume. Restoring the database alone yields an
index pointing at files that no longer exist.

The archive is **push-only and append-only**. Data moves local -> Drive and never back;
nothing in the system reads from the archive, and restore is a deliberate manual act.

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

# DB: small, full dump, timestamped, keep many
docker compose exec -T postgres pg_dump -U mlflow mlflow \
  | gzip > /var/backups/mlflow/db-$STAMP.sql.gz
rclone copy /var/backups/mlflow gdrive:dsio-backup/db

# Artifacts: incremental. MLflow artifacts are immutable once a run finishes,
# so nothing is ever re-uploaded.
rclone copy /var/lib/docker/volumes/mlartifacts/_data gdrive:dsio-backup/artifacts
```

Nightly, on a systemd timer.

**`rclone copy`, never `rclone sync`.** `sync` makes the destination match the source, so it
deletes from the archive whatever disappeared locally — which propagates a corrupted or wiped
volume straight into the backup, at exactly the moment the backup matters. `copy` never
deletes, and with timestamped dumps the archive is append-only by construction.

**Scope the remote to `drive.file`.** That OAuth scope grants access only to files the
application itself created, so the one-way property is enforced by the credential rather than
by discipline, and a mistaken flag cannot reach the rest of the Drive.

Google Drive rather than a second local disk because the failure that erases everything is
machine loss, which a local copy does not survive. `rclone crypt` if the corpus is clinical:
subject-identified accelerometer data should not sit unencrypted in consumer cloud storage.

**Keep `MLFlowLogger(log_model=False)`** — the default. Not for quota, but because per-epoch
checkpoints in the artifact store make the registry meaningless: the clean-tree gate exists
to mark a model as *kept*, and if every epoch's weights land there anyway, promotion
distinguishes nothing. Candidates are transient; promoted models are artifacts.

Accepted imprecision: `pg_dump` is atomic, but the artifact copy is not taken at the same
instant, so a run writing during the backup can straddle the two. At nightly cadence the
exposure is one in-flight run, which is not worth engineering away.

This is infrastructure. It adds no lines to the spine.

**Local GPU note:** the target machine is an RTX 5070 Ti — Blackwell, `sm_120`. torch 2.6 has
no kernels for it. Floor the pin at **torch ≥ 2.7 with the cu128 index**, or the first local
run fails with "no kernel image is available for execution on the device". 16 GB VRAM makes
bf16 autocast, gradient accumulation and gradient checkpointing default concerns in the
Lightning config rather than reactions to an OOM.

### 9. One CLI command

A command exists only if it is needed *before* there is a Python session. That leaves
`dsio run` (launching training, listing presets when called bare). Everything else —
inspecting a store, checking a split, reading an artifact — happens where Python is already
available, in a repo you own.

`envelope.py` survives, shrunk: the `{ok, error, code, retryable}` shape lets an automated
caller distinguish "your input was wrong" from "try again", which matters when an agent drives
the loop.

Cut commands do not remove their behaviour: integrity-on-load lives in the loader, and the
clean-tree gate lives in `artifacts/`, regardless of how either is invoked.

---

## Layout

```
src/dsio/
  contracts.py     190   DsioModel (extra="forbid"), canonical_json, atomic writes
  config/          461   RunConfig, @preset, component registries, overrides
  runs/           ~215   provenance stamp, dirty-diff capture, reproduce script, seed record
  artifacts/       ~80   digest on save, fail-closed load, promotion blockers
  eval/           ~550   single-run artifact contract, torchmetrics registry, verdict
  data/         ~1,560   DenseStore, lazy views, .bin/.idx format, mmap reader,
                         Examples protocol, skip-if-exists staging
  splits/         ~485   SplitFile, resolve→positions, fold_at, purged walk-forward
  model/          ~890   slot registries, one LightningModule chain, components
  dataset/        ~240   torch Dataset + DataModule
  train/          ~495   LightningModule tasks, runner, callbacks
  cli/            ~204   dsio run, JSON envelope
```

**≈5,400 lines** (estimates; the real number lands when the code moves), 10 packages and one
flat module, down from 13,345 across 14 packages.

### Dependency layers

```
contracts.py                                  imports nothing
config/  runs/  artifacts/  eval/             import contracts only; mutually independent
data/                                         imports contracts
splits/  model/  dataset/                     import data (and config, runs.seeding)
train/  callbacks                             composition
cli/                                          composition root
```

Import contracts retained: foundation modules mutually independent; `eval` forbidden from
importing `data`, `splits`, `train`, `runs`, `artifacts`, `cli`.

**Cycle to fix:** `data/adapters.py:183` currently does a function-local
`from dsio.splits.temporal import window_times` to dodge a circular import — `data → splits`
and `splits → data`. `window_times` computes a window's time span from a `WindowIndex` and a
store, which is view knowledge. Move it to `data/views.py`; the cycle disappears and the local
import becomes a module-level one.

### Comparison and the import contract

`eval` keeps the pure statistics — `noise_floor`, `paired_noise_floor`, `_classify`,
`autocorrelation`, `effective_sample_size` — as functions over plain arrays. Gathering runs
(querying an MLflow experiment, downloading prediction artifacts) lives outside `eval`, which
preserves the forbidden-import contract and is the right seam regardless.

---

## What is cut, and why

| Cut | Lines | Reason |
|---|---|---|
| `agents/` | 1,042 | Out of scope under Lightning-only; an eval harness, not a training path |
| `train/tabular.py` | 270 | Lightning-only |
| `ssl/` as a directory | 814 | Paradigm folder; 371 lines survive as components, 194 as a callback |
| `data/cache.py` | 478 | Eight strategy classes and two Protocols for "hash the config, write an npz, skip if present" |
| `data/remote.py` | 289 | Justified only by cloud training, now deferred |
| `eval/multiplicity.py` + `select.py` | 615 | DSR/PBO need hundreds of trials; DL runs give tens. ~80 lines of ESS kept (overlapping windows genuinely inflate N) |
| `eval/loop.py` | 176 | Decision 6 |
| `splits/stratify.py` | 406 | A modelling choice, not a skeleton concern |
| `splits/generate.py` + `SplitSpec` | 341 | Reimplements sklearn's `GroupKFold`/`StratifiedGroupKFold`/`LeaveOneGroupOut`; generation is an offline script that may depend on anything |
| `matrix/` | 700 | Decision 6 makes resume free; Optuna/MLflow sweeps handle search externally |
| `tracking/` | 182 | Lightning `Logger` |
| `eval/metrics.py` implementations | ~200 | torchmetrics, with a ~60-line name registry kept |
| CLI commands (26 → 1) | ~1,250 | Decision 9 |
| `__init__.py` façades | ~250 | 41 and 30 re-exported symbols existed for a library you could not edit |
| `runs/` ledger machinery | ~270 | Decision 7 |
| `artifacts/` storage + listing | ~180 | Decision 7 |
| `readers.FileReader`, `adapters.KeyedExamples`, `views.WindowView` | ~100 | Second implementations for cases decided otherwise, or duplicated by the torch Dataset |
| Copier/Jinja machinery | 6 files | Decision 2 |

### Kept, and why

`temporal.py` (252) is the deliberate exception to the day-one criterion. Purged, embargoed
walk-forward has no sklearn equivalent, and the failure it prevents is silent: a training
window whose *label horizon* resolves inside the test period has seen the future, and serial
correlation leaks backward across the boundary too. Both inflate results while looking normal.
The distinction claimed: purging is a property of the split file's *correctness*, which the
skeleton owns; stratification is a modelling choice, which it does not.

`eval/contract.py` (~180) keeps a fixed artifact shape so two runs are comparable without
reading two scripts, and so a metric nobody thought to log can be recomputed without
retraining — which MLflow's flat metric store cannot give back.

`views.py` keeps entity and group on *every window*: overlapping windows straddling a split
boundary put near-identical rows in train and test simultaneously (Kapoor & Narayanan L1.4
and L3.2), and that is invisible if the index is only offsets.

`config/` is untouched. Structure in Python, YAML as recorded output, variants as function
arguments, is the highest-value decision in the repository.

### Known extension points

Recorded so re-entry is planned rather than improvised:

- **`BlobStore`** (~150) — variable-length records for vision. Same memmap+offsets concept.
- **`backend="shards"`** — sharded sequential streaming, if a dataset ever exceeds local NVMe.
- **Remote sync** — designed with the cloud discussion, against real requirements.
- **`dsio runs compare`** — if eyeballing deltas replaces running the paired-floor comparison,
  that is the signal to add it back; the verdict logic is untouched either way.

---

## ADRs to write

1. **Lightning is the only training path** — records decision 1 and the four deleted
   neutrality abstractions; supersedes the multi-modality scope of the original spec.
2. **MLflow is the source of truth; runs fail without it** — supersedes ADR 0002. Records the
   artifact-not-param resolution and the Postgres single-point-of-failure risk.
3. **The process boundary is the fold boundary** — records decision 6 and where each
   guarantee moved.
4. **The repository is a repository, not a template** — records decision 2 and the downgrade
   of the project→dsio direction from structural to lint-enforced.

Existing ADRs 0001 (config in Python), 0003 (never block, gate at promotion) and 0005
(flat-binary store) stand unchanged. ADR 0004's survey stands as history.

ADR 0006 (splits are committed group lists) stands but is **amended**: one file holds all
folds as an ordered list, rather than one file per fold. Order becomes explicit in the
document, which makes the fold-ordering bug unrepresentable instead of defended against, and
lets cross-fold disjointness be checked at load.

ADR 0009 (content-addressed remotes) is **suspended, not superseded**: the code is cut with
`remote.py` and the decision should be revisited on its merits during the cloud discussion,
when there are real requirements to design against.

ADR 0007 (stage cache), 0008 (fold loop owns comparison), 0010 (Lightning inside a fold),
0011 (SSL is first-class), 0012 (ledger is the resume state), 0013 (agentic is a modality)
and 0014 (selection under multiplicity) are superseded in whole or part by the decisions
above. Mark them; do not delete them — the reasoning is the record of why the code existed.

---

## Verification

The migration is complete when, from a fresh clone:

1. `uv sync && uv run pytest && uv run ruff check . && uv run mypy && uv run lint-imports`
   passes — including the new no-cycle check between `data` and `splits`.
2. `docker compose up -d` brings up Postgres and MLflow; `dsio run <preset>` records a run,
   and killing MLflow makes the next run fail immediately rather than after training.
3. **Determinism:** same config + same seed → identical metrics; different seed → different
   metrics (guards a seed that is recorded but never wired through).
4. **Fold-as-process:** a shell loop over 5 folds produces 5 MLflow runs in one experiment;
   rerunning fold 2 alone reproduces its metrics exactly.
5. **Leakage, named per invariant:** split parts mutually disjoint; cross-fold test
   disjointness caught at `SplitFile` load; purge/embargo drops exactly the windows whose
   label horizon crosses the boundary.
6. **Fail-closed registry:** corrupt a stored model → load raises on digest mismatch rather
   than returning weights.
7. **Round-trip config:** `RunConfig.model_validate(yaml.safe_load(config.resolved.yaml))`
   reconstructs an object equal to the original.
8. **Reproduction:** the `reproduce.sh` logged as an MLflow artifact reruns to identical
   metrics, including from a run made with a dirty working tree.

## Out of scope

Cloud training and everything it implies — remote data sync, a reachable MLflow, artifact
transport. To be designed when the requirement is real, not inferred.
