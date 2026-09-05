# dsio

A reproducible ML/DL experimentation spine.

`dsio` owns the parts every project rebuilds badly: typed configuration, staged data with
content-addressed caching, leakage-safe splits, provenance capture that makes results
reconstructible, and evaluation with honest verdicts. Torch and Lightning are the one
first-class training path (see `docs/adr/0015-lightning-is-the-only-training-path.md`):
a model is a `LightningModule` assembled from registered backbone/head/loss/transform
components, not a per-modality wrapper. There is no universal `Model` abstraction fighting
the framework — Lightning already is one.

## Start a project

```bash
git clone https://github.com/<you>/dsio my-project
cd my-project
git remote rename origin upstream
uv sync --locked --extra gpu   # or --extra cpu on a machine without an NVIDIA GPU
uv run dsio run                # lists the presets
uv run dsio run spine_baseline
```

Selecting an extra is mandatory: a bare `uv sync` installs no torch at all, and that is
deliberate — it fails loudly instead of silently pulling several gigabytes you may not
want. CI uses `cpu`. `gpu` currently floats on the default PyPI index because that
platform's default wheel is already CUDA-enabled (`torch 2.13.0+cu130`, `cuda.is_available()
== True` on an RTX 5070 Ti, next to `2.13.0+cpu` from `cpu`); if that ever stops being true,
`gpu` needs an explicit CUDA index again.

Add your components under `src/dsio/` — a backbone in `model/`, a preset in `presets.py`.
Later, pull spine improvements without losing your work:

```bash
git fetch upstream && git merge upstream/main
```

That is the whole propagation story. It works because the package name is fixed, so paths
line up between your clone and upstream and a merge conflicts only in files you both
edited. The previous Copier template existed to rename the package per project, which is
precisely what stopped a plain merge from working.

## Shape

A single `uv` project rooted at one package:

```
pyproject.toml    the project
src/dsio/         the package: config, data, dataset, model, splits, train, eval, runs,
                  artifacts, contracts, cli, presets
tests/            its test suite
runs/             a run's local scratch space before its provenance and artifacts land in
                  MLflow (gitignored; MLflow is the source of truth -- see "Tracking" below)
stores/ views/    canonical data and derived indices (manifests committed)
```

`docs/adr/0018-a-repository-not-a-template.md` records why this replaced the earlier
two-distribution Copier layout.

## Principles

**Structure lives in Python, not YAML.** Configs are typed Pydantic objects composed by
preset functions; YAML is a recorded *output* of every run, never an authored input.
Variants are function arguments, so there is no path by which `lr=1e-6` becomes a file you
check in. (`torchtitan` migrated from TOML to this same shape — see `docs/adr/0001`.)

**Store once, index many.** A corpus is stored once as continuous signal; windowing
produces an index of offsets, not a copy. Materialize only what is expensive *and*
deterministic; index everything cheap and combinatorial.

**Never block, always reconstructible.** A dirty working tree does not stop a run — the
diff is captured as an artifact, so even a dirty run reproduces exactly. The clean-tree
gate belongs at model-registry promotion: `dsio.artifacts.store.promotion_blockers`
enforces it today as a policy function with its own tests, ahead of the CLI command
(`dsio registry promote`) that would call it, which is still aspirational — see
`docs/adr/0003-never-block-gate-at-promotion.md`.

**Correctness is structural.** Leakage walls are import-linter contracts, not review
conventions.

## Images and other fixed-size items

Whole-item access is degenerate windowing — "store once, index many" with an index of one
window per entity — so a fixed-size vision corpus needs no code that does not already exist.
Store an image of H×W with C channels as an entity of `H*W` rows and `C` channels, index it
with `WindowSpec(length=H*W, stride=H*W)`, and every entity yields exactly one window: the
image. There is no separate item-store concept to add, because an item *is* a window of
exactly entity length.

```python
with SignalStore.builder("stores/scans", channels=3, dtype="uint8") as builder:
    for name, image, patient in scans():          # image is (8, 8, 3)
        builder.add(name, image.reshape(64, 3), group=patient)

store = SignalStore("stores/scans")
index = build_index(store, WindowSpec(length=64, stride=64))    # 16 images -> 16 windows
batch = next(iter(make_loader(WindowDataset(store, index), batch_size=4)))
batch["x"].reshape(4, 3, 8, 8)                                  # (B, C, H*W) -> (B, C, H, W)
```

`WindowDataset` is `channels_first=True` by default, so a batch arrives as `(B, C, H*W)` and
reshapes to `(B, C, H, W)`. The flattening is row-major and nothing else; the reshape is its
exact inverse.

What this buys over a directory of image files is `group=`. Split by patient, site or camera
and `dsio.splits.resolve.assert_no_row_overlap` proves *at the pixel row* that no image landed
in two parts — the discipline medical and scientific imaging needs and a per-file split
quietly skips, since two scans of one patient are near-identical and a model scored across
them is scored on data it trained on. `tests/dataset/test_fixed_size_items.py` pins the whole
chain: 16 images in, 16 windows out, one per entity, every pixel row covered exactly once,
and the right pixels at the right coordinates after the reshape.

**Fixed-size only.** `WindowSpec.length` is a single int for the whole store, so a corpus of
differently-sized images has no length that means "one item": at `length=64` a 12×12 image
becomes two windows that are not images and drops its last 16 rows, with nothing said. Resize
at ingest, or give each size its own store.

## Developing

```bash
uv sync --locked --extra cpu
uv run --extra cpu pytest -q && uv run --extra cpu ruff check . && uv run --extra cpu mypy && uv run --extra cpu lint-imports
```

Decisions and their reasons live in `docs/adr/`.

## Tracking

MLflow is the source of truth for runs (see decision 7 of
`docs/superpowers/specs/2026-08-20-dsio-lean-design.md`): a run fails if it cannot reach it.
Start the local stack before running anything that trains:

```bash
docker compose up -d      # Postgres 16 + MLflow, both named volumes — nothing lands in the repo
docker compose ps         # both services up, postgres healthy
```

MLflow serves its UI and API on `http://localhost:5000`. The backend store lives in Postgres
and artifacts in the `mlartifacts` volume; both are named volumes, never bind mounts, so
`git status` stays clean regardless of how many runs you log. `restart: unless-stopped` means
the containers come back after a reboot on their own — run `docker compose down` when you
actually want to stop them.

**Repairing a stack older than `49cad22`.** `mlflow server`'s `--default-artifact-root`
only affects experiments created *after* it is set, so a long-lived stack whose Postgres
volume predates that commit keeps experiments (including the built-in Default experiment,
id `0`, which is provisioned once at database init and can never be recreated) pointed at a
bare, non-proxied `/artifacts` path — any client that tries to write to one gets
`PermissionError`. Repoint the existing rows at the proxy scheme directly in Postgres:

```sql
UPDATE experiments SET artifact_location = 'mlflow-artifacts:/' || experiment_id
WHERE artifact_location LIKE '/artifacts/%';
```

A fresh clone with fresh named volumes never hits this.
