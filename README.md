# dsio

A reusable ML/DL experimentation library.

`dsio` currently ships generic data, split, training, and evaluation components. The accepted
architecture extends that spine with generic tracking and inference components.
Consumer projects own their workflows as ordinary Prefect flows; DSio does not provide a
second DAG model, CLI, scheduler, or deployment layer. PyTorch and Lightning are the single
training path, and MLflow is the evidence store.

## Start a project

Configure the consumer project's PyTorch source before resolving DSio. For example, a CPU
development lock can contain:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

Then add a pinned DSio release or immutable commit:

```bash
uv add "dsio @ git+ssh://git@github.com/<org>/dsio.git@<tag-or-commit>"
```

GPU projects select their appropriate PyTorch index instead. DSio's package metadata stays
accelerator-neutral.

Define orchestration in the consumer project and call DSio functions normally:

```python
from prefect import flow, task
from mlflow import MlflowClient

from dsio.contracts import sha256_of
from dsio.tracking import (
    attempt,
    evidence_uri,
    experiment,
    record_provenance,
    resolve_evidence,
)


@task
def identify_dataset(dataset: dict[str, object], parent_run_id: str) -> tuple[str, str]:
    with attempt(parent_run_id) as child:
        digest = sha256_of(dataset)
        record_provenance(
            child.info.run_id,
            {"dataset": dataset, "seed": 7},
            components={"identity": "dsio.contracts:sha256_of"},
        )
        MlflowClient().log_param(child.info.run_id, "dataset_digest", digest)
        MlflowClient().log_dict(
            child.info.run_id,
            {"dataset_digest": digest},
            "outputs/dataset.json",
        )
        return digest, child.info.run_id


@flow
def train_experiment() -> tuple[str, str]:
    with experiment("algae-training") as parent:
        digest, child_run_id = identify_dataset(
            {"name": "algae", "revision": 1}, parent.info.run_id
        )
        if not digest:
            raise ValueError("dataset identity is required")
        return digest, child_run_id


if __name__ == "__main__":
    digest, child_run_id = train_experiment()
    identity = MlflowClient().get_run(child_run_id).data.params["dsio.execution_identity"]
    prior = resolve_evidence(identity, required_artifacts={"outputs/dataset.json"})
    if prior is not None:
        print(evidence_uri(prior.info.run_id, "outputs/dataset.json"))
```

No DSio runner or flow wrapper is involved. Each flow execution explicitly creates a fresh
native MLflow parent Run. Tasks receive its Run ID as ordinary data and open one native child
Run per Prefect attempt. Evidence is logged with the explicit child Run ID; required output
validation stays inside the parent context so MLflow records failure instead of false success.
Prefect executes the project's flow through its normal Python entry point.
`resolve_evidence(...)` treats missing or invalid evidence as a cache miss and returns only a
native successful MLflow `Run` whose immutable identity, provenance, and required artifacts
agree. `evidence_uri(...)` emits an exact `runs:/<run-id>/<artifact>` reference; aliases and
stages are not accepted. Pure serializable tasks can pass `prefect_cache_key` directly as
Prefect's `cache_key_fn`, while evidence-producing tasks continue to resolve through MLflow.
For a downstream-only rerun, validate source Runs in the project flow with
`require_evidence(...)`, build exact artifact URIs, and pass them to only the downstream tasks
the project chooses. Record those Run IDs and URIs as ordinary provenance inputs; DSio does
not infer a rerun plan or invoke upstream work.

## Shape

A single versioned Python distribution rooted at one package:

```
pyproject.toml    distribution metadata and development tooling
src/dsio/         reusable components grouped by pipeline responsibility
tests/            its test suite
```

Projects import a pinned DSio version and own their Prefect DAGs. ADR 0019 records this
boundary; the accepted generic-spine specification describes the target package layout.

## Principles

**Use native contracts.** Prefer Lightning modules, Lightning data modules, TorchMetrics,
MLflow artifacts, and Prefect flows over parallel DSio models for the same concepts.

**Store once, index many.** A corpus is stored once as continuous signal; windowing
produces an index of offsets, not a copy. Materialize only what is expensive *and*
deterministic; index everything cheap and combinatorial.

**Never block, always reconstructible.** A dirty working tree does not stop a run — the
diff is captured as an artifact, so even a dirty run reproduces exactly. The clean-tree
gate belongs at model-registry promotion, and **is not implemented**: there is no
`dsio registry promote`, so nothing enforces it. What a run records is enough to decide
the question later (`RunRecord.reproducible`); deciding it is not wired up. See
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
index = build_index(store, WindowSpec(length=64, stride=64),
                    one_window_per_entity=True)                 # 16 images -> 16 windows
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

**Fixed-size only, and say so.** `WindowSpec.length` is a single int for the whole store, so a
corpus of differently-sized images has no length that means "one item". The two ways that goes
wrong are not equally visible. An item *smaller* than `length` yields no window at all, and
`build_index` refuses that by default — a whole image absent from the index is loss nothing
downstream can see. An item *larger* than `length` is the quiet one: at `length=64` a 12×12
image becomes two windows that are not images. Nothing is lost, so nothing is missing, and the
loader keeps yielding tensors of exactly the right shape — every one of them a crop.

`one_window_per_entity=True` is how you say the corpus is items rather than recordings. It is
off by default because a waveform corpus wants many windows per recording, and it is the one
thing the other checks cannot infer: only the caller knows a window was supposed to be a whole
image. Claim it and a store that stops being uniform — one 12×12 scan among the 8×8s — is
refused by name, on the cache-hit path too, rather than quietly training on strips. Then resize
at ingest, or give each size its own store.

## Developing

```bash
uv sync --locked
uv build
uv run pytest -q
uv run ruff check .
uv run mypy
uv run lint-imports
```

Decisions and their reasons live in `docs/adr/`.

## Tracking

MLflow is the source of truth for run evidence (see the accepted generic-spine spec).
`dsio.tracking.experiment(...)` adds the flow-level lifecycle invariant: a fresh parent Run
per execution. `dsio.tracking.attempt(parent_run_id)` adds one native child Run for the current
Prefect task attempt, tagged with its task and retry identity. Both contexts persist
`FINISHED`, `FAILED`, or `KILLED` and fail closed when required lifecycle evidence cannot be
written. Child logging always targets `child.info.run_id` explicitly; DSio does not introduce
an evidence wrapper or depend on MLflow's ambient active Run inside concurrent tasks.
`record_provenance(...)` writes the node's safe normalized configuration and deterministic
identity as native MLflow evidence; projects still own the configuration itself and explicitly
declare secret and ephemeral fields to omit.

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
