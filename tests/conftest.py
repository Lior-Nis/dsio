"""Shared fixtures.

Every test that touches a run's local scratch directory or the data store is pointed at a
tmp_path root. `DSIO_RUNS_ROOT` (`dsio.runs.record.RUNS_ROOT_ENV`) no longer points at a
ledger -- there is not one any more, per decision 7 of the lean design -- but it still
matters: unset, a run's scratch directory (`Run.dir`, holding the resolved config, the
reproduce script and whatever a runner writes before it becomes an MLflow artifact) lands
under the OS temp directory, not a tmp_path pytest cleans up on its own. Pointing it here
keeps that scratch space contained the same way `stores/` already needs to be.

The same isolation covers MLflow: pointing `MLFLOW_TRACKING_URI` at a tmp_path isolates
saved artifacts between tests. `dsio.train.tracking.require_mlflow` makes every
torch/ssl_pretrain run
fail immediately if MLflow is unreachable, and `uv run pytest` must still pass
on a machine with nothing running (the plan's "the suite must not need Docker"
constraint) -- so every test, by default, is pointed at a `file:` tracking URI under its
own `tmp_path` rather than the compose stack's `localhost:5000`. That makes the *default*
suite exercise the real MLflow client against a local backend (a "fake-backed" test, in the
plan's own vocabulary), not a mock, and not a live server. `MLFLOW_ALLOW_FILE_STORE` opts
back into the file store, which recent MLflow versions otherwise refuse with a "migrate to
a database backend" error. Tests that need the live compose stack override
`MLFLOW_TRACKING_URI` themselves and are marked `live`.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from dsio.config import RunConfig
from dsio.data.store import DATA_ROOT_ENV
from dsio.runs.record import RUNS_ROOT_ENV


@pytest.fixture(autouse=True)
def _isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow

    monkeypatch.setenv(RUNS_ROOT_ENV, str(tmp_path / "runs"))
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "stores"))
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunConfig:
    """A real, tiny `TorchTask` config — small enough to train in under a second.

    Built the same way `tests/train/test_torch_runner.py` builds its own: a synthetic
    two-class `SignalStore` with a hand-picked single-fold split, so callers that
    actually execute the config (determinism tests) get real, seed-sensitive training,
    not just a config object that happens to validate.
    """
    from dsio.data.adapters import entity_examples
    from dsio.data.splits.models import SplitFile, SplitFold
    from dsio.data.store import DATA_ROOT_ENV, SignalStore
    from dsio.data.views import WindowSpec
    from dsio.train import load_runners
    from dsio.train.torch_task import TorchTask, TrainerConfig

    load_runners()

    root = tmp_path / "stores"
    monkeypatch.setenv(DATA_ROOT_ENV, str(root))
    rng = np.random.default_rng(0)
    with SignalStore.builder(root / "tone", channels=1) as builder:
        for group in range(4):
            positive = group % 2 == 0
            t = np.arange(200) / 100.0
            signal = (rng.standard_normal((200, 1)) * 0.5).astype("float32")
            if positive:
                signal[:, 0] += (np.sin(2 * np.pi * 5 * t) * 2.0).astype("float32")
            builder.add(
                f"p{group}", signal, group=f"p{group}", attrs={"positive": int(positive)}
            )

    store = SignalStore(root / "tone")
    digest = entity_examples(store).digest
    parts = {"test": ["p0"], "val": ["p1"], "train": ["p2", "p3"]}
    SplitFile(
        store=store.path.name,
        store_manifest_sha256=digest,
        name="k1",
        folds=[
            SplitFold(
                index=0,
                counts={part: len(members) for part, members in parts.items()},
                parts=parts,
            )
        ],
    ).save(tmp_path / "splits" / "k1" / "split.yaml")

    task = TorchTask(
        store="tone",
        window=WindowSpec(length=64, stride=32, label_policy="majority"),
        labels={
            "reference": "dsio.data.labels:entity_attribute_labels",
            "parameters": {"attribute": "positive"},
        },
        split="k1",
        fold=0,
        splits_root=tmp_path / "splits",
        backbone={
            "reference": "dsio.model.components:Conv1dEncoder",
            "parameters": {"hidden": 4, "out_dim": 4, "depth": 1},
        },
        head={
            "reference": "dsio.model.components:linear_head",
            "parameters": {"out_dim": 2},
        },
        loss={
            "reference": "dsio.model.components:CrossEntropy",
            "parameters": {"threshold": 0.5},
        },
        transform={"reference": "dsio.model.components:InstanceStandardize"},
        batch_size=4,
        metrics=("accuracy",),
        trainer=TrainerConfig(
            max_epochs=2, accelerator="cpu", devices=1, checkpoint=False,
            enable_progress_bar=False,
        ),
    )
    return RunConfig(name="test", seed=42, task=task)


@pytest.fixture
def git_repo(tmp_path: Path) -> Iterator[Path]:
    """A real git repository with one commit, for provenance tests."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*args: str) -> None:
        subprocess.run(args, cwd=repo, check=True, capture_output=True)

    run("git", "init", "-q", "-b", "main")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    (repo / "tracked.txt").write_text("original\n")
    run("git", "add", "tracked.txt")
    run("git", "commit", "-q", "-m", "initial")
    yield repo
