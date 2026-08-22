"""Shared fixtures.

Every test that touches the ledger, registry or data store is pointed at a tmp_path
root. A test that writes into the real `runs/` (or `stores/`) would pollute the very
record the system exists to protect — `spine_baseline` (`dsio.presets`) now stages a
synthetic corpus on resolution, so this isolation is no longer just about the ledger.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from dsio.artifacts.store import REGISTRY_ROOT_ENV
from dsio.config import RunConfig
from dsio.data.store import DATA_ROOT_ENV
from dsio.runs.record import RUNS_ROOT_ENV, RunLedger


@pytest.fixture(autouse=True)
def _isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUNS_ROOT_ENV, str(tmp_path / "runs"))
    monkeypatch.setenv(REGISTRY_ROOT_ENV, str(tmp_path / "models"))
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "stores"))


@pytest.fixture
def ledger() -> RunLedger:
    return RunLedger()


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunConfig:
    """A real, tiny `TorchTask` config — small enough to train in under a second.

    Built the same way `tests/train/test_torch_runner.py` builds its own: a synthetic
    two-class `SignalStore` with a hand-picked single-fold split, so callers that
    actually execute the config (determinism tests) get real, seed-sensitive training,
    not just a config object that happens to validate.
    """
    from dsio.data.adapters import entity_examples
    from dsio.data.store import DATA_ROOT_ENV, SignalStore
    from dsio.data.views import WindowSpec
    from dsio.model.registry import LABELS, labels
    from dsio.splits.models import SplitFile
    from dsio.train import load_runners
    from dsio.train.torch_task import Component, TorchTask, TrainerConfig

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

    if "tone" not in LABELS:

        @labels("tone")
        def _tone(store: SignalStore) -> np.ndarray:
            out = np.zeros(store.n_rows, dtype=np.float32)
            for entity in store.entities:
                out[entity.start_row : entity.end_row] = float(entity.attrs["positive"])
            return out

    store = SignalStore(root / "tone")
    digest = entity_examples(store).digest
    parts = {"test": ["p0"], "val": ["p1"], "train": ["p2", "p3"]}
    SplitFile(
        store=store.path.name,
        store_manifest_sha256=digest,
        name="k1",
        fold=0,
        counts={part: len(members) for part, members in parts.items()},
        parts=parts,
    ).save(tmp_path / "splits" / "k1" / "fold0.yaml")

    task = TorchTask(
        store="tone",
        window=WindowSpec(length=64, stride=32, label_policy="majority"),
        labels="tone",
        split="k1",
        splits_root=tmp_path / "splits",
        backbone=Component(name="conv1d", params={"hidden": 4, "out_dim": 4, "depth": 1}),
        head=Component(name="linear", params={"out_dim": 2}),
        loss=Component(name="cross_entropy", params={"threshold": 0.5}),
        transform=Component(name="instance_standardize"),
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
