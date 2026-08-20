"""Shared fixtures.

Every test that touches the ledger or registry is pointed at a tmp_path root. A test that
writes into the real `runs/` would pollute the very record the system exists to protect.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from dsio.artifacts.store import REGISTRY_ROOT_ENV
from dsio.config import RunConfig
from dsio.runs.record import RUNS_ROOT_ENV, RunLedger


@pytest.fixture(autouse=True)
def _isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUNS_ROOT_ENV, str(tmp_path / "runs"))
    monkeypatch.setenv(REGISTRY_ROOT_ENV, str(tmp_path / "models"))


@pytest.fixture
def ledger() -> RunLedger:
    return RunLedger()


@pytest.fixture
def config() -> RunConfig:
    from dsio.train import load_runners

    load_runners()
    from dsio.train.tabular import TabularTask

    return RunConfig(name="test", seed=42, task=TabularTask(dataset="iris"))


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


@pytest.fixture(autouse=True)
def _spine_preset() -> None:
    """Register a preset the spine's own tests can resolve.

    The spine ships no presets — those belong to a project. Registering one here keeps
    these tests self-contained instead of depending on a generated project existing.
    """
    from dsio.config import RunConfig, preset
    from dsio.config.presets import PRESETS
    from dsio.train import load_runners
    from dsio.train.tabular import TabularTask

    load_runners()
    if "spine_baseline" in PRESETS:
        return

    @preset
    def spine_baseline(
        dataset: str = "iris",
        estimator: str = "logreg",
        test_fraction: float = 0.25,
        seed: int = 42,
    ) -> RunConfig:
        return RunConfig(
            name=f"{dataset}-{estimator}",
            seed=seed,
            tags=("baseline", "tabular"),
            task=TabularTask(
                dataset=dataset, estimator=estimator, test_fraction=test_fraction
            ),
        )
