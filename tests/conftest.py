"""Shared isolated MLflow, data-store, and Git fixtures."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from dsio.data.store import DATA_ROOT_ENV


@pytest.fixture(autouse=True)
def _isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow

    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "stores"))
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)


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
