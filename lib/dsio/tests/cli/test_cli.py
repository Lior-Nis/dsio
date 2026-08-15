"""The CLI contract: one JSON envelope, always, with a machine-readable code."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def dsio(*args: str, cwd: Path, env_extra: dict[str, str] | None = None) -> tuple[int, dict]:
    """Invoke the CLI as a subprocess and parse its envelope."""
    import os

    env = {**os.environ, **(env_extra or {})}
    completed = subprocess.run(
        [sys.executable, "-m", "dsio.cli.main", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(completed.stdout) if completed.stdout.strip() else {}
    return completed.returncode, payload


_FIXTURE_PRESET = '''
from dsio.config import RunConfig, preset
from dsio.train.tabular import TabularTask


@preset
def spine_baseline(dataset: str = "iris", estimator: str = "logreg") -> RunConfig:
    return RunConfig(
        name=f"{dataset}-{estimator}",
        task=TabularTask(dataset=dataset, estimator=estimator),
    )
'''


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """A scratch directory carrying a preset module the spawned CLI can discover.

    The spine ships no presets of its own, and a subprocess cannot see one registered in
    conftest. Writing a module here and pointing DSIO_PRESET_MODULES at it exercises the
    environment-override discovery path as a side effect.
    """
    (tmp_path / "presets_fixture.py").write_text(_FIXTURE_PRESET)
    return tmp_path


@pytest.fixture
def preset_env(workdir: Path) -> dict[str, str]:
    return {
        "PYTHONPATH": str(workdir),
        "DSIO_PRESET_MODULES": "presets_fixture",
    }


def test_success_envelope_shape(workdir: Path) -> None:
    code, payload = dsio("presets", cwd=workdir)
    assert code == 0
    assert payload["ok"] is True
    assert payload["error"] is None and payload["code"] is None
    assert payload["retryable"] is False


def test_failure_envelope_carries_a_code(workdir: Path) -> None:
    code, payload = dsio("run", "nope", "--dry-run", cwd=workdir)
    assert code == 1
    assert payload["ok"] is False
    assert payload["code"] == "unknown_component"
    assert "unknown preset 'nope'" in payload["error"]


def test_bad_override_is_classified(workdir: Path, preset_env: dict[str, str]) -> None:
    code, payload = dsio(
        "run", "spine_baseline", "task.nope=1", "--dry-run", cwd=workdir, env_extra=preset_env
    )
    assert code == 1
    assert payload["code"] == "bad_override"


def test_usage_error_is_json_not_text(workdir: Path) -> None:
    """Even Click's own parse failures must render as an envelope."""
    code, payload = dsio("run", cwd=workdir)
    assert code != 0
    assert payload.get("ok") is False
    assert payload.get("code") is not None


def test_run_then_list_then_promote(workdir: Path, preset_env: dict[str, str]) -> None:
    """The end-to-end path the plan calls the headline test."""
    env = {
        **preset_env,
        "DSIO_RUNS_ROOT": str(workdir / "runs"),
        "DSIO_REGISTRY_ROOT": str(workdir / "models"),
    }
    code, run_payload = dsio("run", "spine_baseline", "--summary", cwd=workdir, env_extra=env)
    assert code == 0, run_payload
    run_id = run_payload["run_id"]
    assert run_payload["metrics"]["accuracy"] > 0.5

    code, listing = dsio("runs", "list", cwd=workdir, env_extra=env)
    assert code == 0
    assert listing["count"] == 1
    assert listing["runs"][0]["run_id"] == run_id

    # No git provenance in a bare tmp dir, so promotion must refuse.
    code, blocked = dsio(
        "models", "promote", run_id, "--name", "m", cwd=workdir, env_extra=env
    )
    assert code == 1
    assert blocked["code"] == "blocked"


def test_summary_projection_omits_the_config(
    workdir: Path, preset_env: dict[str, str]
) -> None:
    """Output projection ships from day one; algua had to retrofit it."""
    env = {**preset_env, "DSIO_RUNS_ROOT": str(workdir / "runs")}
    _, full = dsio("run", "spine_baseline", "--dry-run", cwd=workdir, env_extra=env)
    _, brief = dsio(
        "run", "spine_baseline", "--dry-run", "--summary", cwd=workdir, env_extra=env
    )
    assert "config" in full
    assert "config" not in brief
    assert full["config_hash"] == brief["config_hash"]
