"""The CLI contract: one JSON envelope, always, with a machine-readable code."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
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


# The label provider is registered here, at the project preset module's import time —
# the spawned CLI subprocess never sees this test module, only what
# `DSIO_PRESET_MODULES` points it at, so anything the preset's preflight needs to
# resolve (the "tone" label) has to be declared inside the fixture module itself.
_FIXTURE_PRESET = '''
from dsio.config import RunConfig, preset
from dsio.data.views import WindowSpec
from dsio.nn.registry import labels
from dsio.train.torch_task import Component, TorchTask, TrainerConfig


@labels("tone")
def _tone(store):
    import numpy as np

    out = np.zeros(store.n_rows, dtype=np.float32)
    for entity in store.entities:
        out[entity.start_row : entity.end_row] = float(entity.attrs["positive"])
    return out


@preset
def project_preset(store: str = "tone", split: str = "k1", lr: float = 3e-3) -> RunConfig:
    return RunConfig(
        name=f"{store}-lr{lr}",
        task=TorchTask(
            store=store,
            window=WindowSpec(length=64, stride=32, label_policy="majority"),
            labels="tone",
            split=split,
            backbone=Component(name="conv1d", params={"hidden": 8, "out_dim": 8, "depth": 1}),
            head=Component(name="linear", params={"out_dim": 2}),
            loss=Component(name="cross_entropy", params={"threshold": 0.5}),
            transform=Component(name="instance_standardize"),
            lr=lr,
            metrics=("accuracy", "roc_auc"),
            trainer=TrainerConfig(
                max_epochs=30, accelerator="cpu", devices=1, checkpoint=False,
                enable_progress_bar=False,
            ),
        ),
    )
'''


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """A scratch directory carrying a preset module the spawned CLI can discover.

    dsio ships `spine_baseline` (`dsio.presets`) by default, so a bare subprocess already
    has a preset to resolve. This fixture writes a *second*, differently-named preset
    module and points DSIO_PRESET_MODULES at it, to exercise the environment-override
    discovery path — the way a real project adds presets alongside the built-in ones.

    `project_preset` is a `TorchTask`, so its preflight needs a real corpus and a
    committed split file — both staged here, on disk, at the exact relative locations
    `TorchTask`'s own defaults resolve against (`./stores/<name>` and `./splits/<name>`
    under the subprocess's `cwd`, which is this directory).
    """
    from dsio.data.store import SignalStore
    from dsio.splits.models import SplitFile

    (tmp_path / "presets_fixture.py").write_text(_FIXTURE_PRESET)

    rng = np.random.default_rng(0)
    with SignalStore.builder(tmp_path / "stores" / "tone", channels=1) as builder:
        for group in range(6):
            positive = group % 2 == 0
            t = np.arange(400) / 100.0
            signal = (rng.standard_normal((400, 1)) * 0.5).astype("float32")
            if positive:
                signal[:, 0] += (np.sin(2 * np.pi * 5 * t) * 2.0).astype("float32")
            builder.add(
                f"p{group}", signal, group=f"p{group}", attrs={"positive": int(positive)}
            )

    SplitFile(
        store="tone",
        name="k1",
        fold=0,
        parts={
            "test": ["p0", "p1"],
            "val": ["p2"],
            "train": ["p3", "p4", "p5"],
        },
    ).save(tmp_path / "splits" / "k1" / "fold0.yaml")
    return tmp_path


@pytest.fixture
def preset_env(workdir: Path) -> dict[str, str]:
    return {
        "PYTHONPATH": str(workdir),
        "DSIO_PRESET_MODULES": "presets_fixture",
    }


def test_success_envelope_shape(workdir: Path) -> None:
    code, payload = dsio("run", cwd=workdir)
    assert code == 0
    assert payload["ok"] is True
    assert payload["error"] is None and payload["code"] is None
    assert payload["retryable"] is False


def test_bare_run_lists_presets(workdir: Path) -> None:
    code, payload = dsio("run", cwd=workdir)
    assert code == 0
    assert "presets" in payload


def test_a_project_preset_does_not_hide_the_builtin_ones(
    workdir: Path, preset_env: dict[str, str]
) -> None:
    """DSIO_PRESET_MODULES adds to discovery, it does not replace it — a project preset
    and a built-in preset must both be listed once both are loaded."""
    code, payload = dsio("run", cwd=workdir, env_extra=preset_env)
    assert code == 0
    assert {"spine_baseline", "project_preset"} <= payload["presets"].keys()


def test_failure_envelope_carries_a_code(workdir: Path) -> None:
    code, payload = dsio("run", "nope", "--dry-run", cwd=workdir)
    assert code == 1
    assert payload["ok"] is False
    assert payload["code"] == "unknown_component"
    assert "unknown preset 'nope'" in payload["error"]


def test_bad_override_is_classified(workdir: Path, preset_env: dict[str, str]) -> None:
    code, payload = dsio(
        "run", "project_preset", "task.nope=1", "--dry-run", cwd=workdir, env_extra=preset_env
    )
    assert code == 1
    assert payload["code"] == "bad_override"


def test_usage_error_is_json_not_text(workdir: Path) -> None:
    """Even Click's own parse failures must render as an envelope."""
    code, payload = dsio("run", "--bogus-flag", cwd=workdir)
    assert code != 0
    assert payload.get("ok") is False
    assert payload.get("code") is not None


def test_run_happy_path(workdir: Path, preset_env: dict[str, str]) -> None:
    """``dsio run <preset>`` resolves, executes, and reports metrics."""
    env = {**preset_env, "DSIO_RUNS_ROOT": str(workdir / "runs")}
    code, run_payload = dsio("run", "project_preset", "--summary", cwd=workdir, env_extra=env)
    assert code == 0, run_payload
    assert run_payload["run_id"]
    # roc_auc, not accuracy: the fixture's tone-vs-noise signal is separable enough that
    # a wired-correctly run should score near-perfect on it (mirrors
    # tests/train/test_torch_runner.py::test_the_runner_learns_a_separable_signal).
    assert run_payload["metrics"]["roc_auc"] > 0.9


def test_summary_projection_omits_the_config(
    workdir: Path, preset_env: dict[str, str]
) -> None:
    """Output projection ships from day one; prior work had to retrofit it."""
    env = {**preset_env, "DSIO_RUNS_ROOT": str(workdir / "runs")}
    _, full = dsio("run", "project_preset", "--dry-run", cwd=workdir, env_extra=env)
    _, brief = dsio(
        "run", "project_preset", "--dry-run", "--summary", cwd=workdir, env_extra=env
    )
    assert "config" in full
    assert "config" not in brief
    assert full["config_hash"] == brief["config_hash"]
