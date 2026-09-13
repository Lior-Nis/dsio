"""Guards the runner-registry bootstrap against a reproduce path that looks alive but
is not.

``dsio.application.load_run_config`` loads each module in ``_BUILTIN_RUNNER_MODULES``, and
each of those modules registers a task kind into ``dsio.config.schema.TASKS`` as an
import-time side effect (``@TASKS.register("torch")`` on ``TorchTask``, and similarly
for the ssl runner). If ``_BUILTIN_RUNNER_MODULES`` were ever emptied, ``load_runners()``
would still return an empty list without raising, ``dsio run`` would still print the
presets envelope, and ``dsio run <preset> --dry-run`` would still resolve a config — but
reading that same config back with ``RunConfig.model_validate`` would die with
``UnknownComponentError: unknown task 'torch'; no tasks are registered``, because
nothing ever imported the module that registers it. That is the reproduce path, and it
would be dead.

This can't be checked in-process: sibling test modules (``tests/train/test_torch_runner.py``
and friends) import ``dsio.train.torch_task`` at collection time, which registers
``TorchTask`` before any test body runs — regardless of whether ``load_runners``
still does its job. A subprocess that calls only the public recorded-config loader is the
only way to see what a virgin process actually gets.
"""

from __future__ import annotations

import subprocess
import sys

_PROBE = """
import os

os.environ["DSIO_PRESET_MODULES"] = "must_not_import_this_preset_module"
from dsio.application import load_run_config

# The shape of a recorded config.resolved.yaml for a torch run: enough for
# RunConfig.model_validate to resolve TaskConfig's "kind" through the task registry.
# None of these names need to resolve through a component registry — that happens at
# preflight/execute time, not at model_validate time.
recorded = {
    "name": "reproduce-check",
    "seed": 42,
    "task": {
        "kind": "torch",
        "store": "does-not-need-to-exist-for-validation",
        "window": {"length": 8, "stride": 8, "label_policy": "majority"},
        "labels": "does-not-need-to-exist-for-validation",
        "split": "does-not-need-to-exist-for-validation",
        "fold": 0,
        "backbone": {"name": "does-not-need-to-exist-for-validation"},
    },
}
first = load_run_config(recorded)
second = load_run_config(recorded)
assert first.task.kind == second.task.kind == "torch"
"""


def test_public_loader_bootstraps_a_recorded_config_without_loading_presets() -> None:
    """The reproduce path (`RunConfig.model_validate` on a recorded config) must work
    through its public loader — not after some other test module has already imported the
    runner or configured preset modules as a side effect. Repeated loads remain safe."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"public recorded-config loading failed in a virgin process\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
