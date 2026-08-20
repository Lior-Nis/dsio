"""Guards the runner-registry bootstrap against a reproduce path that looks alive but
is not.

``dsio.train.load_runners`` imports each module in ``_BUILTIN_RUNNER_MODULES``, and
each of those modules registers a task kind into ``dsio.config.schema.TASKS`` as an
import-time side effect (``@TASKS.register("tabular")`` on ``TabularTask``, and
similarly for the torch and ssl runners). If ``_BUILTIN_RUNNER_MODULES`` were ever
emptied, ``load_runners()`` would still return an empty list without raising, ``dsio
run`` would still print the presets envelope, and ``dsio run <preset> --dry-run``
would still resolve a config — but reading that same config back with
``RunConfig.model_validate`` would die with ``UnknownComponentError: unknown task
'tabular'; no tasks are registered``, because nothing ever imported the module that
registers it. That is the reproduce path, and it would be dead.

This can't be checked in-process: sibling test modules (``tests/train/test_tabular.py``
and friends) import ``dsio.train.tabular`` at collection time, which registers
``TabularTask`` before any test body runs — regardless of whether ``load_runners``
still does its job. A subprocess that calls *only* ``load_runners()`` is the only way
to see what a virgin process actually gets.
"""

from __future__ import annotations

import subprocess
import sys

_PROBE = """
from dsio.train import load_runners

load_runners()

from dsio.config.schema import RunConfig

# The shape of a recorded config.resolved.yaml for a tabular run: enough for
# RunConfig.model_validate to resolve TaskConfig's "kind" through the task registry.
recorded = {
    "name": "reproduce-check",
    "seed": 42,
    "task": {"kind": "tabular", "dataset": "does-not-need-to-exist-for-validation"},
}
cfg = RunConfig.model_validate(recorded)
assert cfg.task.kind == "tabular"
"""


def test_load_runners_alone_is_enough_to_read_back_a_recorded_config() -> None:
    """The reproduce path (`RunConfig.model_validate` on a recorded config) must work
    after nothing but `load_runners()` has run — not after some other test module has
    already imported the runner as a side effect."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"reproducing a recorded config failed in a virgin process after load_runners()\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
