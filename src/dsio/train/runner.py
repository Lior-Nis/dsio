"""Task dispatch: one entrypoint, many runners.

Near-identical training scripts that differ only in a class name and a tag list are the
thing this replaces. There is exactly one entrypoint; the task kind selects the runner.

A runner receives a validated config and a live :class:`~dsio.runs.record.Run`, and
returns its summary metrics. It owns nothing about provenance or recording — those happen
around it, identically for every task kind. Seeding happens here, in the dispatcher, so
every runner gets it for free rather than each reimplementing it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from dsio.config.registry import Registry
from dsio.config.schema import RunConfig
from dsio.runs.seeding import seed_everything

if TYPE_CHECKING:
    from dsio.runs.record import Run

RunnerFn = Callable[[RunConfig, "Run"], dict[str, float]]
PreflightFn = Callable[[RunConfig], None]

RUNNERS: Registry[RunnerFn] = Registry("runner")
PREFLIGHTS: Registry[PreflightFn] = Registry("preflight")


def runner(kind: str) -> Callable[[RunnerFn], RunnerFn]:
    """Register a runner for a task kind."""
    return RUNNERS.register(kind)


def preflight(kind: str) -> Callable[[PreflightFn], PreflightFn]:
    """Register a pre-flight check for a task kind.

    A pre-flight resolves every name the run depends on — estimators, datasets, losses —
    without doing any work. It exists so a typo costs microseconds instead of the time it
    takes to load a corpus. Registering one is optional but strongly expected: a runner
    without a pre-flight will report its typos after the data is already in memory.
    """
    return PREFLIGHTS.register(kind)


def check(config: RunConfig) -> None:
    """Verify a run can execute, before it touches any data."""
    RUNNERS.get(config.task.kind)
    if config.task.kind in PREFLIGHTS:
        PREFLIGHTS.get(config.task.kind)(config)


def execute(config: RunConfig, run: Run) -> dict[str, float]:
    """Seed every RNG a runner can touch, then dispatch ``config`` to its runner.

    ``dsio run`` (``src/dsio/cli/run_cmd.py``) also calls ``seed_everything`` up front, to
    capture the seeds actually applied for the run record before the ledger opens — that
    call stays, since ``execute`` returns only metrics and has nowhere to hand the seeds
    dict back. Seeding here as well is what makes a *direct* call to ``execute`` (every
    test, and any caller that is not going through the CLI) reproducible: same config and
    seed now produce identical metrics regardless of caller. ``seed_everything`` is
    idempotent, so the CLI path's second call is harmless.
    """
    seed_everything(config.seed)
    return RUNNERS.get(config.task.kind)(config, run)
