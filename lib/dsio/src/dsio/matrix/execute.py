"""Running a matrix, resumably.

**The resume state is the run ledger, not a sidecar.** A cell is done when the ledger holds
a completed run whose ``config_hash`` matches it. Nothing else is written and nothing else
is consulted.

That is a deliberate correction of FORGE's ``.probed_checkpoints`` file, which is the
standard shape for this and is wrong in a specific way: a sidecar records an *intention* and
can disagree with what happened. It says "done" for a run that crashed after the line was
appended, or omits a run that completed before the write. Deriving resume state from the
ledger means it cannot drift, because the ledger is the same record the result is read from.

**A failed cell does not stop the sweep, and is not swallowed either.** It is recorded as a
failed run — with its traceback in the record — the sweep continues, and the summary reports
every failure with a non-zero exit. One bad cell out of two hundred should cost one cell,
but a sweep that quietly reports success while a fifth of it failed is worse than one that
stops. ``fail_fast`` gives the other behaviour when that is what you want.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from dsio.config.presets import resolve
from dsio.matrix.cells import Cell
from dsio.runs.record import RunLedger, RunStatus
from dsio.runs.seeding import seed_everything
from dsio.train.runner import check, execute


@dataclass
class CellResult:
    """What happened to one cell."""

    index: int
    label: str
    config_hash: str
    status: str
    run_id: str | None = None
    seconds: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "config_hash": self.config_hash[:12],
            "status": self.status,
            "run_id": self.run_id,
            "seconds": round(self.seconds, 3),
            "metrics": self.metrics,
            "error": self.error,
        }


@dataclass
class MatrixReport:
    """The result of a whole sweep."""

    preset: str
    total: int
    cells: list[CellResult] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return sum(1 for cell in self.cells if cell.status == "completed")

    @property
    def skipped(self) -> int:
        return sum(1 for cell in self.cells if cell.status == "skipped")

    @property
    def failed(self) -> list[CellResult]:
        return [cell for cell in self.cells if cell.status == "failed"]

    @property
    def ok(self) -> bool:
        return not self.failed

    def as_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "total": self.total,
            "completed": self.completed,
            "skipped": self.skipped,
            "failed": len(self.failed),
            "ok": self.ok,
            "cells": [cell.as_dict() for cell in self.cells],
        }


def completed_runs(ledger: RunLedger) -> dict[str, Any]:
    """Every completed run, keyed by config hash, newest first losing to oldest.

    Keeping the record rather than just the hash is what lets a search reuse a metric the
    matrix already computed, instead of retraining an identical configuration.
    """
    found: dict[str, Any] = {}
    for record in ledger.list_runs():
        if record.status is RunStatus.COMPLETED:
            found.setdefault(record.config_hash, record)
    return found


def completed_hashes(ledger: RunLedger) -> set[str]:
    """Config hashes of every run the ledger records as completed.

    The whole of the resume mechanism. A run that crashed is absent, so it will be retried;
    a run that finished is present, so it will not.
    """
    return set(completed_runs(ledger))


def run_matrix(
    preset: str,
    cells: Sequence[Cell],
    *,
    ledger: RunLedger | None = None,
    fail_fast: bool = False,
    dry_run: bool = False,
    retry_failed: bool = True,
    on_cell: Callable[[CellResult], None] | None = None,
    command: tuple[str, ...] = (),
) -> MatrixReport:
    """Execute every cell that has not already completed.

    ``retry_failed`` is on by default because a failed cell is not a finished one — the
    usual reason a cell failed is a bug that has since been fixed, and re-running it is the
    point of invoking the sweep again.
    """
    ledger = ledger or RunLedger()
    done = completed_hashes(ledger)
    report = MatrixReport(preset=preset, total=len(cells))

    for cell in cells:
        config = resolve(preset, list(cell.overrides))
        digest = config.config_hash

        if digest in done:
            result = CellResult(
                index=cell.index,
                label=cell.label,
                config_hash=digest,
                status="skipped",
            )
            report.cells.append(result)
            if on_cell is not None:
                on_cell(result)
            continue

        if dry_run:
            result = CellResult(
                index=cell.index, label=cell.label, config_hash=digest, status="pending"
            )
            report.cells.append(result)
            if on_cell is not None:
                on_cell(result)
            continue

        started = time.perf_counter()
        try:
            check(config)
            seeds = seed_everything(config.seed)
            with ledger.start(
                name=config.name,
                config=config.to_dict(),
                config_hash=digest,
                seed=config.seed,
                seeds=seeds,
                tags=config.tags,
                command=command,
            ) as active:
                metrics = execute(config, active)
                active.finish(RunStatus.COMPLETED, metrics=metrics)
            result = CellResult(
                index=cell.index,
                label=cell.label,
                config_hash=digest,
                status="completed",
                run_id=active.run_id,
                seconds=time.perf_counter() - started,
                metrics=metrics,
            )
            # Guard against two cells resolving to one config within a single sweep: the
            # second would otherwise re-run work the first just did.
            done.add(digest)
        except Exception as error:  # recorded on the report and re-surfaced, never dropped
            result = CellResult(
                index=cell.index,
                label=cell.label,
                config_hash=digest,
                status="failed",
                seconds=time.perf_counter() - started,
                error=f"{type(error).__name__}: {error}",
            )
            result.metrics = {}
            if fail_fast:
                report.cells.append(result)
                if on_cell is not None:
                    on_cell(result)
                raise
            traceback.print_exc()

        report.cells.append(result)
        if on_cell is not None:
            on_cell(result)

    return report
