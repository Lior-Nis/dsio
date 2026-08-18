"""Matrix invariants: content-addressed cells, and resume derived from the ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

from dsio.matrix import (
    Axis,
    MatrixError,
    completed_hashes,
    expand,
    parse_axes,
    parse_axis,
    run_matrix,
    size,
    validate,
)
from dsio.runs import RunLedger, RunStatus

# --- axes -----------------------------------------------------------------------------


def test_an_axis_parses_a_comma_list() -> None:
    axis = parse_axis("task.folds=0,1,2")
    assert axis.path == "task.folds"
    assert axis.values == ("0", "1", "2")


def test_an_axis_rejects_a_repeated_value() -> None:
    """A repeated value silently halves the apparent size of the sweep."""
    with pytest.raises(MatrixError, match="repeats"):
        Axis(path="task.lr", values=("1e-3", "1e-3"))


def test_an_axis_without_an_equals_is_rejected() -> None:
    with pytest.raises(MatrixError, match="path=value"):
        parse_axis("task.lr")


def test_an_empty_axis_is_rejected() -> None:
    with pytest.raises(MatrixError, match="no values"):
        parse_axis("task.lr=")


def test_a_bad_value_fails_at_parse_not_at_run() -> None:
    """Each value is validated as a real override immediately, so a typo costs a parse."""
    with pytest.raises(Exception):
        parse_axis("=1,2")


def test_a_repeated_axis_path_is_rejected() -> None:
    """The second would silently win and half the sweep would never run."""
    with pytest.raises(MatrixError, match="given twice"):
        parse_axes(["task.lr=1e-3,1e-4", "task.lr=1e-5"])


def test_a_glob_axis_is_resolved_and_sorted(tmp_path: Path) -> None:
    """Resolved at parse time: a matrix whose size depends on when it was expanded cannot
    be resumed, because the second invocation would be a different matrix."""
    for name in ("b.ckpt", "a.ckpt", "c.ckpt"):
        (tmp_path / name).write_text("x")
    axis = parse_axis("task.ckpt=glob:*.ckpt", root=tmp_path)
    assert [Path(v).name for v in axis.values] == ["a.ckpt", "b.ckpt", "c.ckpt"]


def test_a_glob_matching_nothing_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(MatrixError, match="matched no paths"):
        parse_axis("task.ckpt=glob:*.nope", root=tmp_path)


# --- expansion --------------------------------------------------------------------------


def test_the_product_has_the_expected_shape() -> None:
    axes = parse_axes(["task.folds=0,1,2", "task.estimator=logreg,random_forest"])
    cells = expand(axes)
    assert size(axes) == 6
    assert len(cells) == 6
    assert len({cell.overrides for cell in cells}) == 6


def test_the_last_axis_varies_fastest() -> None:
    """The order a nested loop would produce, so a partial sweep reads the way it ran."""
    cells = expand(parse_axes(["a.x=1,2", "a.y=p,q"]))
    assert [cell.label for cell in cells] == [
        "a.x=1 a.y=p",
        "a.x=1 a.y=q",
        "a.x=2 a.y=p",
        "a.x=2 a.y=q",
    ]


def test_base_overrides_apply_to_every_cell() -> None:
    cells = expand(parse_axes(["a.x=1,2"]), base=("a.seed=7",))
    assert all("a.seed=7" in cell.overrides for cell in cells)


def test_no_axes_yields_a_single_cell() -> None:
    cells = expand([])
    assert len(cells) == 1 and cells[0].coordinates == ()


# --- identity ----------------------------------------------------------------------------


def test_identity_does_not_depend_on_the_order_axes_were_given() -> None:
    """Identity is the resolved config, not the path taken to it.

    Swapping ``--axis a --axis b`` for ``--axis b --axis a`` reorders the cells but must not
    change which jobs exist — otherwise reordering a command line would silently re-run a
    completed sweep from scratch.
    """
    forward = validate(
        "spine_baseline", expand(parse_axes(["task.folds=2,3", "task.estimator=logreg"]))
    )
    reversed_ = validate(
        "spine_baseline", expand(parse_axes(["task.estimator=logreg", "task.folds=2,3"]))
    )
    assert set(forward) == set(reversed_)


def test_adding_an_axis_value_does_not_change_existing_identities() -> None:
    small = validate("spine_baseline", expand(parse_axes(["task.folds=2,3"])))
    large = validate("spine_baseline", expand(parse_axes(["task.folds=2,3,4"])))
    assert small == large[:2]


def test_cells_reaching_the_same_config_share_an_identity() -> None:
    """A duplicate is impossible rather than merely unlikely."""
    direct = validate("spine_baseline", expand(parse_axes(["task.folds=3"])))
    via_base = validate("spine_baseline", expand([], base=("task.folds=3",)))
    assert direct == via_base


def test_validation_names_the_cell_that_does_not_resolve() -> None:
    """Two hundred cells that each fail after loading a corpus is two hundred wasted loads."""
    cells = expand(parse_axes(["task.no_such_field=1,2"]))
    with pytest.raises(MatrixError, match="does not resolve"):
        validate("spine_baseline", cells)


# --- resume ---------------------------------------------------------------------------------


def test_resume_state_comes_from_the_ledger(ledger: RunLedger) -> None:
    """Not from a sidecar. FORGE's `.probed_checkpoints` records an intention and can
    disagree with what happened; the ledger is the same record the result is read from."""
    assert completed_hashes(ledger) == set()

    run = ledger.start(name="x", config={}, config_hash="a" * 64, seed=1)
    assert completed_hashes(ledger) == set(), "a running run is not a completed one"

    run.finish(RunStatus.COMPLETED)
    assert completed_hashes(ledger) == {"a" * 64}


def test_a_crashed_run_is_not_treated_as_done(ledger: RunLedger) -> None:
    run = ledger.start(name="x", config={}, config_hash="b" * 64, seed=1)
    run.finish(RunStatus.FAILED, error="boom")
    assert completed_hashes(ledger) == set()


def test_a_completed_cell_is_skipped_on_reinvocation(ledger: RunLedger) -> None:
    cells = expand(parse_axes(["task.folds=2,3"]))
    first = run_matrix("spine_baseline", cells, ledger=ledger)
    assert first.completed == 2 and first.skipped == 0

    second = run_matrix("spine_baseline", cells, ledger=ledger)
    assert second.skipped == 2 and second.completed == 0


def test_killing_a_sweep_mid_flight_resumes_exactly_where_it_stopped(
    ledger: RunLedger,
) -> None:
    """The plan's headline check for this phase.

    The first invocation dies on cell 2 of 4. The second must run only the two that never
    finished, and must not touch the two that did.
    """
    cells = expand(parse_axes(["task.folds=2,3,4,5"]))

    class Boom(RuntimeError):
        pass

    seen: list[int] = []
    original = run_matrix

    def crashing_cell(cell) -> None:  # type: ignore[no-untyped-def]
        seen.append(cell.index)
        if cell.index == 2 and cell.status != "skipped":
            raise Boom("killed")

    with pytest.raises(Boom):
        original("spine_baseline", cells, ledger=ledger, on_cell=crashing_cell)

    completed_first = completed_hashes(ledger)
    assert len(completed_first) == 3, "cells 0, 1 and the one that crashed after finishing"

    report = original("spine_baseline", cells, ledger=ledger)
    assert report.skipped == 3
    assert report.completed == 1
    assert report.ok


def test_a_dry_run_executes_nothing(ledger: RunLedger) -> None:
    cells = expand(parse_axes(["task.folds=2,3"]))
    report = run_matrix("spine_baseline", cells, ledger=ledger, dry_run=True)
    assert all(cell.status == "pending" for cell in report.cells)
    assert ledger.list_runs() == []


def test_a_dry_run_still_reports_what_would_be_skipped(ledger: RunLedger) -> None:
    cells = expand(parse_axes(["task.folds=2,3"]))
    run_matrix("spine_baseline", cells[:1], ledger=ledger)
    report = run_matrix("spine_baseline", cells, ledger=ledger, dry_run=True)
    assert [cell.status for cell in report.cells] == ["skipped", "pending"]


# --- failures ---------------------------------------------------------------------------------


def test_one_failing_cell_does_not_stop_the_sweep(ledger: RunLedger) -> None:
    """One bad cell out of two hundred should cost one cell."""
    cells = expand(parse_axes(["task.estimator=logreg,not_an_estimator,random_forest"]))
    report = run_matrix("spine_baseline", cells, ledger=ledger)
    assert report.completed == 2
    assert len(report.failed) == 1
    assert report.ok is False


def test_a_failure_is_recorded_not_swallowed(ledger: RunLedger) -> None:
    """A sweep that quietly reports success while a fifth of it failed is worse than one
    that stops."""
    cells = expand(parse_axes(["task.estimator=not_an_estimator"]))
    report = run_matrix("spine_baseline", cells, ledger=ledger)
    failure = report.failed[0]
    assert "not_an_estimator" in (failure.error or "")
    assert report.as_dict()["ok"] is False


def test_fail_fast_stops_at_the_first_failure(ledger: RunLedger) -> None:
    cells = expand(parse_axes(["task.estimator=not_an_estimator,logreg"]))
    with pytest.raises(Exception):
        run_matrix("spine_baseline", cells, ledger=ledger, fail_fast=True)
    assert completed_hashes(ledger) == set()


def test_a_failed_cell_is_retried_on_the_next_invocation(ledger: RunLedger) -> None:
    """A failed cell is not a finished one; the usual reason it failed is a bug since fixed."""
    bad = expand(parse_axes(["task.estimator=not_an_estimator"]))
    run_matrix("spine_baseline", bad, ledger=ledger)
    again = run_matrix("spine_baseline", bad, ledger=ledger)
    assert again.skipped == 0
    assert len(again.failed) == 1


# --- reporting ------------------------------------------------------------------------------------


def test_the_report_counts_every_cell(ledger: RunLedger) -> None:
    cells = expand(parse_axes(["task.folds=2,3,4"]))
    report = run_matrix("spine_baseline", cells, ledger=ledger)
    assert report.total == 3
    assert report.completed + report.skipped + len(report.failed) == 3
    assert all(cell.run_id for cell in report.cells if cell.status == "completed")


def test_cells_carry_their_coordinates(ledger: RunLedger) -> None:
    cells = expand(parse_axes(["task.folds=2,3"]))
    report = run_matrix("spine_baseline", cells, ledger=ledger, dry_run=True)
    assert report.cells[0].label == "task.folds=2"
    assert report.as_dict()["cells"][0]["label"] == "task.folds=2"
