"""Optuna search: trials are ordinary Runs, and a bad configuration is not fatal."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("optuna")

from dsio.matrix import MatrixError, parse_space, run_search  # noqa: E402
from dsio.runs import RunLedger, RunStatus  # noqa: E402

# --- parsing the space --------------------------------------------------------------


def test_a_log_space_is_named_explicitly() -> None:
    """A learning rate searched uniformly over [1e-5, 1e-3] spends 90% of its trials above
    1e-4 — a silent and expensive mistake that nothing in the numbers reveals."""
    space = parse_space("task.lr=loguniform(1e-5,1e-3)")
    assert space.kind == "loguniform"
    assert space.low == pytest.approx(1e-5)
    assert space.high == pytest.approx(1e-3)


def test_every_distribution_parses() -> None:
    assert parse_space("a=uniform(0,1)").kind == "uniform"
    assert parse_space("a=int(2,8)").kind == "int"
    assert parse_space("a=categorical(x,y,z)").choices == ("x", "y", "z")


def test_an_unknown_distribution_is_rejected() -> None:
    with pytest.raises(MatrixError, match="unknown distribution"):
        parse_space("a=magic(1,2)")


def test_a_malformed_space_is_rejected() -> None:
    with pytest.raises(MatrixError, match="path=kind"):
        parse_space("a=1,2")


def test_an_inverted_range_is_rejected() -> None:
    with pytest.raises(MatrixError, match="low .* >= high"):
        parse_space("a=uniform(1,0)")


def test_a_categorical_with_no_choices_is_rejected() -> None:
    with pytest.raises(MatrixError, match="no choices"):
        parse_space("a=categorical()")


def test_a_range_distribution_needs_exactly_two_bounds() -> None:
    with pytest.raises(MatrixError, match="exactly low and high"):
        parse_space("a=uniform(1,2,3)")


# --- searching -------------------------------------------------------------------------


def test_a_search_produces_one_run_per_distinct_config(ledger: RunLedger) -> None:
    """Same ledger, same config hash, same artifact contract — so a searched result and a
    hand-run one are compared by the same machinery.

    One run per *distinct config*, not per trial: a sampler that revisits a point reads the
    recorded metric instead of retraining an identical configuration.
    """
    report = run_search(
        "spine_baseline",
        [parse_space("task.folds=int(2,4)")],
        metric="accuracy",
        n_trials=6,
        ledger=ledger,
    )
    assert len(report.trials) == 6
    records = ledger.list_runs()
    assert len(records) == len({trial.config_hash for trial in report.trials})
    assert all(record.status is RunStatus.COMPLETED for record in records)
    assert all("search" in record.tags for record in records)


def test_a_revisited_point_is_reused_not_retrained(ledger: RunLedger) -> None:
    """int(2,3) has two possible configs, so six trials must produce two runs."""
    report = run_search(
        "spine_baseline",
        [parse_space("task.folds=int(2,3)")],
        metric="accuracy",
        n_trials=6,
        ledger=ledger,
    )
    assert len(ledger.list_runs()) == 2
    assert report.reused == 4
    assert report.ok


def test_a_search_reuses_what_a_matrix_already_computed(ledger: RunLedger) -> None:
    """The payoff of one shared identity: a search after a sweep costs only the points the
    sweep did not cover."""
    from dsio.matrix import expand, parse_axes, run_matrix

    run_matrix("spine_baseline", expand(parse_axes(["task.folds=2,3"])), ledger=ledger)
    before = len(ledger.list_runs())

    report = run_search(
        "spine_baseline",
        [parse_space("task.folds=int(2,3)")],
        metric="accuracy",
        n_trials=4,
        ledger=ledger,
    )
    assert report.reused == 4
    assert len(ledger.list_runs()) == before, "no new runs were needed"
    assert report.best_value is not None


def test_reuse_can_be_switched_off(ledger: RunLedger) -> None:
    report = run_search(
        "spine_baseline",
        [parse_space("task.folds=int(2,3)")],
        metric="accuracy",
        n_trials=4,
        ledger=ledger,
        reuse_completed=False,
    )
    assert report.reused == 0
    assert len(ledger.list_runs()) == 4


def test_the_best_trial_is_reported_with_its_run(ledger: RunLedger) -> None:
    report = run_search(
        "spine_baseline",
        [parse_space("task.folds=int(2,4)")],
        metric="accuracy",
        n_trials=4,
        ledger=ledger,
    )
    assert report.best_value is not None
    assert report.best_run_id is not None
    assert report.best_value == max(t.value for t in report.trials if t.value is not None)


def test_minimize_picks_the_lowest(ledger: RunLedger) -> None:
    report = run_search(
        "spine_baseline",
        [parse_space("task.folds=int(2,4)")],
        metric="accuracy",
        direction="minimize",
        n_trials=4,
        ledger=ledger,
    )
    assert report.best_value == min(t.value for t in report.trials if t.value is not None)


def test_a_search_is_reproducible_from_its_seed(ledger: RunLedger) -> None:
    def params(seed: int) -> list[dict]:
        return [
            trial.params
            for trial in run_search(
                "spine_baseline",
                [parse_space("task.folds=int(2,6)")],
                metric="accuracy",
                n_trials=3,
                seed=seed,
                ledger=RunLedger(ledger.root / f"s{seed}"),
            ).trials
        ]

    assert params(7) == params(7)
    assert params(7) != params(8)


def test_a_search_needs_something_to_vary(ledger: RunLedger) -> None:
    with pytest.raises(MatrixError, match="at least one parameter"):
        run_search("spine_baseline", [], metric="accuracy", ledger=ledger)


def test_a_bad_direction_is_rejected(ledger: RunLedger) -> None:
    with pytest.raises(MatrixError, match="maximize or minimize"):
        run_search(
            "spine_baseline",
            [parse_space("task.folds=int(2,3)")],
            metric="accuracy",
            direction="sideways",
            ledger=ledger,
        )


# --- failures ---------------------------------------------------------------------------


def test_a_failing_trial_does_not_end_the_search(ledger: RunLedger) -> None:
    """A search whose fifth trial hits an unusable configuration should record it and carry
    on; one that dies there has wasted the four before it."""
    report = run_search(
        "spine_baseline",
        [parse_space("task.estimator=categorical(logreg,not_an_estimator)")],
        metric="accuracy",
        n_trials=8,
        ledger=ledger,
    )
    assert report.failed, "the bad estimator should have failed at least once"
    assert any(trial.status == "completed" for trial in report.trials)
    assert report.ok is False


def test_a_missing_metric_is_a_failure_not_a_silent_zero(ledger: RunLedger) -> None:
    """Scoring a trial that never produced the metric would optimise against a default."""
    report = run_search(
        "spine_baseline",
        [parse_space("task.folds=int(2,3)")],
        metric="no_such_metric",
        n_trials=2,
        ledger=ledger,
    )
    assert len(report.failed) == 2
    assert "no_such_metric" in (report.failed[0].error or "")
    assert report.best_value is None


# --- resume -------------------------------------------------------------------------------


def test_a_study_on_disk_resumes(ledger: RunLedger, tmp_path: Path) -> None:
    """Optuna's storage remembers the points tried; the ledger remembers the runs."""
    storage = tmp_path / "study.db"
    first = run_search(
        "spine_baseline",
        [parse_space("task.folds=int(2,6)")],
        metric="accuracy",
        n_trials=3,
        ledger=ledger,
        storage=storage,
        study_name="resume-me",
    )
    second = run_search(
        "spine_baseline",
        [parse_space("task.folds=int(2,6)")],
        metric="accuracy",
        n_trials=3,
        ledger=ledger,
        storage=storage,
        study_name="resume-me",
    )
    assert storage.is_file()
    assert len(first.trials) == 3 and len(second.trials) == 3

    import optuna

    study = optuna.load_study(study_name="resume-me", storage=f"sqlite:///{storage}")
    assert len(study.trials) == 6, "the second search continued the same study"
