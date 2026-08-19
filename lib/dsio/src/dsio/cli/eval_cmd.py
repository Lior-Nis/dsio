"""``dsio eval`` — read evaluation reports and judge results against the noise floor.

The commands here exist so the question "is this improvement real?" has an answer that does
not depend on who is asking or which notebook they opened. Every one of them reports the
floor alongside the delta, because a delta on its own is the thing that misleads.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

from dsio.cli.envelope import json_command
from dsio.eval import (
    CVReport,
    Outcome,
    blocks_from_reports,
    compare,
    minimum_detectable_rows,
    read_report,
    sampling_noise,
    select,
)
from dsio.runs import RunLedger
from dsio.runs.record import ARTIFACTS_DIR

app = typer.Typer(help="Read evaluation reports and judge results.", no_args_is_help=True)


@app.command("show")
@json_command
def show(
    run_id: Annotated[str, typer.Argument(help="Run identifier.")],
    folds: Annotated[bool, typer.Option("--folds", help="Include per-fold detail.")] = False,
) -> dict[str, Any]:
    """Show a run's evaluation report: pooled metrics, fold spread, coverage."""
    report = _load(run_id)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "n_folds": report.n_folds,
        "n_rows": report.n_rows,
        "predicted_rows": report.predicted_rows,
        "coverage": report.coverage,
        "fold_fingerprint": report.fold_fingerprint,
        "metrics": report.metrics,
        "per_fold_mean": report.per_fold_mean,
        "per_fold_std": report.per_fold_std,
        "summary": report.summary_lines(),
    }
    if folds:
        payload["folds"] = [fold.model_dump(mode="json") for fold in report.folds]
    return payload


@app.command("verdict")
@json_command
def verdict_cmd(
    run_id: Annotated[str, typer.Argument(help="The run to evaluate.")],
    baseline: Annotated[str, typer.Option("--baseline", help="Run to compare against.")],
    metric: Annotated[str, typer.Option(help="Metric to judge.")] = "accuracy",
    higher_is_better: Annotated[bool, typer.Option(help="Direction of improvement.")] = True,
    k: Annotated[float, typer.Option(help="Noise floor in standard deviations.")] = 1.0,
    any_folds: Annotated[
        bool,
        typer.Option("--any-folds", help="Allow comparison across different fold assignments."),
    ] = False,
) -> dict[str, Any]:
    """Judge one run against a baseline, filtered by the noise floor.

    Refuses two runs whose fold assignments differ, unless ``--any-folds`` is passed. The
    fold assignment is the single source of truth; a delta measured across different folds
    measures the folds, not the models.
    """
    comparison = compare(
        _load(run_id),
        _load(baseline),
        metric=metric,
        higher_is_better=higher_is_better,
        k=k,
        require_same_folds=not any_folds,
    )
    payload = comparison.model_dump(mode="json")
    payload["summary"] = comparison.summary_line()
    payload["candidate_run"] = run_id
    payload["baseline_run"] = baseline
    return payload


@app.command("rank")
@json_command
def rank(
    metric: Annotated[str, typer.Option(help="Metric to rank on.")] = "accuracy",
    higher_is_better: Annotated[bool, typer.Option(help="Direction of improvement.")] = True,
    name: Annotated[str | None, typer.Option(help="Only runs with this name.")] = None,
    limit: Annotated[int, typer.Option(help="Maximum runs to return.")] = 20,
    baseline: Annotated[
        str | None, typer.Option("--baseline", help="Judge every run against this one.")
    ] = None,
) -> dict[str, Any]:
    """Rank runs by a metric, and — with a baseline — say which differences are real.

    Ranking without a baseline is deliberately just an ordering. The top of a leaderboard
    of 200 ablations is the maximum of 200 draws, which is a different thing from the best
    model; selection under multiplicity is what turns an ordering into a claim.
    """
    ledger = RunLedger()
    records = ledger.list_runs()
    if name:
        records = [record for record in records if record.name == name]

    reference = _load(baseline) if baseline else None
    rows: list[dict[str, Any]] = []
    for record in records:
        report = _try_load(ledger, record.run_id)
        if report is None or metric not in report.metrics:
            continue
        row: dict[str, Any] = {
            "run_id": record.run_id,
            "name": record.name,
            "value": report.metrics[metric],
            "fold_sd": report.per_fold_std.get(metric),
            "coverage": report.coverage,
            "dirty": record.git.dirty,
        }
        if reference is not None and record.run_id != baseline:
            comparison = compare(
                report,
                reference,
                metric=metric,
                higher_is_better=higher_is_better,
                k=1.0,
                require_same_folds=False,
            )
            row["outcome"] = str(comparison.outcome)
            row["improvement"] = comparison.improvement
            row["noise_floor"] = comparison.noise_floor
            row["comparable"] = comparison.method == "paired"
        rows.append(row)

    rows.sort(key=lambda item: item["value"], reverse=higher_is_better)
    payload: dict[str, Any] = {
        "metric": metric,
        "count": len(rows),
        "runs": rows[:limit],
    }
    if reference is not None:
        payload["baseline_run"] = baseline
        payload["wins"] = sum(1 for row in rows if row.get("outcome") == Outcome.WIN)
        payload["note"] = (
            "A ranking is an ordering, not a claim. With many candidates the top score is "
            "the maximum of many draws; treat individual verdicts as the evidence."
        )
    return payload


@app.command("noise")
@json_command
def noise(
    n_rows: Annotated[int, typer.Option(help="Size of the evaluation set.")] = 0,
    delta: Annotated[
        float, typer.Option(help="Improvement you hope to detect.")
    ] = 0.0,
    p: Annotated[float, typer.Option(help="Base rate of the metric.")] = 0.5,
) -> dict[str, Any]:
    """How small a difference an evaluation set of this size can actually resolve.

    The question worth asking before running the experiment rather than after. Fold
    agreement says nothing about whether the evaluation set was ever large enough: at a 0.5
    base rate, resolving a 0.001 accuracy difference takes a quarter of a million rows.
    """
    payload: dict[str, Any] = {"p": p}
    if n_rows:
        sigma = sampling_noise(n_rows, p)
        payload["n_rows"] = n_rows
        payload["sampling_sigma"] = sigma
        payload["resolvable_delta"] = sigma
    if delta:
        payload["delta"] = delta
        payload["rows_needed"] = minimum_detectable_rows(delta, p)
    if not n_rows and not delta:
        raise ValueError("pass --n-rows, --delta, or both")
    if n_rows and delta:
        payload["detectable"] = delta >= payload["sampling_sigma"]
    return payload


def _load(run_id: str) -> CVReport:
    report, _ = read_report(RunLedger().load(run_id).dir / ARTIFACTS_DIR)
    return report


def _try_load(ledger: RunLedger, run_id: str) -> CVReport | None:
    try:
        report, _ = read_report(ledger.load(run_id).dir / ARTIFACTS_DIR)
    except (FileNotFoundError, ValueError):
        return None
    return report


@app.command("select")
@json_command
def select_cmd(
    metric: Annotated[str, typer.Option(help="Metric to select on.")] = "accuracy",
    name: Annotated[str | None, typer.Option(help="Only runs with this name.")] = None,
    alpha: Annotated[float, typer.Option(help="Significance level for the search.")] = 0.05,
    pbo_threshold: Annotated[
        float, typer.Option(help="Maximum tolerable probability of overfitting.")
    ] = 0.25,
    fold_sd: Annotated[
        float | None, typer.Option(help="Per-config spread. Defaults to the spread across runs.")
    ] = None,
) -> dict[str, Any]:
    """Judge the best run against the search that produced it.

    A ranking is an ordering; this is what turns one into a claim, or refuses to. Three
    gates: the winner must beat what luck alone reaches across this many trials, the search
    must be unlikely enough under the null, and the ranking must transfer out of sample.

    The most useful answer is often the negative one — "a sweep of 200 configurations found
    nothing that survives the correction" — and a system that cannot say that will always
    hand back a winner.
    """
    ledger = RunLedger()
    records = ledger.list_runs()
    if name:
        records = [record for record in records if record.name == name]

    scores: dict[str, float] = {}
    reports: dict[str, CVReport] = {}
    for record in records:
        report = _try_load(ledger, record.run_id)
        if report is not None and metric in report.metrics:
            scores[record.run_id] = report.metrics[metric]
            reports[record.run_id] = report

    if len(scores) < 2:
        raise ValueError(
            f"need at least two runs recording {metric!r} to select between; found "
            f"{len(scores)}"
        )

    blocks = blocks_from_reports(reports, metric)
    usable = blocks if len(blocks) == len(scores) else None
    result = select(
        scores,
        blocks=usable,
        fold_sd=fold_sd,
        alpha=alpha,
        pbo_threshold=pbo_threshold,
    )
    payload = result.model_dump(mode="json")
    payload["metric"] = metric
    payload["summary"] = result.summary_lines()
    payload["n_runs"] = len(scores)
    if usable is None:
        payload["note"] = (
            "not every run reports per-fold scores for this metric, so the transfer test "
            "was skipped and the verdict is provisional"
        )
    return payload
