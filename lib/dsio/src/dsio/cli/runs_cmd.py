"""``dsio runs`` — inspect and compare the ledger.

Every listing command has a projected form. algua had to retrofit ``--summary`` onto its
sweep output because full JSON overwhelmed an agent's context window; shipping the
projection from the start is cheaper than discovering the need at 200 runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from dsio.cli.envelope import json_command
from dsio.eval import CVReport, EvalError, read_report
from dsio.eval import compare as eval_compare
from dsio.runs import RunLedger
from dsio.runs.record import ARTIFACTS_DIR

app = typer.Typer(help="Inspect the run ledger.", no_args_is_help=True)


@app.command("list")
@json_command
def list_runs(
    limit: Annotated[int, typer.Option(help="Maximum runs to return.")] = 20,
    name: Annotated[str | None, typer.Option(help="Only runs with this name.")] = None,
) -> dict[str, Any]:
    """List runs, newest first."""
    records = RunLedger().list_runs()
    if name:
        records = [record for record in records if record.name == name]
    return {
        "count": len(records),
        "runs": [
            {
                "run_id": record.run_id,
                "name": record.name,
                "status": str(record.status),
                "created_at": record.created_at,
                "config_hash": record.config_hash[:12],
                "dirty": record.git.dirty,
                "metrics": record.metrics,
            }
            for record in records[:limit]
        ],
    }


@app.command("show")
@json_command
def show(
    run_id: Annotated[str, typer.Argument(help="Run identifier.")],
    summary: Annotated[
        bool, typer.Option("--summary", help="Omit the resolved config.")
    ] = False,
) -> dict[str, Any]:
    """Show one run's full record."""
    record = RunLedger().load(run_id).record
    payload = record.model_dump(mode="json")
    if summary:
        payload.pop("config", None)
    payload["reproducible"] = record.reproducible
    return payload


@app.command("compare")
@json_command
def compare(
    run_id: Annotated[str, typer.Argument(help="The run to evaluate.")],
    baseline: Annotated[str, typer.Option("--baseline", help="Run to compare against.")],
    metric: Annotated[str, typer.Option(help="Metric to compare.")] = "accuracy",
    higher_is_better: Annotated[bool, typer.Option(help="Direction of improvement.")] = True,
    k: Annotated[float, typer.Option(help="Noise floor in standard deviations.")] = 1.0,
    any_folds: Annotated[
        bool,
        typer.Option(
            "--any-folds",
            help="Compare even when the two runs used different fold assignments.",
        ),
    ] = False,
) -> dict[str, Any]:
    """Compare two runs on one metric, with a verdict filtered by the noise floor.

    The delta alone is the thing that misleads, so it is never reported without a floor to
    judge it against. Where both runs wrote an evaluation report the verdict is computed
    from fold spread — paired across folds when the two runs provably held out the same
    rows, which makes a real improvement visible under fold-to-fold variation that would
    otherwise bury it.

    The config diff matters as much as the delta: a metric change is only attributable if
    you know which knob moved.
    """
    ledger = RunLedger()
    candidate_run = ledger.load(run_id)
    baseline_run = ledger.load(baseline)
    candidate, reference = candidate_run.record, baseline_run.record

    candidate_report = _report_for(candidate_run.dir)
    baseline_report = _report_for(baseline_run.dir)

    if candidate_report is not None and baseline_report is not None:
        comparison = eval_compare(
            candidate_report,
            baseline_report,
            metric=metric,
            higher_is_better=higher_is_better,
            k=k,
            require_same_folds=not any_folds,
        )
        verdict_payload = comparison.model_dump(mode="json")
        verdict_payload["summary"] = comparison.summary_line()
    else:
        if metric not in candidate.metrics or metric not in reference.metrics:
            missing = [
                run.run_id for run in (candidate, reference) if metric not in run.metrics
            ]
            raise ValueError(f"metric {metric!r} is not recorded on: {', '.join(missing)}")
        delta = candidate.metrics[metric] - reference.metrics[metric]
        improvement = delta if higher_is_better else -delta
        verdict_payload = {
            "metric": metric,
            "outcome": "unknown",
            "candidate": candidate.metrics[metric],
            "baseline": reference.metrics[metric],
            "improvement": improvement,
            "noise_floor": None,
            "method": "none",
            "reason": (
                "one or both runs have no evaluation report, so there is no fold spread "
                "to judge this delta against; it may be entirely noise"
            ),
        }

    return {
        "metric": metric,
        "candidate_run": candidate.run_id,
        "baseline_run": reference.run_id,
        "verdict": verdict_payload,
        "config_diff": _diff(reference.config, candidate.config),
    }


def _report_for(directory: Path) -> CVReport | None:
    """Read a run's evaluation report, or ``None`` if it never wrote one.

    Absence is normal — a run that crashed before scoring, or an older run — and must
    degrade to an honest "unknown" rather than to a delta presented as if it meant
    something.
    """
    try:
        report, _ = read_report(directory / ARTIFACTS_DIR)
    except EvalError:
        return None
    return report


def _diff(before: dict[str, Any], after: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Return the leaf-level differences between two config mappings."""
    changes: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        path = f"{prefix}.{key}" if prefix else key
        left, right = before.get(key), after.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            changes.update(_diff(left, right, path))
        elif left != right:
            changes[path] = {"baseline": left, "candidate": right}
    return changes
