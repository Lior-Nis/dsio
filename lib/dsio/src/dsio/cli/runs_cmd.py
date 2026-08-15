"""``dsio runs`` — inspect and compare the ledger.

Every listing command has a projected form. algua had to retrofit ``--summary`` onto its
sweep output because full JSON overwhelmed an agent's context window; shipping the
projection from the start is cheaper than discovering the need at 200 runs.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

from dsio.cli.envelope import json_command
from dsio.runs import RunLedger

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
) -> dict[str, Any]:
    """Compare two runs on one metric, and report what actually differs between them.

    The config diff matters as much as the delta: a metric change is only attributable if
    you know which knob moved. A full noise-floor verdict over fold spread arrives with
    ``dsio.eval`` in phase 3 — until then this reports the delta without claiming
    significance, because an unqualified delta is exactly the thing that misleads.
    """
    ledger = RunLedger()
    candidate = ledger.load(run_id).record
    reference = ledger.load(baseline).record

    if metric not in candidate.metrics or metric not in reference.metrics:
        missing = [
            run.run_id for run in (candidate, reference) if metric not in run.metrics
        ]
        raise ValueError(f"metric {metric!r} is not recorded on: {', '.join(missing)}")

    delta = candidate.metrics[metric] - reference.metrics[metric]
    improved = delta > 0 if higher_is_better else delta < 0
    return {
        "metric": metric,
        "candidate": {"run_id": candidate.run_id, "value": candidate.metrics[metric]},
        "baseline": {"run_id": reference.run_id, "value": reference.metrics[metric]},
        "delta": delta,
        "improved": improved,
        "significance": "not assessed; no fold spread available",
        "config_diff": _diff(reference.config, candidate.config),
    }


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
