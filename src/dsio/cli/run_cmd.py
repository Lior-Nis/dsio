"""``dsio run`` — resolve a preset, execute it, and record everything."""

from __future__ import annotations

import sys
from typing import Annotated, Any

import typer

from dsio.application import (
    CompletedRun,
    DryRunResult,
    available_presets,
    execute_preset,
)
from dsio.cli.envelope import json_command


@json_command
def run(
    preset: Annotated[
        str | None, typer.Argument(help="Preset to run; omit to list them.")
    ] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Argument(help="key=value or nested.path=value overrides."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Resolve and validate only; execute nothing.")
    ] = False,
    summary: Annotated[
        bool, typer.Option("--summary", help="Omit the resolved config from the output.")
    ] = False,
) -> dict[str, Any]:
    """Resolve ``preset``, apply ``overrides``, and run it, tracked in MLflow.

    Called bare, with no preset, it lists every registered preset and the arguments it
    accepts instead of running anything.
    """
    if preset is None:
        return {"presets": available_presets()}

    result = execute_preset(
        preset,
        list(overrides or []),
        dry_run=dry_run,
        command=tuple(sys.argv),
    )
    config = result.config
    if isinstance(result, DryRunResult):
        return {
            "preset": result.preset,
            "config_hash": result.config.config_hash,
            "dry_run": True,
            **({"pending_stage": result.pending_stage} if result.pending_stage else {}),
            **({} if summary else {"config": config.to_dict()}),
        }

    assert isinstance(result, CompletedRun)
    record = result.record
    payload: dict[str, Any] = {
        "run_id": record.run_id,
        "mlflow_run_id": result.mlflow_run_id,
        "status": "completed",
        "config_hash": record.config_hash,
        "metrics": result.metrics,
        "dirty": record.git.dirty,
        "reproducible": record.reproducible,
    }
    if not summary:
        payload["config"] = config.to_dict()
    return payload
