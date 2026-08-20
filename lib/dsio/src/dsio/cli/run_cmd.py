"""``dsio run`` — resolve a preset, execute it, and record everything."""

from __future__ import annotations

import sys
from typing import Annotated, Any

import typer

from dsio.cli.envelope import json_command
from dsio.config import PRESETS, RunConfig, load_preset_modules, preset_parameters, resolve
from dsio.runs import RunLedger, RunStatus, seed_everything
from dsio.train import check, execute, load_runners

app = typer.Typer(help="Execute a preset as a tracked run.", no_args_is_help=True)


def _bootstrap() -> None:
    """Import runner and preset modules so their registrations exist."""
    load_runners()
    load_preset_modules()


@app.command("run")
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
    """Resolve ``preset``, apply ``overrides``, and run it under the ledger.

    Called bare, with no preset, it lists every registered preset and the arguments it
    accepts instead of running anything.
    """
    if preset is None:
        _bootstrap()
        return {
            "presets": {
                name: [
                    param
                    for param in preset_parameters(name)
                    if param not in {"args", "kwargs"}
                ]
                for name in PRESETS.names()
            }
        }

    _bootstrap()
    config = resolve(preset, list(overrides or []))
    _preflight(config)

    if dry_run:
        return {
            "preset": preset,
            "config_hash": config.config_hash,
            "dry_run": True,
            **({} if summary else {"config": config.to_dict()}),
        }

    seeds = seed_everything(config.seed)
    ledger = RunLedger()
    with ledger.start(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
        seeds=seeds,
        tags=config.tags,
        command=tuple(sys.argv),
    ) as active:
        metrics = execute(config, active)
        active.finish(RunStatus.COMPLETED, metrics=metrics)

    record = active.record
    payload: dict[str, Any] = {
        "run_id": record.run_id,
        "status": str(record.status),
        "config_hash": record.config_hash,
        "metrics": record.metrics,
        "dirty": record.git.dirty,
        "reproducible": record.reproducible,
    }
    if not summary:
        payload["config"] = config.to_dict()
        payload["run_dir"] = str(active.dir)
    return payload


def _preflight(config: RunConfig) -> None:
    """Fail before any data is touched if the run cannot possibly execute.

    Deferred validation surfaces errors after data loading has started, so a
    typo cost a coffee break. Resolving every name up front costs microseconds.
    """
    check(config)
