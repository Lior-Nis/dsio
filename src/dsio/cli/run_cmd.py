"""``dsio run`` — resolve a preset, execute it, and record everything."""

from __future__ import annotations

import sys
from typing import Annotated, Any

import typer

from dsio.cli.envelope import json_command
from dsio.config.presets import (
    PRESETS,
    STAGE_HOOKS,
    load_preset_modules,
    preset_parameters,
    resolve,
)
from dsio.config.registry import UnknownComponentError
from dsio.config.schema import RunConfig
from dsio.runs.record import RunLedger, RunStatus
from dsio.runs.seeding import seed_everything
from dsio.splits.models import SplitError
from dsio.train import load_runners
from dsio.train.runner import check, execute

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

    if dry_run:
        # Resolving must stay pure — "resolve and validate only" cannot itself write a
        # file or register a label, so staging (below) never runs here. A preset with a
        # stage hook may therefore preflight-fail purely because its prerequisites are
        # not staged *yet*; that is reported explicitly rather than as a hard failure
        # or silently passed over, so a reader sees "this would be staged first".
        pending_stage: str | None = None
        try:
            _preflight(config)
        except (UnknownComponentError, SplitError) as exc:
            if preset not in STAGE_HOOKS:
                raise
            pending_stage = (
                f"not staged yet; running for real (without --dry-run) stages it "
                f"automatically — preflight said: {exc}"
            )
        return {
            "preset": preset,
            "config_hash": config.config_hash,
            "dry_run": True,
            **({"pending_stage": pending_stage} if pending_stage else {}),
            **({} if summary else {"config": config.to_dict()}),
        }

    if preset in STAGE_HOOKS:
        STAGE_HOOKS.get(preset)(config)
    _preflight(config)

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
        # Not every task kind carries a fold (`TaskConfig` itself does not), so this
        # reaches for it defensively rather than assuming `config.task.fold` exists.
        fold=getattr(config.task, "fold", None),
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
