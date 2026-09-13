"""Application composition for configuration discovery and tracked preset execution."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dsio.config.presets import (
    PRESETS,
    STAGE_HOOKS,
    load_preset_modules,
    preset_parameters,
    resolve,
)
from dsio.config.registry import UnknownComponentError
from dsio.config.schema import RunConfig
from dsio.runs.record import RunRecord, start_run
from dsio.runs.seeding import seed_everything
from dsio.splits.models import SplitError
from dsio.train import load_runners
from dsio.train.runner import check, execute


@dataclass(frozen=True)
class DryRunResult:
    """A resolved preset that was checked without executing it."""

    preset: str
    config: RunConfig
    pending_stage: str | None = None


@dataclass(frozen=True)
class CompletedRun:
    """The durable outcome of a successfully executed preset."""

    config: RunConfig
    record: RunRecord
    mlflow_run_id: str | None
    metrics: dict[str, float]


def bootstrap() -> None:
    """Import runner modules, then preset modules, so their registrations exist."""
    load_runners()
    load_preset_modules()


def load_run_config(data: Mapping[str, Any]) -> RunConfig:
    """Validate a recorded config after loading the task types it may name."""
    load_runners()
    return RunConfig.model_validate(data)


def available_presets() -> dict[str, list[str]]:
    """Return the discovered presets and their public parameters."""
    bootstrap()
    return {
        name: [
            parameter
            for parameter in preset_parameters(name)
            if parameter not in {"args", "kwargs"}
        ]
        for name in PRESETS.names()
    }


def execute_preset(
    name: str,
    overrides: list[str],
    *,
    dry_run: bool,
    command: tuple[str, ...],
) -> DryRunResult | CompletedRun:
    """Resolve and check a preset, then optionally execute it as a tracked run."""
    bootstrap()
    config = resolve(name, overrides)

    if dry_run:
        pending_stage: str | None = None
        try:
            check(config)
        except (UnknownComponentError, SplitError) as exc:
            if name not in STAGE_HOOKS:
                raise
            pending_stage = (
                "not staged yet; running for real (without --dry-run) stages it "
                f"automatically — preflight said: {exc}"
            )
        return DryRunResult(name, config, pending_stage)

    if name in STAGE_HOOKS:
        STAGE_HOOKS.get(name)(config)
    check(config)

    seeds = seed_everything(config.seed)
    run = start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
        seeds=seeds,
        tags=config.tags,
        command=command,
        fold=getattr(config.task, "fold", None),
    )
    try:
        metrics = execute(config, run)
    finally:
        shutil.rmtree(run.dir, ignore_errors=True)

    return CompletedRun(config, run.record, run.mlflow_run_id, metrics)
