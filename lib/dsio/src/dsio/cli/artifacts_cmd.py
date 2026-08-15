"""``dsio registry`` — list, inspect, and promote model artifacts.

Promotion is the one gate in the system. Exploration is never blocked, so a dirty tree
runs freely; but a registered model outlives the session that produced it, so it must
name a commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from dsio.artifacts import ModelRegistry, promotion_blockers
from dsio.cli.envelope import BlockedError, json_command
from dsio.runs import RunLedger

app = typer.Typer(help="Versioned model artifacts.", no_args_is_help=True)


@app.command("list")
@json_command
def list_models() -> dict[str, Any]:
    """List registered model names and their version counts."""
    registry = ModelRegistry()
    return {
        "models": {
            name: len(registry.versions(name)) for name in registry.names()
        }
    }


@app.command("show")
@json_command
def show(name: Annotated[str, typer.Argument(help="Model name.")]) -> dict[str, Any]:
    """Show every version of one model."""
    versions = ModelRegistry().versions(name)
    if not versions:
        raise FileNotFoundError(f"no model named {name!r} in the registry")
    return {
        "name": name,
        "versions": [version.model_dump(mode="json") for version in versions],
    }


@app.command("promote")
@json_command
def promote(
    run_id: Annotated[str, typer.Argument(help="Run whose artifact to register.")],
    name: Annotated[str, typer.Option(help="Name to register the model under.")],
    artifact: Annotated[
        str, typer.Option(help="Artifact filename inside the run's artifacts dir.")
    ] = "model.pkl",
) -> dict[str, Any]:
    """Register a run's artifact as a pinned model version.

    Refuses runs that cannot be reconstructed. This is the only place dsio says no.
    """
    run = RunLedger().load(run_id)
    record = run.record

    blockers = promotion_blockers(record)
    if blockers:
        raise BlockedError(
            f"run {run_id} cannot be promoted: {'; '.join(blockers)}"
        )

    path = Path(run.dir) / "artifacts" / artifact
    if not path.is_file():
        raise FileNotFoundError(f"run {run_id} has no artifact at {path}")

    version = ModelRegistry().save(
        name,
        path.read_bytes(),
        run_id=record.run_id,
        config_hash=record.config_hash,
        code_hash=record.git.code_hash,
        seed=record.seed,
        metrics=record.metrics,
    )
    return {"ref": str(version.ref), "version": version.model_dump(mode="json")}
