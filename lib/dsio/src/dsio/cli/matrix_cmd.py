"""``dsio matrix`` and ``dsio search`` — sweeps that resume.

Both emit ordinary Runs into the same ledger, so a swept result and a hand-run one are read
and compared by exactly the same commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from dsio.cli.envelope import json_command
from dsio.config import load_preset_modules
from dsio.matrix import (
    expand,
    parse_axes,
    parse_space,
    run_matrix,
    run_search,
    size,
    validate,
)
from dsio.runs import RunLedger
from dsio.train import load_runners

app = typer.Typer(help="Run sweeps and searches.", no_args_is_help=True)


def _bootstrap() -> None:
    load_runners()
    load_preset_modules()


@app.command("matrix")
@json_command
def matrix(
    preset: Annotated[str, typer.Argument(help="Name of a registered preset.")],
    axis: Annotated[
        list[str] | None,
        typer.Option(
            "--axis",
            "-a",
            help="path=v1,v2,v3 or path=glob:pattern. Repeatable; the product is swept.",
        ),
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override applied to every cell.", show_default=False),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List the cells and what would be skipped.")
    ] = False,
    fail_fast: Annotated[
        bool, typer.Option("--fail-fast", help="Stop at the first failing cell.")
    ] = False,
    max_cells: Annotated[
        int, typer.Option(help="Refuse a sweep larger than this. 0 disables the guard.")
    ] = 500,
    summary: Annotated[
        bool, typer.Option("--summary", help="Omit the per-cell detail.")
    ] = False,
) -> dict[str, Any]:
    """Sweep the cross product of the given axes, skipping cells that already completed.

    A cell's identity is the sha256 of its resolved config, so re-invoking after a crash
    runs only what is missing — and the resume state is the run ledger itself, never a
    sidecar that could disagree with it.
    """
    _bootstrap()
    axes = parse_axes(axis or [])
    planned = size(axes)
    if max_cells and planned > max_cells:
        raise ValueError(
            f"this sweep has {planned} cells, above the {max_cells} guard. A sweep that "
            "would take a week should be refused before it starts, not after; raise "
            "--max-cells if that is genuinely what you want."
        )

    cells = expand(axes, base=tuple(set_ or []))
    # Resolving every cell now turns a typo in one axis into a parse error rather than
    # into 200 runs that each fail after loading a corpus.
    validate(preset, cells)

    report = run_matrix(
        preset,
        cells,
        ledger=RunLedger(),
        fail_fast=fail_fast,
        dry_run=dry_run,
        command=("dsio", "matrix", preset),
    )
    payload = report.as_dict()
    payload["axes"] = [{"path": item.path, "values": list(item.values)} for item in axes]
    if summary:
        payload.pop("cells", None)
    if not report.ok:
        payload["failures"] = [cell.as_dict() for cell in report.failed]
    return payload


@app.command("search")
@json_command
def search(
    preset: Annotated[str, typer.Argument(help="Name of a registered preset.")],
    space: Annotated[
        list[str] | None,
        typer.Option(
            "--space",
            "-s",
            help="path=loguniform(lo,hi) | uniform(lo,hi) | int(lo,hi) | categorical(a,b).",
        ),
    ] = None,
    metric: Annotated[str, typer.Option(help="Metric to optimise.")] = "accuracy",
    direction: Annotated[str, typer.Option(help="maximize or minimize.")] = "maximize",
    n_trials: Annotated[int, typer.Option(help="How many trials to run.")] = 20,
    seed: Annotated[int, typer.Option(help="Sampler seed.")] = 42,
    set_: Annotated[
        list[str] | None, typer.Option("--set", help="Override applied to every trial.")
    ] = None,
    storage: Annotated[
        Path | None,
        typer.Option(help="sqlite file for the study, which makes the search resumable."),
    ] = None,
    summary: Annotated[
        bool, typer.Option("--summary", help="Omit the per-trial detail.")
    ] = False,
) -> dict[str, Any]:
    """Search for the configuration that optimises a metric.

    Each trial is an ordinary Run in the same ledger. The distribution is named explicitly
    rather than inferred: a learning rate searched uniformly between 1e-5 and 1e-3 spends
    90% of its trials above 1e-4, and nothing in the numbers reveals which was meant.
    """
    _bootstrap()
    spaces = [parse_space(token) for token in (space or [])]
    report = run_search(
        preset,
        spaces,
        metric=metric,
        direction=direction,
        n_trials=n_trials,
        seed=seed,
        base=tuple(set_ or []),
        ledger=RunLedger(),
        storage=storage,
        command=("dsio", "search", preset),
    )
    payload = report.as_dict()
    if summary:
        payload.pop("trials", None)
    return payload
