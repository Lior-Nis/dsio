"""``dsio splits`` — inspect and prove split files.

A split is provenance: a result has to state which groups were held out, and these YAML
files are that statement, diffable and reviewable in git. Generating them is a project
concern — an offline script whose output gets committed — so this surface only reads them:
``show`` reads one without parsing, and ``check`` proves the property they exist to
guarantee.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from dsio.cli.envelope import json_command
from dsio.data import SignalExamples, SignalStore, WindowSpec, build_index
from dsio.splits import (
    SplitFile,
    assert_no_row_overlap,
    fold_paths,
    folds_from_splits,
    resolve,
    summarise,
)

app = typer.Typer(help="Inspect and verify leakage-safe splits.", no_args_is_help=True)

SPLITS_ROOT = Path("splits")


@app.command("show")
@json_command
def show(
    path: Annotated[Path, typer.Argument(help="Path to a split YAML.")],
    groups: Annotated[
        bool, typer.Option("--groups", help="Include the full group lists.")
    ] = False,
) -> dict[str, Any]:
    """Describe one split file."""
    split = SplitFile.load(path)
    payload: dict[str, Any] = {
        "name": split.name,
        "fold": split.fold,
        "store": split.store,
        "group_key": split.group_key,
        "counts": split.counts,
        "parts": {part: len(members) for part, members in split.parts.items()},
        "temporal": None if split.temporal is None else split.temporal.model_dump(mode="json"),
    }
    if groups:
        payload["group_lists"] = split.parts
    return payload


@app.command("check")
@json_command
def check(
    store: Annotated[Path, typer.Argument(help="Path to a store directory.")],
    name: Annotated[str, typer.Option(help="Split family to check.")],
    length: Annotated[int, typer.Option(help="Window length in rows.")] = 500,
    stride: Annotated[int, typer.Option(help="Rows between window starts.")] = 250,
    root: Annotated[Path, typer.Option(help="Where the split files live.")] = SPLITS_ROOT,
    prove_rows: Annotated[
        bool,
        typer.Option(
            "--prove-rows/--no-prove-rows",
            help="Materialise every covered row to prove no overlap. Expensive.",
        ),
    ] = True,
) -> dict[str, Any]:
    """Prove that a split family actually holds its guarantee.

    Three things are checked, and they are different properties:

    1. every fold resolves against this store and assigns every group present;
    2. no window is tested by two folds, which would double-count in a pooled metric;
    3. no raw row appears in two parts of any fold.

    The third is the structural guarantee — windows never cross an entity boundary, every
    entity has one group, parts are disjoint over groups — so it cannot fail if the first
    two hold. It is verified anyway, because a guarantee nobody checks is a guarantee
    nobody notices losing. It is O(windows x length), hence ``--no-prove-rows`` for very
    large corpora.
    """
    signal = SignalStore(store)
    index = build_index(signal, WindowSpec(length=length, stride=stride))
    paths = fold_paths(root, name)
    splits = [SplitFile.load(path) for path in paths]

    # Building the folds is itself the cross-fold disjointness check.
    examples = SignalExamples(signal, index)
    folds = folds_from_splits(examples, splits, require_total=False)

    per_fold: list[dict[str, Any]] = []
    for split, fold in zip(splits, folds, strict=True):
        parts = resolve(examples, split, require_total=False)
        if prove_rows:
            assert_no_row_overlap(parts)
        per_fold.append(
            {
                "fold": fold.index,
                "name": fold.name,
                "sizes": fold.sizes,
                "parts": summarise(parts),
            }
        )

    tested = sum(fold.test.size for fold in folds)
    return {
        "store": signal.path.name,
        "name": name,
        "windows": len(index),
        "folds": len(folds),
        "ok": True,
        "rows_proved_disjoint": prove_rows,
        "tested_windows": tested,
        "coverage": tested / len(index) if len(index) else 0.0,
        "per_fold": per_fold,
    }
