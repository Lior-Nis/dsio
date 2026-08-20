"""``dsio splits`` — generate, inspect and prove split files.

A split is provenance: a result has to state which groups were held out, and these YAML
files are that statement, diffable and reviewable in git. The commands are shaped around
that — ``make`` writes them, ``show`` reads one without parsing, and ``check`` proves the
property they exist to guarantee.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from dsio.cli.envelope import json_command
from dsio.data import SignalExamples, SignalStore, WindowSpec, build_index, entity_examples
from dsio.splits import (
    SplitFile,
    SplitSpec,
    TemporalSpec,
    assert_no_row_overlap,
    fold_paths,
    folds_from_splits,
    resolve,
    summarise,
    write_splits,
    write_temporal_splits,
)

app = typer.Typer(help="Generate and verify leakage-safe splits.", no_args_is_help=True)

SPLITS_ROOT = Path("splits")


@app.command("make")
@json_command
def make(
    store: Annotated[Path, typer.Argument(help="Path to a store directory.")],
    name: Annotated[str, typer.Option(help="Name for this split family.")],
    scheme: Annotated[str, typer.Option(help="kfold, holdout, leave_one_group_out, explicit")] = (
        "kfold"
    ),
    k: Annotated[int, typer.Option(help="Number of folds.")] = 5,
    seed: Annotated[int, typer.Option(help="Assignment seed.")] = 42,
    always_train: Annotated[
        list[str] | None,
        typer.Option("--always-train", help="Groups pinned to train. Repeatable."),
    ] = None,
    root: Annotated[Path, typer.Option(help="Where to write the files.")] = SPLITS_ROOT,
) -> dict[str, Any]:
    """Generate split files and write them for committing.

    The output is meant to be reviewed and committed. Regenerating with the same spec and
    seed reproduces it exactly, but the committed file is what a result cites — reading a
    published number should never require re-running a generator.
    """
    signal = SignalStore(store)
    spec = SplitSpec(
        scheme=scheme,  # type: ignore[arg-type]
        k=k,
        seed=seed,
        always_train=tuple(always_train or []),
    )
    paths = write_splits(entity_examples(signal), spec, name=name, root=root)
    files = [SplitFile.load(path) for path in paths]
    return {
        "store": signal.path.name,
        "name": name,
        "spec": spec.model_dump(mode="json"),
        "count": len(paths),
        "paths": [str(path) for path in paths],
        "folds": [
            {"fold": file.fold, "counts": file.counts}
            for file in files
        ],
    }


@app.command("make-temporal")
@json_command
def make_temporal(
    store: Annotated[Path, typer.Argument(help="Path to a store directory.")],
    name: Annotated[str, typer.Option(help="Name for this split family.")],
    length: Annotated[int, typer.Option(help="Window length in rows.")] = 500,
    stride: Annotated[int, typer.Option(help="Rows between window starts.")] = 250,
    n_splits: Annotated[int, typer.Option(help="Walk-forward folds.")] = 5,
    test_fraction: Annotated[float, typer.Option(help="Test span as a fraction.")] = 0.2,
    label_horizon: Annotated[
        float, typer.Option(help="How far past its end a label resolves. Drives purging.")
    ] = 0.0,
    embargo: Annotated[float, typer.Option(help="Gap after test before training resumes.")] = 0.0,
    root: Annotated[Path, typer.Option(help="Where to write the files.")] = SPLITS_ROOT,
) -> dict[str, Any]:
    """Generate purged, embargoed walk-forward splits.

    ``label_horizon`` and ``embargo`` are not optional refinements. A training window whose
    label resolves inside the test period has seen it, and the window immediately after the
    test period leaks the same autocorrelated regime. The band between is discarded and
    belongs to neither part.
    """
    signal = SignalStore(store)
    index = build_index(signal, WindowSpec(length=length, stride=stride))
    spec = TemporalSpec(
        n_splits=n_splits,
        test_fraction=test_fraction,
        label_horizon=label_horizon,
        embargo=embargo,
    )
    paths = write_temporal_splits(SignalExamples(signal, index), spec, name=name, root=root)
    return {
        "store": signal.path.name,
        "name": name,
        "spec": spec.model_dump(mode="json"),
        "count": len(paths),
        "paths": [str(path) for path in paths],
        "windows": len(index),
    }


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
