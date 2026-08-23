"""Pool the per-fold predictions that N single-fold runs wrote.

Decision 6 made the process boundary the fold boundary: `dsio run` trains one config
against one fold and writes one ``predictions.npz``. Cross-validation is running the entry
point N times. Nothing then holds all the folds at once -- so the checks the in-process
fold loop used to make while accumulating have to be made here instead, when the files are
read back.

Two of those checks are the reason this is not a `np.concatenate`:

- **A row predicted by two folds.** Either the folds overlap or a runner returned the wrong
  positions, and both make the pooled metric double-count. `SplitFile` already proves the
  *split* assigns each group to one test fold, at load time. This catches the other case:
  a correct split that a runner read wrongly.
- **Scores present for some folds and absent for others.** Pooling those computes a ranking
  metric over a mixture of probabilities and hard labels. The writer omits the ``y_score``
  key entirely rather than storing zeros, so this is decided on key presence -- a fold whose
  scores are genuinely all zero pools fine.

Each file also carries the split family and store digest it came from, so runs from
different families or different store snapshots are refused rather than silently mixed.
That binding is checked at `SplitFile` load time too, but pooling happens in a different
process than the one that validated it, so it is re-checked against what the runs recorded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dsio.eval.contract import PREDICTIONS_FILE, EvalError
from dsio.eval.metrics import MetricError, compute


@dataclass(frozen=True)
class Pooled:
    """Out-of-fold predictions from every fold of one split family, and their metrics."""

    row_id: np.ndarray
    fold: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray | None
    metrics: dict[str, float]
    folds: tuple[int, ...]
    split: str
    split_digest: str


def _read(path: Path) -> dict[str, np.ndarray]:
    file = path / PREDICTIONS_FILE if path.is_dir() else path
    if not file.is_file():
        raise EvalError(f"{file} does not exist; a fold that never ran cannot be pooled")
    with np.load(file, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def pool_folds(paths: Sequence[Path | str], *, metrics: Sequence[str]) -> Pooled:
    """Pool one ``predictions.npz`` per fold and score the result.

    ``paths`` may be run-artifact directories or the files themselves. Order does not
    matter: folds are sorted by their recorded index before concatenation, never by
    filename or argument position. Sorting lexically would give 0, 1, 10, 2 and produce a
    perfectly plausible pooled number with the folds silently permuted -- the same trap
    `dsio.splits.folds.fold_paths` documents, and no downstream assertion can detect it
    because every fold is individually valid.
    """
    if not paths:
        raise EvalError("pool_folds needs at least one fold's predictions")

    files = [_read(Path(p)) for p in paths]
    files.sort(key=lambda d: int(d["fold"]))

    split = str(files[0]["split"])
    digest = str(files[0]["split_digest"])
    for data in files[1:]:
        if str(data["split"]) != split or str(data["split_digest"]) != digest:
            raise EvalError(
                f"refusing to pool fold {int(data['fold'])} of "
                f"{str(data['split'])!r} (store {str(data['split_digest'])[:12]}) with "
                f"fold {int(files[0]['fold'])} of {split!r} (store {digest[:12]}). Pooling "
                "across split families or store snapshots measures the data, not the model"
            )

    scored = [int(d["fold"]) for d in files if "y_score" in d]
    if scored and len(scored) != len(files):
        missing = [int(d["fold"]) for d in files if "y_score" not in d]
        raise EvalError(
            f"{len(scored)} of {len(files)} folds recorded y_score; fold(s) "
            f"{', '.join(str(f) for f in missing[:5])} did not. Pooling a mixture of "
            "probabilities and hard labels produces a ranking metric that means nothing"
        )

    seen: dict[int, int] = {}
    for data in files:
        index = int(data["fold"])
        for row in data["row_id"].tolist():
            if row in seen:
                raise EvalError(
                    f"row {row} was predicted by fold {seen[row]} and again by fold "
                    f"{index}; folds must be disjoint or the pooled metric double-counts "
                    "them. The split assigns each group to one test fold, so this is a "
                    "runner reporting the wrong positions rather than a bad split"
                )
            seen[row] = index

    y_true = np.concatenate([d["y_true"] for d in files])
    y_pred = np.concatenate([d["y_pred"] for d in files])
    y_score = np.concatenate([d["y_score"] for d in files]) if scored else None

    try:
        values = compute(list(metrics), y_true, y_pred, y_score)
    except MetricError as error:
        raise EvalError(
            f"{error}. Pooled predictions that cannot be scored are a split problem "
            "before they are a metric problem -- check stratification"
        ) from error

    return Pooled(
        row_id=np.concatenate([d["row_id"] for d in files]),
        fold=np.concatenate(
            [np.full(d["row_id"].size, int(d["fold"]), dtype=np.int64) for d in files]
        ),
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        metrics=values,
        folds=tuple(int(d["fold"]) for d in files),
        split=split,
        split_digest=digest,
    )
