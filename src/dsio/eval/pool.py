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

**A fifth invariant, easy to miss because `cross_validate` never had to state it.** One
closure and one `Examples` meant every fold's `y_true`/`y_pred`/`row_id` structurally came
from one model configuration and one coordinate system -- guaranteed by construction, never
checked, and so never catalogued among the four guards ADR 0017 named. Fold-as-process turns
N single-fold processes loose to disagree on both: a `WindowSpec` digest, so `row_id` means
the same row in every file, and a config identity (everything but `fold`), so every file
came from the same backbone and hyperparameters. Both are recorded per fold in
`predictions.npz` and checked here, alongside split family and store digest, before any
metric is computed over the pooled rows.
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
    """Out-of-fold predictions from every fold of one split family, and their metrics.

    ``fold_metrics`` carries each fold's *own* metrics alongside the pooled headline number,
    keyed by fold index. `dsio.eval.verdict.compare` reads it to pair two runs' per-fold
    series -- the reason this reader, and not the pooled metric alone, is the unit two runs
    are judged by.
    """

    row_id: np.ndarray
    fold: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray | None
    metrics: dict[str, float]
    fold_metrics: dict[int, dict[str, float]]
    folds: tuple[int, ...]
    split: str
    split_digest: str


def _read(path: Path) -> dict[str, np.ndarray]:
    file = path / PREDICTIONS_FILE if path.is_dir() else path
    if not file.is_file():
        raise EvalError(f"{file} does not exist; a fold that never ran cannot be pooled")
    with np.load(file, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def pool_folds(
    paths: Sequence[Path | str],
    *,
    metrics: Sequence[str],
    expected_folds: Sequence[int] | None = None,
) -> Pooled:
    """Pool one ``predictions.npz`` per fold and score the result.

    ``paths`` may be run-artifact directories or the files themselves. Order does not
    matter: folds are sorted by the fold index each file recorded when it was written,
    never by filename or argument position. A run directory's name is whatever the run
    ledger or a shell loop happened to call it; sorting by that string lexically would
    give ten runs back as 0, 1, 10, 2 and produce a perfectly plausible pooled number
    with the folds silently permuted -- undetectable downstream because every individual
    fold's predictions are still valid on their own. Splits used to carry this exact
    hazard too, one committed file per fold sorted by a number parsed from its filename;
    it is why that layout is gone rather than kept, and why this reader does not repeat
    the mistake by trusting a filename instead of the data.

    ``expected_folds``, when given, refuses to pool unless every one of those fold indices
    is present. This is opt-in because ``pool_folds`` never opens a split file itself -- it
    only ever sees the paths it was handed -- so it has no way to know how many folds an
    experiment *should* have unless a caller supplies that. A caller that has already
    loaded the ``SplitFile`` can pass ``[f.index for f in split_file.folds]`` and turn a
    shell loop that silently skipped a fold into a refusal instead of a pooled metric over
    fewer folds than it claims.
    """
    if not paths:
        raise EvalError("pool_folds needs at least one fold's predictions")

    files = [_read(Path(p)) for p in paths]
    files.sort(key=lambda d: int(d["fold"]))

    if expected_folds is not None:
        present = {int(d["fold"]) for d in files}
        missing = sorted(set(expected_folds) - present)
        if missing:
            raise EvalError(
                f"pooled {len(present)} of {len(set(expected_folds))} folds the caller "
                f"expected; fold(s) {missing} never ran. A pooled metric over incomplete "
                "folds is silently biased toward whichever folds happened to run"
            )

    split = str(files[0]["split"])
    digest = str(files[0]["split_digest"])
    window_digest = str(files[0]["window_digest"])
    config_identity = str(files[0]["config_identity"])
    for data in files[1:]:
        if str(data["split"]) != split or str(data["split_digest"]) != digest:
            raise EvalError(
                f"refusing to pool fold {int(data['fold'])} of "
                f"{str(data['split'])!r} (store {str(data['split_digest'])[:12]}) with "
                f"fold {int(files[0]['fold'])} of {split!r} (store {digest[:12]}). Pooling "
                "across split families or store snapshots measures the data, not the model"
            )
        if str(data["window_digest"]) != window_digest:
            raise EvalError(
                f"refusing to pool fold {int(data['fold'])} (window "
                f"{str(data['window_digest'])[:12]}) with fold {int(files[0]['fold'])} "
                f"(window {window_digest[:12]}). Different WindowSpecs give row_id a "
                "different meaning in each file -- the same row_id in two folds would not "
                "even name the same window, let alone the same row"
            )
        if str(data["config_identity"]) != config_identity:
            raise EvalError(
                f"refusing to pool fold {int(data['fold'])} with fold "
                f"{int(files[0]['fold'])}: their configs differ outside of `fold` "
                "(backbone, hyperparameters, or a component). Folds of one experiment must "
                "agree on everything except which fold they are -- these are predictions "
                "from two different models that happen to share a split family"
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
                    "them. The split assigns each group to one test fold, and every file "
                    "here already agreed on the same window spec and config, so this is a "
                    "runner reporting the wrong positions rather than a bad split or a "
                    "mismatched WindowSpec"
                )
            seen[row] = index

    y_true = np.concatenate([d["y_true"] for d in files])
    y_pred = np.concatenate([d["y_pred"] for d in files])
    y_score = np.concatenate([d["y_score"] for d in files]) if scored else None

    try:
        values = compute(list(metrics), y_true, y_pred, y_score)
        fold_metrics = {
            int(d["fold"]): compute(
                list(metrics), d["y_true"], d["y_pred"], d["y_score"] if scored else None
            )
            for d in files
        }
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
        fold_metrics=fold_metrics,
        folds=tuple(int(d["fold"]) for d in files),
        split=split,
        split_digest=digest,
    )
