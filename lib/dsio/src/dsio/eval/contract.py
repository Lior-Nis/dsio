"""The fixed artifact contract every run produces, whatever the modality.

Stating this contract in prose and then re-implementing it per script is the norm, and it
is why comparing two runs usually means reading two scripts first. Here it is one shape,
written by framework code:

```
artifacts/
    oof.npz         row_id, fold, y_true, y_pred, y_score   <- out-of-fold predictions
    folds.json      per-fold metrics, sizes, timings
    metrics.json    pooled metrics, per-fold mean and std
```

**Why npz and not parquet.** numpy is a hard dependency of the spine; pyarrow is an
optional extra. Making the contract depend on an extra would leave the single most
important artifact in the system untested in a bare install, and would drag pyarrow into a
torch-only project that has no other use for it. ``np.load`` reads these in one line, and
``dsio eval export`` converts to parquet where that is wanted.

**Why out-of-fold predictions are the artifact and not just the metrics.** A metric is a
lossy summary chosen before you knew what you would need to ask. Predictions answer
questions you have not thought of yet — error analysis by subgroup, ensembling, calibration,
threshold selection, the noise floor. Keeping them costs kilobytes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from dsio.contracts import DsioModel, atomic_write

OOF_FILE = "oof.npz"
FOLDS_FILE = "folds.json"
METRICS_FILE = "metrics.json"


class EvalError(ValueError):
    """Raised when predictions contradict the fold structure that produced them."""


@dataclass(frozen=True)
class Fold:
    """One train/test division, as integer row positions.

    Positions rather than boolean masks or sub-objects, because that is the one currency
    every modality shares: they index a dataframe's rows, a numpy array's rows, and a
    :class:`~dsio.data.views.WindowIndex`'s windows identically. The fold loop therefore
    does not need to know what it is folding.

    Disjointness is checked on construction. It is already guaranteed for folds built from
    split files — group parts are validated disjoint there — but folds built from a
    sklearn splitter, or by hand in a notebook, get no such guarantee, and this is the last
    place to catch it before a leaked row silently inflates a score.
    """

    index: int
    train: np.ndarray
    test: np.ndarray
    val: np.ndarray | None = None
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "train", np.asarray(self.train, dtype=np.int64))
        object.__setattr__(self, "test", np.asarray(self.test, dtype=np.int64))
        if self.val is not None:
            object.__setattr__(self, "val", np.asarray(self.val, dtype=np.int64))
        if not self.name:
            object.__setattr__(self, "name", f"fold{self.index}")

        parts = {"train": self.train, "test": self.test}
        if self.val is not None:
            parts["val"] = self.val
        for label, positions in parts.items():
            if positions.size and np.unique(positions).size != positions.size:
                raise EvalError(f"{self.name}: {label} lists the same row more than once")
        names = sorted(parts)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                shared = np.intersect1d(parts[left], parts[right])
                if shared.size:
                    raise EvalError(
                        f"{self.name}: {shared.size} row(s) appear in both {left} and "
                        f"{right}, first at position {int(shared[0])}"
                    )
        if self.test.size == 0:
            raise EvalError(f"{self.name}: test is empty; there is nothing to score")
        if self.train.size == 0:
            raise EvalError(f"{self.name}: train is empty; there is nothing to fit")

    @property
    def sizes(self) -> dict[str, int]:
        sizes = {"train": int(self.train.size), "test": int(self.test.size)}
        if self.val is not None:
            sizes["val"] = int(self.val.size)
        return sizes


@dataclass(frozen=True)
class FoldPrediction:
    """What a runner returns for one fold.

    ``y_score`` is optional because not every model produces one, but omitting it makes
    every ranking metric unavailable — so the loop records which folds supplied it rather
    than quietly scoring hard labels as if they were probabilities.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.y_true) != len(self.y_pred):
            raise EvalError(
                f"y_true has {len(self.y_true)} rows but y_pred has {len(self.y_pred)}"
            )
        if self.y_score is not None and len(self.y_score) != len(self.y_true):
            raise EvalError(
                f"y_true has {len(self.y_true)} rows but y_score has {len(self.y_score)}"
            )


class FoldMetrics(DsioModel):
    """One row of ``folds.json``."""

    fold: int
    name: str
    sizes: dict[str, int]
    seconds: float
    metrics: dict[str, float]


class CVReport(DsioModel):
    """The scored result of a fold loop.

    Pooled and per-fold metrics are both reported because they answer different questions
    and disagree in a way that matters. Pooled is the estimate; per-fold spread is the
    uncertainty. For a rare-event problem, averaging per-fold AP across folds that each
    contain three positives is close to meaningless, which is why ``metrics`` — the headline
    — is computed on the concatenated out-of-fold predictions.
    """

    n_folds: int
    n_rows: int
    predicted_rows: int
    metrics: dict[str, float] = Field(description="Pooled over all out-of-fold predictions.")
    per_fold_mean: dict[str, float] = Field(default_factory=dict)
    per_fold_std: dict[str, float] = Field(
        default_factory=dict,
        description="Fold-to-fold spread. This is the noise floor a verdict is judged against.",
    )
    folds: tuple[FoldMetrics, ...] = ()
    scored_folds: tuple[str, ...] = ()
    has_scores: bool = True
    fold_fingerprint: str | None = Field(
        default=None,
        description="Digest of the exact fold assignment. Two runs are comparable iff equal.",
    )

    @property
    def coverage(self) -> float:
        """Fraction of rows that received an out-of-fold prediction.

        Below 1.0 is legitimate — a purged temporal split deliberately discards a band, a
        grouped holdout tests one cohort — but it must be visible. Reading a pooled metric
        without knowing it covers 40% of the data is how a result gets over-claimed.
        """
        return 0.0 if self.n_rows == 0 else self.predicted_rows / self.n_rows

    def summary_lines(self) -> list[str]:
        lines = [
            f"{self.n_folds} fold(s), {self.predicted_rows:,}/{self.n_rows:,} rows predicted "
            f"({self.coverage:.1%} coverage)"
        ]
        for name in sorted(self.metrics):
            spread = self.per_fold_std.get(name)
            suffix = f"  (fold sd {spread:.4f})" if spread is not None else ""
            lines.append(f"  {name:<20} {self.metrics[name]:.4f}{suffix}")
        return lines


def fold_fingerprint(folds: Sequence[Fold]) -> str:
    """Digest of an exact fold assignment.

The fold assignment is the single source of truth: comparing runs on different folds
    measures the folds, not the models. That is easy to state and impossible to enforce
    unless something records which folds a number came from. Recording it makes the rule
    checkable — :func:`dsio.eval.verdict.compare` refuses two runs whose fingerprints
    differ, instead of returning a confident and worthless delta.

    Only the held-out positions matter. Two runs that hold out identical rows are
    comparable even if one of them trained on less, which is exactly the situation when a
    baseline is fitted on a subset.
    """
    digest = hashlib.sha256()
    for fold in sorted(folds, key=lambda f: f.index):
        digest.update(str(fold.index).encode())
        digest.update(b"\x00")
        digest.update(np.sort(fold.test).astype("<i8").tobytes())
        digest.update(b"\xff")
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class OutOfFold:
    """Concatenated held-out predictions, one row per predicted position."""

    row_id: np.ndarray
    fold: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.row_id.size)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # `Any` because numpy's savez stub types **kwargs as bool.
        arrays: dict[str, Any] = {
            "row_id": self.row_id,
            "fold": self.fold,
            "y_true": self.y_true,
            "y_pred": self.y_pred,
        }
        if self.y_score is not None:
            arrays["y_score"] = self.y_score
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> OutOfFold:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                row_id=data["row_id"],
                fold=data["fold"],
                y_true=data["y_true"],
                y_pred=data["y_pred"],
                y_score=data["y_score"] if "y_score" in data.files else None,
            )


def write_report(directory: Path, report: CVReport, oof: OutOfFold | None = None) -> None:
    """Write the fixed contract into an artifacts directory."""
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write(
        directory / METRICS_FILE,
        json.dumps(
            {
                "metrics": report.metrics,
                "per_fold_mean": report.per_fold_mean,
                "per_fold_std": report.per_fold_std,
                "n_folds": report.n_folds,
                "n_rows": report.n_rows,
                "predicted_rows": report.predicted_rows,
                "coverage": report.coverage,
                "fold_fingerprint": report.fold_fingerprint,
            },
            indent=2,
            sort_keys=True,
        ).encode(),
    )
    atomic_write(
        directory / FOLDS_FILE,
        json.dumps(
            [fold.model_dump(mode="json") for fold in report.folds], indent=2, sort_keys=True
        ).encode(),
    )
    if oof is not None:
        oof.save(directory / OOF_FILE)


def read_report(directory: Path) -> tuple[CVReport, OutOfFold | None]:
    """Read back a written contract. The inverse of :func:`write_report`."""
    metrics_path = directory / METRICS_FILE
    if not metrics_path.is_file():
        raise EvalError(f"no evaluation report at {metrics_path}")
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    folds_path = directory / FOLDS_FILE
    folds = (
        tuple(
            FoldMetrics.model_validate(row)
            for row in json.loads(folds_path.read_text(encoding="utf-8"))
        )
        if folds_path.is_file()
        else ()
    )
    report = CVReport(
        n_folds=summary["n_folds"],
        n_rows=summary["n_rows"],
        predicted_rows=summary["predicted_rows"],
        metrics=summary["metrics"],
        per_fold_mean=summary.get("per_fold_mean", {}),
        per_fold_std=summary.get("per_fold_std", {}),
        folds=folds,
        fold_fingerprint=summary.get("fold_fingerprint"),
    )
    oof_path = directory / OOF_FILE
    return report, OutOfFold.load(oof_path) if oof_path.is_file() else None
