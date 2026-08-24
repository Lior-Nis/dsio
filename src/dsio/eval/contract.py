"""The fixed artifact contract every run produces, whatever the modality.

Stating this contract in prose and then re-implementing it per script is the norm, and it
is why comparing two runs usually means reading two scripts first. Decision 6 (fold-as-
process) makes the process boundary the fold boundary: one run trains and predicts one
fold and writes one file --

```
artifacts/
    predictions.npz    row_id, fold, y_true, y_pred, y_score, split, split_digest,
                        window_digest, config_identity
```

Cross-validation is running the entry point N times; nothing in this module accumulates
folds any more. :func:`dsio.eval.pool.pool_folds` reads N of these files back and pools
them, and :mod:`dsio.eval.verdict` compares two pooled sets against each other.

**Why npz and not parquet.** numpy is a hard dependency of the spine; pyarrow is an
optional extra. Making the contract depend on an extra would leave the single most
important artifact in the system untested in a bare install, and would drag pyarrow into a
torch-only project that has no other use for it. ``np.load`` reads these in one line;
converting to parquet, or anything else, is the reader's business.

**Why held-out predictions are the artifact and not just the metrics.** A metric is a
lossy summary chosen before you knew what you would need to ask. Predictions answer
questions you have not thought of yet — error analysis by subgroup, ensembling, calibration,
threshold selection, the noise floor. Keeping them costs kilobytes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Decision 6 (fold-as-process): the artifact one run writes for its one fold. `dsio.train`
# imports this name rather than defining its own copy, so a pooling reader (`dsio.eval.pool`)
# and the writer (`dsio.train.torch_task._write_predictions`) can never drift apart on the
# filename they agree on.
PREDICTIONS_FILE = "predictions.npz"


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

    ``evaluation_only`` permits an empty train part. Evaluating something that was not
    trained here — a benchmark pass, a shipped model, an agent behind an API — is a real
    shape, and the alternative is inventing a token training set that both lies and
    violates the disjointness the class exists to enforce.
    """

    index: int
    train: np.ndarray
    test: np.ndarray
    val: np.ndarray | None = None
    name: str = ""
    evaluation_only: bool = False

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
        if self.train.size == 0 and not self.evaluation_only:
            raise EvalError(
                f"{self.name}: train is empty; there is nothing to fit. If this is an "
                "evaluation of something already trained — a benchmark pass, a shipped "
                "model, an agent — set evaluation_only=True to say so."
            )


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

