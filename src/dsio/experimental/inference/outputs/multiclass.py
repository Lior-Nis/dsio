"""Multiclass and ordinal prediction outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from dsio.experimental.inference.outputs import contracts as _contracts
from dsio.experimental.inference.outputs.contracts import PredictionViolation

_SIMPLEX_TOLERANCE = 1e-6
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class MulticlassOutput(nn.Module):
    """Argmax class labels, optionally with probabilities and an ordinal label offset.

    Consumes:
        Finite logits ``[batch, classes]``.

    Produces:
        Int64 ``prediction`` in the offset class range and optional softmax probabilities.

    Parameters:
        ``classes``: class count; ``scores``: emit probabilities; ``label_offset``: added
        to indices; ``score_field``: probability field (default ``score``).

    Devices:
        CPU and accelerators.

    Limitations:
        One mutually exclusive label per sample.

    Example:
        >>> import torch
        >>> output = MulticlassOutput(classes=3, scores=True, label_offset=1)
        >>> result = output(torch.tensor([[0.0, 2.0, 1.0]]))
        >>> result["prediction"].tolist()
        [2]
        >>> output.validator(result)
    """

    def __init__(
        self,
        classes: int,
        scores: bool = False,
        label_offset: int = 0,
        score_field: str = "score",
    ) -> None:
        super().__init__()
        _check_multiclass(classes, scores, label_offset, score_field)
        self.classes = classes
        self.scores = scores
        self.label_offset = label_offset
        self.score_field = score_field
        self.validator = MulticlassValidator(classes, scores, label_offset, score_field)

    def forward(self, logits: Tensor) -> dict[str, Tensor]:
        if logits.ndim != 2 or logits.shape[1] != self.classes:
            raise PredictionViolation(
                "shape", f"logits must be [batch, {self.classes}], got {tuple(logits.shape)}"
            )
        _contracts._require_floating(logits, "logits")
        _contracts._finite(logits, "logits")
        if not self.scores:
            return {"prediction": torch.argmax(logits, dim=1) + self.label_offset}
        probability = torch.softmax(logits, dim=1)
        return {
            "prediction": torch.argmax(probability, dim=1) + self.label_offset,
            self.score_field: probability,
        }


class MulticlassValidator:
    """Validate class range, simplex, and argmax consistency.

    Consumes:
        A MulticlassOutput result.

    Produces:
        Nothing; raises PredictionViolation naming the failed rule.

    Parameters:
        ``classes``, ``scores``, ``label_offset``, and ``score_field`` match the output.

    Devices:
        CPU and accelerators.

    Limitations:
        Probability rows use a dtype-aware absolute simplex tolerance.

    Example:
        >>> import torch
        >>> MulticlassValidator(classes=3, label_offset=1)({"prediction": torch.tensor([2])})
    """

    def __init__(
        self,
        classes: int,
        scores: bool = False,
        label_offset: int = 0,
        score_field: str = "score",
    ) -> None:
        _check_multiclass(classes, scores, label_offset, score_field)
        self.classes = classes
        self.scores = scores
        self.label_offset = label_offset
        self.score_field = score_field

    def __call__(self, output: Mapping[str, Any]) -> None:
        prediction = _contracts._tensor(output, "prediction", torch.int64)
        if prediction.ndim != 1:
            raise PredictionViolation(
                "shape", f"prediction must be [batch], got {tuple(prediction.shape)}"
            )
        low, high = self.label_offset, self.label_offset + self.classes - 1
        if not bool(((prediction >= low) & (prediction <= high)).all()):
            raise PredictionViolation("range", f"predictions must be in [{low}, {high}]")
        if not self.scores:
            return
        score = _contracts._tensor(output, self.score_field)
        if score.shape != (prediction.shape[0], self.classes):
            raise PredictionViolation(
                "shape",
                f"score must be [{prediction.shape[0]}, {self.classes}], got {tuple(score.shape)}",
            )
        _contracts._finite(score, self.score_field)
        rows = score.float().sum(dim=1)
        tolerance = max(_SIMPLEX_TOLERANCE, torch.finfo(score.dtype).eps)
        if bool(((score < 0) | (score > 1)).any()) or not torch.allclose(
            rows, torch.ones_like(rows), atol=tolerance, rtol=0.0
        ):
            raise PredictionViolation("simplex", "scores must be probability distributions")
        if not torch.equal(prediction, torch.argmax(score, dim=1) + self.label_offset):
            raise PredictionViolation(
                "argmax consistency", "predictions must be the argmax of their scores"
            )


def _check_multiclass(classes: object, scores: object, offset: object, field: object) -> None:
    if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
        raise ValueError(f"classes must be an integer of at least 2, got {classes!r}")
    if not isinstance(scores, bool):
        raise ValueError("scores must be true or false")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError(f"label_offset must be an integer, got {offset!r}")
    if offset < _INT64_MIN or offset + classes - 1 > _INT64_MAX:
        raise ValueError("label_offset and class range must fit in int64")
    _contracts._check_field("score_field", field)


__all__ = ["MulticlassOutput", "MulticlassValidator"]
