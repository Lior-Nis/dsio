"""Binary prediction outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from dsio.experimental.inference.outputs import contracts as _contracts
from dsio.experimental.inference.outputs.contracts import PredictionViolation


class BinaryOutput(nn.Module):
    """Sigmoid probabilities and thresholded predictions for scalar or dense logits.

    Consumes:
        Scalar logits ``[batch]``/``[batch, 1]`` by default. With ``shape``, logits are
        ``[batch, *shape]``; ``None`` marks a dynamic extent such as time.

    Produces:
        Same-shape int64 ``prediction`` and floating probabilities under ``score_field``.

    Parameters:
        ``threshold``: probability threshold in ``(0, 1)`` (default ``0.5``); ``shape``:
        optional per-sample shape; ``score_field``: probability field (default ``score``).

    Devices:
        CPU and accelerators.

    Limitations:
        Independent binary targets only; mutually exclusive classes use MulticlassOutput.

    Example:
        >>> import torch
        >>> output = BinaryOutput(shape=[None, 2], score_field="probability")
        >>> result = output(torch.tensor([[[2.0, -1.0]]]))
        >>> result["prediction"].tolist()
        [[[1, 0]]]
        >>> output.validator(result)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        shape: Sequence[int | None] | None = None,
        score_field: str = "score",
    ) -> None:
        super().__init__()
        self.threshold = _check_threshold(threshold)
        self.shape = None if shape is None else _contracts._check_shape(shape)
        self.score_field = _contracts._check_field("score_field", score_field)
        self.validator = BinaryValidator(self.threshold, self.shape, self.score_field)

    def forward(self, logits: Tensor) -> dict[str, Tensor]:
        _contracts._require_floating(logits, "logits")
        _contracts._finite(logits, "logits")
        if self.shape is None:
            if logits.ndim not in (1, 2) or (logits.ndim == 2 and logits.shape[1] != 1):
                raise PredictionViolation(
                    "shape",
                    f"binary logits must be [batch] or [batch, 1], got {tuple(logits.shape)}",
                )
            probability = torch.sigmoid(logits).reshape(-1)
        else:
            if not _contracts._matches_shape(logits, self.shape):
                raise PredictionViolation(
                    "shape",
                    f"binary logits must be {_contracts._shape_text(self.shape)}, "
                    f"got {tuple(logits.shape)}",
                )
            probability = torch.sigmoid(logits)
        return {
            "prediction": (probability >= self.threshold).to(torch.int64),
            self.score_field: probability,
        }


class BinaryValidator:
    """Validate shape, finiteness, range, and threshold consistency.

    Consumes:
        A BinaryOutput result with ``prediction`` and the configured probability field.

    Produces:
        Nothing; raises PredictionViolation naming the failed rule.

    Parameters:
        ``threshold``, ``shape``, and ``score_field`` match BinaryOutput.

    Devices:
        CPU and accelerators.

    Limitations:
        Intended to be derived from BinaryOutput so parameters cannot diverge.

    Example:
        >>> import torch
        >>> validate = BinaryValidator()
        >>> validate({"prediction": torch.tensor([1]), "score": torch.tensor([0.9])})
    """

    def __init__(
        self,
        threshold: float = 0.5,
        shape: Sequence[int | None] | None = None,
        score_field: str = "score",
    ) -> None:
        self.threshold = _check_threshold(threshold)
        self.shape = None if shape is None else _contracts._check_shape(shape)
        self.score_field = _contracts._check_field("score_field", score_field)

    def __call__(self, output: Mapping[str, Any]) -> None:
        prediction = _contracts._tensor(output, "prediction", torch.int64)
        probability = _contracts._tensor(output, self.score_field)
        if self.shape is None:
            valid_shape = prediction.ndim == 1 and probability.shape == prediction.shape
            expected = "[batch]"
        else:
            valid_shape = (
                _contracts._matches_shape(prediction, self.shape)
                and _contracts._matches_shape(probability, self.shape)
                and probability.shape == prediction.shape
            )
            expected = _contracts._shape_text(self.shape)
        if not valid_shape:
            raise PredictionViolation(
                "shape",
                f"prediction {tuple(prediction.shape)} and {self.score_field} "
                f"{tuple(probability.shape)} must both be {expected}",
            )
        _contracts._finite(probability, self.score_field)
        if not bool(((probability >= 0) & (probability <= 1)).all()):
            raise PredictionViolation("range", "binary probabilities must be in [0, 1]")
        if not bool(((prediction == 0) | (prediction == 1)).all()):
            raise PredictionViolation("range", "binary predictions must be 0 or 1")
        if not torch.equal(prediction, (probability >= self.threshold).to(torch.int64)):
            raise PredictionViolation(
                "threshold consistency",
                f"predictions must equal {self.score_field} >= {self.threshold}",
            )


def _check_threshold(threshold: object) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise ValueError("threshold must be a number")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    return float(threshold)


__all__ = ["BinaryOutput", "BinaryValidator"]
