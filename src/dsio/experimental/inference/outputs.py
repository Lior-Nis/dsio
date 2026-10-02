"""Prediction outputs: binary, multiclass/ordinal and regression, each with its validator.

An output is the predictor's normalizer: it turns the model's raw tensor into named fields.
Its ``validator`` attribute is the matching validator, built from the same parameters, so a
consumer passes ``normalizer=output, validator=output.validator`` and the two never
disagree. Every violation raises :class:`PredictionViolation` naming its kind.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from dsio.inference.predictor import PredictorError

ViolationKind = Literal[
    "shape",
    "finiteness",
    "sign",
    "range",
    "simplex",
    "threshold consistency",
    "argmax consistency",
]
_SIMPLEX_TOLERANCE = 1e-6


class PredictionViolation(PredictorError):
    """A prediction broke its output contract; ``kind`` names the violated rule."""

    def __init__(self, kind: ViolationKind, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind


class BinaryOutput(nn.Module):
    """Sigmoid scores and thresholded 0/1 predictions from one logit per sample.

    Consumes:
        Logits ``[batch]`` or ``[batch, 1]``.

    Produces:
        ``prediction``: int64 ``[batch]``, 1 where ``score >= threshold``; ``score``: the
        sigmoid probability, float ``[batch]``.

    Parameters:
        ``threshold``: in ``(0, 1)``, default ``0.5``.

    Devices:
        CPU and accelerators.

    Limitations:
        One binary target per sample; dense ``[batch, time, k]`` outputs are not covered.

    Example:
        >>> import torch
        >>> output = BinaryOutput()
        >>> result = output(torch.tensor([[2.0], [-1.0]]))
        >>> result["prediction"].tolist(), [round(v, 3) for v in result["score"].tolist()]
        ([1, 0], [0.881, 0.269])
        >>> output.validator(result)
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a number")
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.threshold = float(threshold)
        self.validator = BinaryValidator(self.threshold)

    def forward(self, logits: Tensor) -> dict[str, Tensor]:
        if logits.ndim not in (1, 2) or (logits.ndim == 2 and logits.shape[1] != 1):
            raise PredictionViolation(
                "shape", f"binary logits must be [batch] or [batch, 1], got {tuple(logits.shape)}"
            )
        score = torch.sigmoid(logits).reshape(-1)
        return {"prediction": (score >= self.threshold).to(torch.int64), "score": score}


class BinaryValidator:
    """Validate a :class:`BinaryOutput` result: shape, finiteness, range and threshold.

    Usually taken from ``BinaryOutput(...).validator`` so both share the threshold.

    Consumes:
        A predictor result with ``prediction`` and ``score``.

    Produces:
        Nothing; raises :class:`PredictionViolation` (``shape``, ``finiteness``, ``range``
        or ``threshold consistency``).

    Parameters:
        ``threshold``: the output's threshold (default ``0.5``).

    Devices:
        CPU and accelerators.

    Limitations:
        Checks ``[batch]`` results only.

    Example:
        >>> import torch
        >>> validate = BinaryValidator(threshold=0.5)
        >>> validate({"prediction": torch.tensor([1]), "score": torch.tensor([0.9])})
        >>> try:
        ...     validate({"prediction": torch.tensor([0]), "score": torch.tensor([0.9])})
        ... except PredictionViolation as error:
        ...     error.kind
        'threshold consistency'
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    def __call__(self, output: Mapping[str, Any]) -> None:
        prediction = _tensor(output, "prediction", torch.int64)
        score = _tensor(output, "score")
        if prediction.ndim != 1 or score.shape != prediction.shape:
            raise PredictionViolation(
                "shape",
                f"prediction {tuple(prediction.shape)} and score {tuple(score.shape)} "
                "must both be [batch]",
            )
        _finite(score, "score")
        if not bool(((score >= 0) & (score <= 1)).all()):
            raise PredictionViolation("range", "binary scores must be probabilities in [0, 1]")
        if not bool(((prediction == 0) | (prediction == 1)).all()):
            raise PredictionViolation("range", "binary predictions must be 0 or 1")
        if not torch.equal(prediction, (score >= self.threshold).to(torch.int64)):
            raise PredictionViolation(
                "threshold consistency", f"predictions must equal score >= {self.threshold}"
            )


class MulticlassOutput(nn.Module):
    """Argmax class labels, optionally with softmax scores and an ordinal label offset.

    The prediction is the argmax of the scores when they are reported, so the two always
    agree, and of the logits otherwise.

    Consumes:
        Finite logits ``[batch, classes]``.

    Produces:
        ``prediction``: int64 ``[batch]`` in ``[label_offset, label_offset + classes)``;
        with ``scores``, ``score``: softmax probabilities ``[batch, classes]``.

    Parameters:
        ``classes``: the number of classes (at least 2); ``scores``: report probabilities
        (default ``false``); ``label_offset``: added to the class index, e.g. ``1`` for
        ordinal labels 1..K (default ``0``).

    Devices:
        CPU and accelerators.

    Limitations:
        One label per sample; per-step or multi-label outputs are not covered.

    Example:
        >>> import torch
        >>> output = MulticlassOutput(classes=3, scores=True, label_offset=1)
        >>> result = output(torch.tensor([[0.0, 2.0, 1.0]]))
        >>> result["prediction"].tolist(), round(result["score"].sum().item(), 6)
        ([2], 1.0)
        >>> output.validator(result)
    """

    def __init__(self, classes: int, scores: bool = False, label_offset: int = 0) -> None:
        super().__init__()
        if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
            raise ValueError(f"classes must be an integer of at least 2, got {classes!r}")
        if not isinstance(scores, bool):
            raise ValueError("scores must be true or false")
        if isinstance(label_offset, bool) or not isinstance(label_offset, int):
            raise ValueError(f"label_offset must be an integer, got {label_offset!r}")
        self.classes = classes
        self.scores = scores
        self.label_offset = label_offset
        self.validator = MulticlassValidator(classes, scores, label_offset)

    def forward(self, logits: Tensor) -> dict[str, Tensor]:
        if logits.ndim != 2 or logits.shape[1] != self.classes:
            raise PredictionViolation(
                "shape", f"logits must be [batch, {self.classes}], got {tuple(logits.shape)}"
            )
        _finite(logits, "logits")
        if not self.scores:
            return {"prediction": torch.argmax(logits, dim=1) + self.label_offset}
        probabilities = torch.softmax(logits, dim=1)
        return {
            "prediction": torch.argmax(probabilities, dim=1) + self.label_offset,
            "score": probabilities,
        }


class MulticlassValidator:
    """Validate a :class:`MulticlassOutput` result: shape, range, simplex and argmax.

    Usually taken from ``MulticlassOutput(...).validator`` so both share their parameters.

    Consumes:
        A predictor result with ``prediction`` and, when ``scores``, ``score``.

    Produces:
        Nothing; raises :class:`PredictionViolation` (``shape``, ``finiteness``, ``range``,
        ``simplex`` or ``argmax consistency``).

    Parameters:
        ``classes``, ``scores`` and ``label_offset``, as on the output.

    Devices:
        CPU and accelerators.

    Limitations:
        Rows must sum to 1 within ``1e-6``.

    Example:
        >>> import torch
        >>> validate = MulticlassValidator(classes=3, label_offset=1)
        >>> validate({"prediction": torch.tensor([1, 3])})
        >>> try:
        ...     validate({"prediction": torch.tensor([0])})
        ... except PredictionViolation as error:
        ...     error.kind
        'range'
    """

    def __init__(self, classes: int, scores: bool = False, label_offset: int = 0) -> None:
        self.classes = classes
        self.scores = scores
        self.label_offset = label_offset

    def __call__(self, output: Mapping[str, Any]) -> None:
        prediction = _tensor(output, "prediction", torch.int64)
        if prediction.ndim != 1:
            raise PredictionViolation(
                "shape", f"prediction must be [batch], got {tuple(prediction.shape)}"
            )
        low, high = self.label_offset, self.label_offset + self.classes - 1
        if not bool(((prediction >= low) & (prediction <= high)).all()):
            raise PredictionViolation("range", f"predictions must be in [{low}, {high}]")
        if not self.scores:
            return
        score = _tensor(output, "score")
        if score.shape != (prediction.shape[0], self.classes):
            raise PredictionViolation(
                "shape",
                f"score must be [{prediction.shape[0]}, {self.classes}], got {tuple(score.shape)}",
            )
        _finite(score, "score")
        rows = score.sum(dim=1)
        if bool((score < 0).any()) or not torch.allclose(
            rows, torch.ones_like(rows), atol=_SIMPLEX_TOLERANCE
        ):
            raise PredictionViolation("simplex", "scores must be probability distributions")
        if not torch.equal(prediction, torch.argmax(score, dim=1) + self.label_offset):
            raise PredictionViolation(
                "argmax consistency", "predictions must be the argmax of their scores"
            )


class RegressionOutput(nn.Module):
    """Named regression values, optionally mapped back from a transformed target.

    Nothing is clamped: a negative value under ``non_negative`` is a violation for the
    validator to reject, not a value to hide.

    Consumes:
        Values ``[batch, *shape]``, in target space after any training transform.

    Produces:
        ``prediction``: ``[batch, *shape]`` after the inverse transform; with
        ``raw_field``, also the untransformed values under that name.

    Parameters:
        ``shape``: per-sample shape (e.g. ``[1]`` or ``[horizon]``); ``inverse``: ``expm1``
        undoes a ``log1p`` target (default none); ``non_negative``: the validator rejects
        negative predictions (default ``false``); ``raw_field``: optional field name for
        the untransformed values.

    Devices:
        CPU and accelerators.

    Limitations:
        The only inverse transform is ``expm1``; other target transforms need a new
        declared inverse.

    Example:
        >>> import torch
        >>> output = RegressionOutput(shape=[2], inverse="expm1", non_negative=True)
        >>> result = output(torch.log1p(torch.tensor([[1.0, 3.0]])))
        >>> [round(v, 4) for v in result["prediction"][0].tolist()]
        [1.0, 3.0]
        >>> output.validator(result)
    """

    def __init__(
        self,
        shape: Sequence[int],
        inverse: Literal["expm1"] | None = None,
        non_negative: bool = False,
        raw_field: str | None = None,
    ) -> None:
        super().__init__()
        if isinstance(shape, (str, bytes)) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape
        ):
            raise ValueError(f"shape must be positive integers, got {shape!r}")
        if inverse not in (None, "expm1"):
            raise ValueError(f"inverse must be 'expm1' or null, got {inverse!r}")
        if not isinstance(non_negative, bool):
            raise ValueError("non_negative must be true or false")
        if raw_field is not None and (
            not isinstance(raw_field, str) or raw_field in {"", "prediction", "sample_id"}
        ):
            raise ValueError(f"raw_field must name a new output field, got {raw_field!r}")
        self.shape = tuple(shape)
        self.inverse = inverse
        self.non_negative = non_negative
        self.raw_field = raw_field
        self.validator = RegressionValidator(self.shape, non_negative, raw_field)

    def forward(self, values: Tensor) -> dict[str, Tensor]:
        if tuple(values.shape[1:]) != self.shape or values.ndim != len(self.shape) + 1:
            raise PredictionViolation(
                "shape",
                f"values must be [batch, {', '.join(map(str, self.shape))}], "
                f"got {tuple(values.shape)}",
            )
        prediction = torch.expm1(values) if self.inverse == "expm1" else values
        result = {"prediction": prediction}
        if self.raw_field is not None:
            result[self.raw_field] = values
        return result


class RegressionValidator:
    """Validate a :class:`RegressionOutput` result: shape, finiteness and sign.

    Usually taken from ``RegressionOutput(...).validator`` so both share their parameters.

    Consumes:
        A predictor result with ``prediction`` and, with ``raw_field``, that field.

    Produces:
        Nothing; raises :class:`PredictionViolation` (``shape``, ``finiteness`` or
        ``sign``).

    Parameters:
        ``shape``, ``non_negative`` and ``raw_field``, as on the output.

    Devices:
        CPU and accelerators.

    Limitations:
        Floating-point predictions only.

    Example:
        >>> import torch
        >>> validate = RegressionValidator(shape=[1], non_negative=True)
        >>> validate({"prediction": torch.tensor([[2.5]])})
        >>> try:
        ...     validate({"prediction": torch.tensor([[-1.0]])})
        ... except PredictionViolation as error:
        ...     error.kind
        'sign'
    """

    def __init__(
        self, shape: Sequence[int], non_negative: bool = False, raw_field: str | None = None
    ) -> None:
        self.shape = tuple(shape)
        self.non_negative = non_negative
        self.raw_field = raw_field

    def __call__(self, output: Mapping[str, Any]) -> None:
        fields = ["prediction"] if self.raw_field is None else ["prediction", self.raw_field]
        for field in fields:
            value = _tensor(output, field)
            if value.ndim != len(self.shape) + 1 or tuple(value.shape[1:]) != self.shape:
                raise PredictionViolation(
                    "shape",
                    f"{field} must be [batch, {', '.join(map(str, self.shape))}], "
                    f"got {tuple(value.shape)}",
                )
            _finite(value, field)
        if self.non_negative and bool((output["prediction"] < 0).any()):
            raise PredictionViolation("sign", "predictions must be non-negative")


def _tensor(output: Mapping[str, Any], field: str, dtype: torch.dtype | None = None) -> Tensor:
    value = output.get(field)
    if not isinstance(value, Tensor):
        raise PredictionViolation("shape", f"{field} must be a tensor")
    if dtype is not None and value.dtype != dtype:
        raise PredictionViolation("shape", f"{field} must be {dtype}, got {value.dtype}")
    if dtype is None and not value.is_floating_point():
        raise PredictionViolation("shape", f"{field} must be floating point, got {value.dtype}")
    return value


def _finite(value: Tensor, field: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise PredictionViolation("finiteness", f"{field} must contain only finite values")


__all__ = [
    "BinaryOutput",
    "BinaryValidator",
    "MulticlassOutput",
    "MulticlassValidator",
    "PredictionViolation",
    "RegressionOutput",
    "RegressionValidator",
]
