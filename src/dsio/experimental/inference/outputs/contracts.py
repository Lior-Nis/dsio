"""Shared prediction-output contract checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from torch import Tensor

from dsio.inference.predictor import PredictorError

ViolationKind = Literal[
    "shape",
    "dtype",
    "finiteness",
    "sign",
    "range",
    "simplex",
    "threshold consistency",
    "argmax consistency",
    "transform consistency",
]
Shape = tuple[int | None, ...]


class PredictionViolation(PredictorError):
    """A prediction broke its output contract; ``kind`` names the violated rule."""

    def __init__(self, kind: ViolationKind, message: str) -> None:
        super().__init__(kind, message)
        self.kind = kind

    def __str__(self) -> str:
        return f"{self.args[0]}: {self.args[1]}"


def _check_shape(shape: object) -> Shape:
    if isinstance(shape, str | bytes) or not isinstance(shape, Sequence):
        raise ValueError(f"shape must be a sequence of positive integers or null, got {shape!r}")
    result = tuple(shape)
    if any(
        size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1)
        for size in result
    ):
        raise ValueError(f"shape must contain positive integers or null, got {shape!r}")
    return result


def _matches_shape(value: Tensor, shape: Shape) -> bool:
    return value.ndim == len(shape) + 1 and all(
        (expected is None and actual >= 1) or actual == expected
        for actual, expected in zip(value.shape[1:], shape, strict=True)
    )


def _shape_text(shape: Shape) -> str:
    dimensions = ["*" if size is None else str(size) for size in shape]
    return f"[batch{''.join(f', {dimension}' for dimension in dimensions)}]"


def _tensor(output: Mapping[str, Any], field: str, dtype: torch.dtype | None = None) -> Tensor:
    value = output.get(field)
    if not isinstance(value, Tensor):
        raise PredictionViolation("shape", f"{field} must be a tensor")
    if dtype is not None and value.dtype != dtype:
        raise PredictionViolation("dtype", f"{field} must be {dtype}, got {value.dtype}")
    if dtype is None and not value.is_floating_point():
        raise PredictionViolation("dtype", f"{field} must be floating point, got {value.dtype}")
    return value


def _finite(value: Tensor, field: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise PredictionViolation("finiteness", f"{field} must contain only finite values")


def _require_floating(value: Tensor, field: str) -> None:
    if not value.is_floating_point():
        raise PredictionViolation("dtype", f"{field} must be floating point, got {value.dtype}")


def _check_field(name: str, value: object) -> str:
    if not isinstance(value, str) or value in {"", "prediction", "sample_id"}:
        raise ValueError(f"{name} must name a new output field, got {value!r}")
    return value


__all__ = ["PredictionViolation"]
