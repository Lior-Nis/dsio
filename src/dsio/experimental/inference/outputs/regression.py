"""Regression prediction outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from dsio.experimental.inference.outputs import contracts as _contracts
from dsio.experimental.inference.outputs.contracts import PredictionViolation


class RegressionOutput(nn.Module):
    """Named regression values with optional inverse transform and unit scale.

    Consumes:
        Values ``[batch, *shape]``; ``None`` in ``shape`` marks a dynamic extent.

    Produces:
        Same-shape ``prediction`` after inverse transform and scale, plus an optional raw
        field containing the original model values.

    Parameters:
        ``shape``: per-sample shape; ``inverse``: optional ``expm1``; ``scale``: finite
        positive unit scale; ``non_negative``: reject negative output; ``raw_field``:
        optional field for original values.

    Devices:
        CPU and accelerators.

    Limitations:
        The only inverse transform is ``expm1``; arbitrary transforms need a new block.

    Example:
        >>> import torch
        >>> output = RegressionOutput(shape=[None], scale=20_000)
        >>> result = output(torch.tensor([[0.5, 1.0]]))
        >>> result["prediction"].tolist()
        [[10000.0, 20000.0]]
        >>> output.validator(result)
    """

    def __init__(
        self,
        shape: Sequence[int | None],
        inverse: Literal["expm1"] | None = None,
        non_negative: bool = False,
        raw_field: str | None = None,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.shape = _contracts._check_shape(shape)
        self.non_negative = _check_non_negative(non_negative)
        self.raw_field = (
            None if raw_field is None else _contracts._check_field("raw_field", raw_field)
        )
        self.scale = _check_scale(scale)
        self.inverse = _check_inverse(inverse)
        self.validator = RegressionValidator(
            self.shape,
            inverse=self.inverse,
            non_negative=self.non_negative,
            raw_field=self.raw_field,
            scale=self.scale,
        )

    def forward(self, values: Tensor) -> dict[str, Tensor]:
        if not _contracts._matches_shape(values, self.shape):
            raise PredictionViolation(
                "shape",
                f"values must be {_contracts._shape_text(self.shape)}, got {tuple(values.shape)}",
            )
        _contracts._require_floating(values, "values")
        _contracts._finite(values, "values")
        prediction = _transform(values, self.inverse, self.scale)
        result = {"prediction": prediction}
        if self.raw_field is not None:
            result[self.raw_field] = values
        return result


class RegressionValidator:
    """Validate regression shape, dtype, finiteness, and optional sign.

    Consumes:
        A RegressionOutput result.

    Produces:
        Nothing; raises PredictionViolation naming the failed rule.

    Parameters:
        ``shape``, ``inverse``, ``scale``, ``non_negative``, and ``raw_field`` match
        RegressionOutput.

    Devices:
        CPU and accelerators.

    Limitations:
        Floating-point predictions only.

    Example:
        >>> import torch
        >>> RegressionValidator(shape=[None])({"prediction": torch.tensor([[2.5]])})
    """

    def __init__(
        self,
        shape: Sequence[int | None],
        inverse: Literal["expm1"] | None = None,
        non_negative: bool = False,
        raw_field: str | None = None,
        scale: float = 1.0,
    ) -> None:
        self.shape = _contracts._check_shape(shape)
        self.inverse = _check_inverse(inverse)
        self.non_negative = _check_non_negative(non_negative)
        self.raw_field = (
            None if raw_field is None else _contracts._check_field("raw_field", raw_field)
        )
        self.scale = _check_scale(scale)

    def __call__(self, output: Mapping[str, Any]) -> None:
        prediction = _contracts._tensor(output, "prediction")
        values = {"prediction": prediction}
        if self.raw_field is not None:
            values[self.raw_field] = _contracts._tensor(output, self.raw_field)
        for field, value in values.items():
            if not _contracts._matches_shape(value, self.shape):
                raise PredictionViolation(
                    "shape",
                    f"{field} must be {_contracts._shape_text(self.shape)}, "
                    f"got {tuple(value.shape)}",
                )
            _contracts._finite(value, field)
        if self.raw_field is not None and values[self.raw_field].shape != prediction.shape:
            raise PredictionViolation("shape", "prediction and raw values must have the same shape")
        if self.non_negative and bool((prediction < 0).any()):
            raise PredictionViolation("sign", "predictions must be non-negative")
        if self.raw_field is not None:
            expected = _transform(values[self.raw_field], self.inverse, self.scale)
            if not torch.equal(prediction, expected):
                raise PredictionViolation(
                    "transform consistency",
                    "prediction must equal the inverse-transformed and scaled raw values",
                )


def _transform(values: Tensor, inverse: Literal["expm1"] | None, scale: float) -> Tensor:
    result = torch.expm1(values) if inverse == "expm1" else values
    return result if scale == 1.0 else result * scale


def _check_inverse(value: object) -> Literal["expm1"] | None:
    if value is None:
        return None
    if value == "expm1":
        return "expm1"
    raise ValueError(f"inverse must be 'expm1' or null, got {value!r}")


def _check_non_negative(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("non_negative must be true or false")
    return value


def _check_scale(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("scale must be a number")
    try:
        scale = float(value)
    except OverflowError as error:
        raise ValueError("scale must be finite and positive") from error
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be finite and positive, got {value!r}")
    return scale


__all__ = ["RegressionOutput", "RegressionValidator"]
