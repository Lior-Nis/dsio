"""Declarative collation for aligned variable-length fields."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate

from dsio.data.loading.datasets import DataItem, LoadingError


class PadCollator:
    """Pad aligned fields, stack fixed fields, and optionally emit a validity mask.

    Consumes:
        Mapping items containing ``sample_id`` plus every field declared in
        ``padded_fields`` and ``fixed_fields``. Padded fields are tensors or values
        accepted by ``torch.as_tensor`` and align on axis zero within each sample.

    Produces:
        A CPU tensor batch with padded fields shaped ``[batch, max_length, ...]``,
        stacked fixed fields, ordered ``sample_id``, and optionally a boolean ``mask``
        whose True values identify real positions.

    Parameters:
        ``padded_fields`` maps each ragged field to its padding value; ``fixed_fields``
        names fields stacked without padding; ``emit_mask`` adds the padding-validity
        mask. A declared fixed field may be absent from an inputs-only batch, but it must
        be present in every item or none.

    Devices:
        CPU collation only; device transfer remains Lightning's responsibility.

    Limitations:
        All padded fields belong to one alignment group. Every non-identity field must
        be declared. Padding is only along axis zero.

    Example:
        >>> collate = PadCollator({"x": 0.0}, fixed_fields=["y"], emit_mask=True)
        >>> batch = collate([
        ...     {"sample_id": "a", "x": [1.0], "y": 0},
        ...     {"sample_id": "b", "x": [2.0, 3.0], "y": 1},
        ... ])
        >>> batch["mask"].tolist()
        [[True, False], [True, True]]
    """

    def __init__(
        self,
        padded_fields: Mapping[str, int | float | bool],
        fixed_fields: Sequence[str] = (),
        emit_mask: bool = False,
    ) -> None:
        if not isinstance(padded_fields, Mapping) or not padded_fields:
            raise LoadingError("PadCollator needs at least one padded field")
        if isinstance(fixed_fields, str) or not isinstance(fixed_fields, Sequence):
            raise LoadingError("fixed_fields must be a sequence of field names")
        if not isinstance(emit_mask, bool):
            raise LoadingError("emit_mask must be bool")
        padded = dict(padded_fields)
        fixed = tuple(fixed_fields)
        names = [*padded, *fixed]
        if any(not isinstance(name, str) or not name for name in names):
            raise LoadingError("collator field names must be non-empty strings")
        if "sample_id" in names:
            raise LoadingError("sample_id is preserved automatically and cannot be declared")
        repeated = sorted(set(padded) & set(fixed))
        if repeated:
            raise LoadingError(f"fields cannot be both padded and fixed: {repeated}")
        if len(fixed) != len(set(fixed)):
            raise LoadingError("fixed_fields cannot contain duplicates")
        if emit_mask and "mask" in padded:
            raise LoadingError("emitted mask collides with declared padded field 'mask'")
        if emit_mask and "mask" in fixed:
            raise LoadingError("emitted mask collides with declared fixed field 'mask'")
        for name, value in padded.items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, int | float) or not math.isfinite(value):
                raise LoadingError(f"padding value for field {name!r} must be a finite scalar")
        self.padded_fields = padded
        self.fixed_fields = fixed
        self.emit_mask = emit_mask

    def __call__(self, items: list[DataItem]) -> dict[str, Any]:
        if not items:
            raise LoadingError("cannot collate an empty batch")
        for position, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise LoadingError(f"batch item {position} must be a mapping")
            if not isinstance(item.get("sample_id"), str):
                raise LoadingError(f"batch item {position} has no string sample_id")
        present_fixed = []
        for name in self.fixed_fields:
            count = sum(name in item for item in items)
            if count not in (0, len(items)):
                raise LoadingError(f"fixed field {name!r} must be present in every item or none")
            if count:
                present_fixed.append(name)
        expected = {"sample_id", *self.padded_fields, *present_fixed}
        declared = {"sample_id", *self.padded_fields, *self.fixed_fields}
        identities: list[str] = []
        values: dict[str, list[Tensor]] = {name: [] for name in self.padded_fields}
        for item in items:
            sample_id = item.get("sample_id")
            assert isinstance(sample_id, str)
            identities.append(sample_id)
            missing = sorted(expected - set(item))
            extra = sorted(set(item) - declared)
            if missing:
                raise LoadingError(f"sample {sample_id!r} is missing declared fields {missing}")
            if extra:
                raise LoadingError(f"sample {sample_id!r} has undeclared fields {extra}")
            lengths: dict[str, int] = {}
            for name in self.padded_fields:
                tensor = _tensor(item[name], sample_id, name)
                values[name].append(tensor)
                lengths[name] = len(tensor)
            if len(set(lengths.values())) != 1:
                detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
                raise LoadingError(
                    f"padded fields are misaligned for sample {sample_id!r}: {detail}"
                )
        result: dict[str, Any] = {"sample_id": identities}
        for name, padding_value in self.padded_fields.items():
            shapes = {tuple(value.shape[1:]) for value in values[name]}
            if len(shapes) != 1:
                raise LoadingError(
                    f"padded field {name!r} has inconsistent trailing shapes {sorted(shapes)}"
                )
            dtypes = {value.dtype for value in values[name]}
            if len(dtypes) != 1:
                raise LoadingError(
                    f"padded field {name!r} has inconsistent dtypes {sorted(map(str, dtypes))}"
                )
            _require_representable_padding(name, padding_value, next(iter(dtypes)))
            try:
                result[name] = pad_sequence(
                    values[name], batch_first=True, padding_value=float(padding_value)
                )
            except RuntimeError as error:
                raise LoadingError(f"cannot pad field {name!r}: {error}") from error
        for name in present_fixed:
            try:
                fixed_values = [item[name] for item in items]
                for sample_id, value in zip(identities, fixed_values, strict=True):
                    _require_cpu(value, sample_id, name)
                result[name] = default_collate(fixed_values)
            except Exception as error:
                raise LoadingError(f"cannot stack fixed field {name!r}: {error}") from error
        if self.emit_mask:
            mask_lengths = [len(value) for value in values[next(iter(self.padded_fields))]]
            width = max(mask_lengths)
            result["mask"] = torch.arange(width).unsqueeze(0) < torch.tensor(
                mask_lengths
            ).unsqueeze(1)
        return result


def _tensor(value: object, sample_id: str, field: str) -> Tensor:
    try:
        tensor = torch.as_tensor(value)
    except Exception as error:
        raise LoadingError(
            f"sample {sample_id!r} padded field {field!r} cannot become a tensor: {error}"
        ) from error
    if tensor.ndim == 0:
        raise LoadingError(f"sample {sample_id!r} padded field {field!r} needs an axis zero")
    if tensor.device.type != "cpu":
        raise LoadingError(f"sample {sample_id!r} padded field {field!r} must remain on CPU")
    return tensor


def _require_representable_padding(
    field: str,
    padding_value: int | float | bool,
    dtype: torch.dtype,
) -> None:
    if dtype.is_floating_point or dtype.is_complex:
        return
    if dtype == torch.bool:
        represented = padding_value in (False, True, 0, 1)
    else:
        limits = torch.iinfo(dtype)
        represented = (
            isinstance(padding_value, int | bool) and limits.min <= padding_value <= limits.max
        )
    if not represented:
        raise LoadingError(
            f"padding value {padding_value!r} for field {field!r} cannot be represented by {dtype}"
        )


def _require_cpu(value: object, sample_id: str, field: str) -> None:
    if isinstance(value, Tensor):
        if value.device.type != "cpu":
            raise LoadingError(f"sample {sample_id!r} fixed field {field!r} must remain on CPU")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _require_cpu(item, sample_id, field)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for item in value:
            _require_cpu(item, sample_id, field)


__all__ = ["PadCollator"]
