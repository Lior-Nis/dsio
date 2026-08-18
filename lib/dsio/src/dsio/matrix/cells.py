"""Expanding axes into cells, and giving each cell a content-addressed identity.

An axis is written exactly like an override, with a list of values:
``task.lr=1e-3,3e-4`` or ``task.folds=0,1,2``. A matrix is the cross product of its axes,
so the syntax a person already knows for one run extends to two hundred without a second
grammar to learn.

**A cell's identity is the sha256 of its resolved config**, not its position in the product.
Three consequences, and the third is the one that makes resume trustworthy:

1. re-running a completed cell is a no-op;
2. adding an axis value does not renumber the cells that already ran;
3. two cells that resolve to the same config *are* the same job, even if they were reached
   by different paths — a duplicate is impossible rather than merely unlikely.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dsio.config.overrides import OverrideError, split_override

GLOB_PREFIX = "glob:"


class MatrixError(ValueError):
    """Raised when a matrix specification cannot be expanded."""


@dataclass(frozen=True)
class Axis:
    """One dimension of the product: a config path and the values it takes."""

    path: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise MatrixError(f"axis {self.path!r} has no values")
        duplicates = [v for v in set(self.values) if self.values.count(v) > 1]
        if duplicates:
            raise MatrixError(
                f"axis {self.path!r} repeats {', '.join(sorted(duplicates))}; a repeated "
                "value silently halves the apparent size of the sweep"
            )

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(f"{self.path}={value}" for value in self.values)


@dataclass(frozen=True)
class Cell:
    """One point in the product: the overrides that produce it, and where it sits."""

    index: int
    overrides: tuple[str, ...]
    coordinates: tuple[tuple[str, str], ...]

    @property
    def label(self) -> str:
        return " ".join(f"{path}={value}" for path, value in self.coordinates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "coordinates": dict(self.coordinates),
            "overrides": list(self.overrides),
        }


def parse_axis(token: str, *, root: Path | None = None) -> Axis:
    """Parse ``path=v1,v2,v3`` or ``path=glob:pattern`` into an axis.

    The glob form is what makes "probe every checkpoint that exists" expressible. It is
    resolved and **sorted** at parse time rather than lazily: a matrix whose size depends on
    when it was expanded cannot be resumed, because the second invocation would be a
    different matrix.
    """
    path, _, raw = token.partition("=")
    if not _ or not path:
        raise MatrixError(
            f"axis {token!r} must be path=value,value; got no '='"
        )

    if raw.startswith(GLOB_PREFIX):
        pattern = raw[len(GLOB_PREFIX) :]
        base = root or Path()
        matches = sorted(str(match) for match in base.glob(pattern))
        if not matches:
            raise MatrixError(
                f"axis {path!r} matched no paths for pattern {pattern!r} under {base}"
            )
        return Axis(path=path, values=tuple(matches))

    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise MatrixError(f"axis {path!r} has no values")
    # Validate each as a real override now, so a typo costs a parse rather than a sweep.
    for value in values:
        split_override(f"{path}={value}")
    return Axis(path=path, values=values)


def parse_axes(tokens: Sequence[str], *, root: Path | None = None) -> list[Axis]:
    """Parse every axis, rejecting a path that appears twice."""
    axes = [parse_axis(token, root=root) for token in tokens]
    seen: dict[str, int] = {}
    for axis in axes:
        seen[axis.path] = seen.get(axis.path, 0) + 1
    repeated = [path for path, count in seen.items() if count > 1]
    if repeated:
        raise MatrixError(
            f"axis path(s) given twice: {', '.join(sorted(repeated))}; the second would "
            "silently win and half the sweep would never run"
        )
    return axes


def expand(axes: Sequence[Axis], base: Sequence[str] = ()) -> list[Cell]:
    """Cross product of the axes, in a stable order.

    Order is the axis order given, with the last axis varying fastest — the same order a
    nested loop would produce, so a partially-completed sweep reads the way it ran.
    """
    if not axes:
        return [Cell(index=0, overrides=tuple(base), coordinates=())]

    cells: list[Cell] = []
    for index, combination in enumerate(itertools.product(*(axis.values for axis in axes))):
        coordinates = tuple(zip((axis.path for axis in axes), combination, strict=True))
        overrides = tuple(base) + tuple(f"{path}={value}" for path, value in coordinates)
        cells.append(Cell(index=index, overrides=overrides, coordinates=coordinates))
    return cells


def size(axes: Sequence[Axis]) -> int:
    """How many cells the product has, without building them.

    Worth having separately: a sweep that would take a week should be refusable before it
    is materialised, not after.
    """
    total = 1
    for axis in axes:
        total *= len(axis.values)
    return total


def validate(preset: str, cells: Sequence[Cell]) -> list[str]:
    """Resolve every cell's config now, returning one config hash per cell.

    This is the pre-flight for a sweep. Two hundred cells that each fail after loading a
    corpus is two hundred wasted loads; resolving them all first turns a typo in one axis
    into an immediate parse error. It also produces the identities the resume logic needs,
    so the work is not wasted.
    """
    from dsio.config.presets import resolve

    hashes: list[str] = []
    for cell in cells:
        try:
            config = resolve(preset, list(cell.overrides))
        except (OverrideError, ValueError, KeyError) as error:
            raise MatrixError(
                f"cell {cell.index} ({cell.label or 'no axes'}) does not resolve: {error}"
            ) from error
        hashes.append(config.config_hash)
    return hashes
