"""Temporal splits: walk-forward with purging and embargo.

A group list cannot express a temporal split, because the unit being divided is *time*, not
subject. And a naive time cut is wrong in two specific ways that López de Prado named, both
of which silently inflate results:

**Purging.** A training window whose *label* extends into the test period has seen the test
period. With a label horizon of 5 bars, a window ending one bar before the test starts still
resolves inside it. Those windows must be dropped, not merely ordered earlier.

**Embargo.** Serial correlation means a training window starting immediately *after* the
test period also leaks, because it overlaps the same autocorrelated regime. A gap after the
test set is required, not optional.

The band removed by purge and embargo is genuinely discarded — it belongs to neither part.
A split that reassigns it to train has reintroduced exactly the leak it was meant to remove.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from dsio.contracts import DsioModel
from dsio.data.views import TimeUnit
from dsio.data.views import window_times as window_times


class TemporalError(ValueError):
    """Raised when a temporal split is malformed or cannot be computed."""


class TimeSpan(DsioModel):
    """A half-open interval ``[start, end)`` in the split's time unit."""

    start: float
    end: float

    @model_validator(mode="after")
    def _ordered(self) -> TimeSpan:
        if self.end <= self.start:
            raise ValueError(f"span end {self.end} must exceed start {self.start}")
        return self

    def contains(self, t_start: np.ndarray, t_end: np.ndarray) -> np.ndarray:
        """Windows lying entirely within this span."""
        return (t_start >= self.start) & (t_end <= self.end)


class TemporalBounds(DsioModel):
    """The temporal half of a split: spans per part, plus the leak-prevention rules."""

    time_unit: TimeUnit = "row"
    label_horizon: float = Field(
        default=0.0,
        ge=0.0,
        description="How far past its end a window's label resolves. Drives purging.",
    )
    embargo: float = Field(
        default=0.0,
        ge=0.0,
        description="Gap after the test span before training may resume.",
    )
    spans: dict[str, list[TimeSpan]]

    @model_validator(mode="after")
    def _check(self) -> TemporalBounds:
        if not self.spans:
            raise ValueError("temporal bounds must define at least one part")
        return self


class TemporalSpec(DsioModel):
    """How to generate walk-forward folds."""

    n_splits: int = Field(default=5, ge=1)
    test_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    label_horizon: float = Field(default=0.0, ge=0.0)
    embargo: float = Field(
        default=0.0,
        ge=0.0,
        description="Absolute gap in time units. Prefer embargo_fraction for portability.",
    )
    embargo_fraction: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Embargo as a fraction of the total span; added to `embargo`.",
    )
    mode: Literal["expanding", "rolling"] = "expanding"
    time_unit: TimeUnit = "row"


def apply(
    bounds: TemporalBounds,
    t_start: np.ndarray,
    t_end: np.ndarray,
    *,
    part: str,
    test_part: str = "test",
) -> np.ndarray:
    """Boolean mask of windows belonging to ``part`` under purge and embargo.

    The test part takes windows lying entirely inside its spans. Every other part takes
    windows inside its own spans that additionally survive purging and embargo against
    *every* test span — a window is kept only if it clears all of them.
    """
    if part not in bounds.spans:
        raise TemporalError(f"no spans defined for part {part!r}")

    mask = np.zeros(t_start.shape, dtype=bool)
    for span in bounds.spans[part]:
        mask |= span.contains(t_start, t_end)
    if part == test_part:
        return mask

    for span in bounds.spans.get(test_part, []):
        # Purge: the window's label must resolve strictly before the test span opens.
        before = (t_end + bounds.label_horizon) <= span.start
        # Embargo: or the window must begin after the test span plus the embargo gap.
        after = t_start >= (span.end + bounds.embargo)
        mask &= before | after
    return mask


def walk_forward(
    t_start: np.ndarray,
    t_end: np.ndarray,
    spec: TemporalSpec,
) -> list[TemporalBounds]:
    """Build one :class:`TemporalBounds` per fold, marching test forward through time."""
    if t_start.size == 0:
        raise TemporalError("cannot build a temporal split over zero windows")

    lo = float(t_start.min())
    hi = float(t_end.max())
    total = hi - lo
    if total <= 0:
        raise TemporalError("all windows share a single instant; there is no time to split")

    embargo = spec.embargo + spec.embargo_fraction * total
    test_len = spec.test_fraction * total

    # A test span narrower than one window admits nothing: `TimeSpan.contains` keeps only
    # windows lying *entirely* inside. Without this check the split is structurally valid,
    # every fold is empty, and the failure surfaces much later as "nothing to score" with
    # no indication that the window length was the cause.
    window_len = float(np.max(t_end - t_start)) if t_start.size else 0.0
    if test_len < window_len:
        raise TemporalError(
            f"test_fraction {spec.test_fraction} gives a test span of {test_len:g} "
            f"{spec.time_unit} units, narrower than one {window_len:g}-unit window, so no "
            "window fits inside it; raise test_fraction or shorten the window"
        )

    # Space the test spans so the last one ends at the end of the data and the first
    # leaves room for a non-empty training set.
    if spec.n_splits == 1:
        origins = [hi - test_len]
    else:
        first = lo + test_len
        last = hi - test_len
        if last <= first:
            raise TemporalError(
                f"test_fraction {spec.test_fraction} is too large for {spec.n_splits} "
                "folds; the test spans would overlap the whole series"
            )
        origins = list(np.linspace(first, last, spec.n_splits))

    folds: list[TemporalBounds] = []
    for origin in origins:
        test = TimeSpan(start=float(origin), end=float(origin + test_len))
        if spec.mode == "expanding":
            train_spans = [TimeSpan(start=lo, end=hi)]
        else:
            window = max(test_len, float(origin) - lo)
            train_spans = [TimeSpan(start=max(lo, float(origin) - window), end=hi)]
        folds.append(
            TemporalBounds(
                time_unit=spec.time_unit,
                label_horizon=spec.label_horizon,
                embargo=embargo,
                spans={"train": train_spans, "test": [test]},
            )
        )
    return folds


def describe(
    bounds: TemporalBounds,
    t_start: np.ndarray,
    t_end: np.ndarray,
) -> dict[str, int]:
    """Window counts per part, plus how many the purge/embargo band discarded.

    The discarded count is worth surfacing: if it is large, the label horizon or embargo is
    eating the dataset, and that is a decision to make knowingly rather than discover from
    a training curve.
    """
    counts = {
        part: int(apply(bounds, t_start, t_end, part=part).sum()) for part in bounds.spans
    }
    counts["discarded"] = int(t_start.size - sum(counts.values()))
    return counts
