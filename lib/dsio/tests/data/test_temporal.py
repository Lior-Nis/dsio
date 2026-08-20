"""Temporal split invariants: purging, embargo, and the discarded band."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsio.data import SignalExamples, SignalStore, StoreError, WindowSpec, build_index
from dsio.splits import (
    SplitFile,
    TemporalBounds,
    TemporalError,
    TemporalSpec,
    TimeSpan,
    generate_temporal,
    resolve,
    walk_forward,
    window_times,
)
from dsio.splits.temporal import apply, describe


@pytest.fixture
def market(tmp_path: Path) -> SignalStore:
    """Three symbols, 5000 bars each, on a shared timeline."""
    path = tmp_path / "market"
    rng = np.random.default_rng(0)
    with SignalStore.builder(path, channels=2) as builder:
        for symbol in ("AAPL", "MSFT", "NVDA"):
            builder.add(
                symbol,
                rng.standard_normal((5000, 2)).astype("float32"),
                group=symbol,
                attrs={"t_start": 1_600_000_000.0, "sample_rate": 1.0},
            )
    return SignalStore(path)


@pytest.fixture
def index(market: SignalStore):
    return build_index(market, WindowSpec(length=100, stride=50))


# --- time coordinate ----------------------------------------------------------------


def test_row_time_is_relative_to_the_entity(market: SignalStore, index) -> None:
    """Global row offset would make one symbol's bar 1000 look simultaneous with another's.

    Each recording is its own timeline; a walk-forward cut on global offsets would slice
    every symbol at a different point in its own history.
    """
    t_start, _ = window_times(market, index, unit="row")
    assert t_start.min() == 0.0
    for name in index.entity_names:
        mask = index.entity_ids == name
        assert t_start[mask].min() == 0.0


def test_epoch_time_requires_the_attributes(tmp_path: Path) -> None:
    """Falling back to row order silently would produce a plausible, wrong split."""
    path = tmp_path / "bare"
    with SignalStore.builder(path, channels=1) as builder:
        builder.add("a", np.zeros((1000, 1), "float32"), group="a")
    store = SignalStore(path)
    idx = build_index(store, WindowSpec(length=100, stride=50))
    with pytest.raises(StoreError, match="t_start"):
        window_times(store, idx, unit="epoch_s")


def test_epoch_time_uses_sample_rate(market: SignalStore, index) -> None:
    t_start, t_end = window_times(market, index, unit="epoch_s")
    assert t_start.min() == 1_600_000_000.0
    assert np.allclose(t_end - t_start, 100.0)


# --- purging ------------------------------------------------------------------------


def test_purging_drops_windows_whose_label_reaches_the_test(market: SignalStore, index) -> None:
    """A window ending just before test still resolves inside it when the label looks ahead."""
    t_start, t_end = window_times(market, index, unit="row")
    spans = {"train": [TimeSpan(start=0, end=5000)], "test": [TimeSpan(start=2000, end=3000)]}

    without = apply(
        TemporalBounds(spans=spans, label_horizon=0), t_start, t_end, part="train"
    ).sum()
    with_horizon = apply(
        TemporalBounds(spans=spans, label_horizon=200), t_start, t_end, part="train"
    ).sum()
    assert with_horizon < without


def test_purged_windows_end_strictly_before_the_test(market: SignalStore, index) -> None:
    t_start, t_end = window_times(market, index, unit="row")
    bounds = TemporalBounds(
        spans={"train": [TimeSpan(start=0, end=5000)], "test": [TimeSpan(start=2000, end=3000)]},
        label_horizon=150,
    )
    mask = apply(bounds, t_start, t_end, part="train")
    kept_before = mask & (t_start < 2000)
    assert np.all(t_end[kept_before] + 150 <= 2000)


# --- embargo ------------------------------------------------------------------------


def test_embargo_creates_a_gap_after_the_test(market: SignalStore, index) -> None:
    """Serial correlation means the window right after test leaks the same regime."""
    t_start, t_end = window_times(market, index, unit="row")
    bounds = TemporalBounds(
        spans={"train": [TimeSpan(start=0, end=5000)], "test": [TimeSpan(start=2000, end=3000)]},
        embargo=300,
    )
    mask = apply(bounds, t_start, t_end, part="train")
    kept_after = mask & (t_start >= 3000)
    assert kept_after.sum() > 0
    assert t_start[kept_after].min() >= 3300


def test_embargo_zero_keeps_the_adjacent_window(market: SignalStore, index) -> None:
    t_start, t_end = window_times(market, index, unit="row")
    spans = {"train": [TimeSpan(start=0, end=5000)], "test": [TimeSpan(start=2000, end=3000)]}
    none = apply(TemporalBounds(spans=spans), t_start, t_end, part="train").sum()
    some = apply(
        TemporalBounds(spans=spans, embargo=500), t_start, t_end, part="train"
    ).sum()
    assert some < none


# --- train and test never overlap ---------------------------------------------------


def test_train_and_test_are_disjoint_in_time(market: SignalStore, index) -> None:
    t_start, t_end = window_times(market, index, unit="row")
    bounds = TemporalBounds(
        spans={"train": [TimeSpan(start=0, end=5000)], "test": [TimeSpan(start=2000, end=3000)]},
        label_horizon=50,
        embargo=100,
    )
    train = apply(bounds, t_start, t_end, part="train")
    test = apply(bounds, t_start, t_end, part="test")
    assert not (train & test).any()


def test_the_discarded_band_belongs_to_neither_part(market: SignalStore, index) -> None:
    """Reassigning the purged band to train reintroduces exactly the leak it removes."""
    t_start, t_end = window_times(market, index, unit="row")
    bounds = TemporalBounds(
        spans={"train": [TimeSpan(start=0, end=5000)], "test": [TimeSpan(start=2000, end=3000)]},
        label_horizon=200,
        embargo=200,
    )
    counts = describe(bounds, t_start, t_end)
    assert counts["discarded"] > 0
    assert counts["train"] + counts["test"] + counts["discarded"] == len(index)


# --- walk forward -------------------------------------------------------------------


def test_walk_forward_marches_test_forward(market: SignalStore, index) -> None:
    t_start, t_end = window_times(market, index, unit="row")
    folds = walk_forward(t_start, t_end, TemporalSpec(n_splits=4, test_fraction=0.15))
    origins = [b.spans["test"][0].start for b in folds]
    assert origins == sorted(origins)
    assert len(set(origins)) == 4


def test_walk_forward_rejects_an_oversized_test_fraction(market: SignalStore, index) -> None:
    t_start, t_end = window_times(market, index, unit="row")
    with pytest.raises(TemporalError, match="too large"):
        walk_forward(t_start, t_end, TemporalSpec(n_splits=3, test_fraction=0.95))


def test_walk_forward_needs_windows(market: SignalStore) -> None:
    with pytest.raises(TemporalError, match="zero windows"):
        walk_forward(np.array([]), np.array([]), TemporalSpec())


def test_a_test_span_narrower_than_one_window_is_rejected(market: SignalStore) -> None:
    """Otherwise every fold is structurally valid and empty.

    `TimeSpan.contains` keeps only windows lying entirely inside the span, so a test span
    shorter than the window length admits nothing. The split, the files and the fold all
    look fine; the failure surfaces much later as "nothing to score", pointing at the
    evaluation rather than at the window length that actually caused it.
    """
    index = build_index(market, WindowSpec(length=2000, stride=500))
    t_start, t_end = window_times(market, index, unit="row")
    with pytest.raises(TemporalError, match="narrower than one"):
        walk_forward(t_start, t_end, TemporalSpec(n_splits=3, test_fraction=0.2))


def test_embargo_fraction_scales_with_the_span(market: SignalStore, index) -> None:
    t_start, t_end = window_times(market, index, unit="row")
    folds = walk_forward(
        t_start, t_end, TemporalSpec(n_splits=2, test_fraction=0.2, embargo_fraction=0.05)
    )
    assert folds[0].embargo == pytest.approx(0.05 * (t_end.max() - t_start.min()))


# --- integration with SplitFile -----------------------------------------------------


def test_generated_temporal_split_resolves(market: SignalStore, index) -> None:
    spec = TemporalSpec(n_splits=3, test_fraction=0.2, label_horizon=20, embargo_fraction=0.02)
    folds = generate_temporal(SignalExamples(market, index), spec, name="wf")
    assert len(folds) == 3

    parts = resolve(SignalExamples(market, index), folds[0])
    assert set(parts) == {"train", "test"}
    assert len(parts["train"]) > 0 and len(parts["test"]) > 0
    assert len(parts["train"]) + len(parts["test"]) < len(index)  # the band was discarded


def test_a_temporal_split_needs_a_dataset_with_a_clock(market: SignalStore, index) -> None:
    """Returning None from times() is what makes purged splitting unavailable rather than
    silently wrong on data that has an order but no meaningful clock."""
    from dsio.data import TableExamples
    from dsio.splits import SplitError

    folds = generate_temporal(SignalExamples(market, index), TemporalSpec(n_splits=1), name="wf")
    timeless = TableExamples(
        name=market.path.name,
        groups=[str(g) for g in index.groups],
        digest=SignalExamples(market, index).digest,
    )
    with pytest.raises(SplitError, match="no time coordinates"):
        resolve(timeless, folds[0])


def test_temporal_split_round_trips(market: SignalStore, index, tmp_path: Path) -> None:
    folds = generate_temporal(SignalExamples(market, index), TemporalSpec(n_splits=2,
        label_horizon=30, embargo=40), name="wf"
    )
    path = tmp_path / "wf0.yaml"
    folds[0].save(path)
    restored = SplitFile.load(path)
    assert restored.temporal is not None
    assert restored.temporal.label_horizon == 30
    assert restored.temporal.embargo == 40
    assert restored.temporal.spans["test"][0].start == folds[0].temporal.spans["test"][0].start


def test_temporal_header_states_the_rules(market: SignalStore, index) -> None:
    folds = generate_temporal(SignalExamples(market, index), TemporalSpec(n_splits=1,
        label_horizon=25, embargo=60), name="wf"
    )
    header = folds[0].to_yaml()
    assert "# temporal: unit=row, label_horizon=25, embargo=60" in header
    assert "discarded by purge and embargo" in header


def test_temporal_composes_with_a_group_partition(market: SignalStore, index) -> None:
    """Purged walk-forward within a held-out cohort: generalise over symbols AND time."""
    groups = {"train": ["AAPL", "MSFT"], "test": ["NVDA"]}
    folds = generate_temporal(SignalExamples(market, index), TemporalSpec(n_splits=1,
        test_fraction=0.2), name="wf", groups=groups
    )
    parts = resolve(SignalExamples(market, index), folds[0])
    assert set(parts["train"].groups.tolist()) <= {"AAPL", "MSFT"}
    assert set(parts["test"].groups.tolist()) <= {"NVDA"}


def test_span_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="must exceed start"):
        TimeSpan(start=10, end=10)
