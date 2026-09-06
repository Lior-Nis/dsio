"""What `build_index` leaves out of the index, and which omissions it refuses.

The bug these protect against: a store of eight entities and a 64-row window kept 320 of
634 rows -- three whole recordings absent from the index, four truncated -- and every
check in the codebase passed. The examples contract, `assert_index_matches_store`, the
dataset and the loader all reported a perfectly healthy half-corpus.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsio.data.store import SignalStore
from dsio.data.views import (
    ViewError,
    WindowSpec,
    build_index,
    load_or_build,
    window_discards,
)

# The spike's corpus: three entities cannot hold a single 64-row window, four lose a
# trailing remainder, one fits exactly.
SPIKE_SIZES = (100, 81, 121, 49, 49, 64, 121, 49)


def _store(tmp_path: Path, sizes: tuple[int, ...], name: str = "spike") -> SignalStore:
    path = tmp_path / name
    with SignalStore.builder(path, channels=1) as builder:
        for i, rows in enumerate(sizes):
            builder.add(f"v{i}", np.zeros((rows, 1), "float32"), group=f"g{i}")
    return SignalStore(path)


def test_an_entity_too_short_for_one_window_is_refused(tmp_path: Path) -> None:
    """The unambiguous half of the loss: a recording absent from the index entirely.

    Nothing downstream can notice. The index is internally consistent, the store is
    intact, and the entity is simply not mentioned -- so the only place this can be
    caught is where it happens.
    """
    store = _store(tmp_path, SPIKE_SIZES)
    with pytest.raises(ViewError) as error:
        build_index(store, WindowSpec(length=64, stride=64))
    assert "v3" in str(error.value)


def test_the_refusal_names_the_entities_the_rows_and_the_remedy(tmp_path: Path) -> None:
    """A refusal that does not say which recordings were lost is a different silence."""
    store = _store(tmp_path, SPIKE_SIZES)
    with pytest.raises(ViewError) as error:
        build_index(store, WindowSpec(length=64, stride=64))
    message = str(error.value)
    for name in ("v3", "v4", "v7"):
        assert name in message, f"the refusal does not name lost entity {name}"
    assert "49" in message, "the refusal does not say how many rows were lost"
    assert "length" in message, "the refusal does not name the parameter to lower"
    assert "drop" in message, "the refusal does not name the explicit opt-out"


def test_dropping_short_entities_can_be_asked_for_explicitly(tmp_path: Path) -> None:
    """Dropping recordings shorter than a window is a legitimate choice, once chosen."""
    store = _store(tmp_path, SPIKE_SIZES)
    index = build_index(store, WindowSpec(length=64, stride=64), on_dropped_entities="drop")
    assert set(index.entity_ids.tolist()) == {"v0", "v1", "v2", "v5", "v6"}


def test_a_store_that_loses_nothing_builds_exactly_as_before(tmp_path: Path) -> None:
    """The guard must be invisible where there is nothing to report.

    Every entity here yields at least one window, so the index -- offsets, codes, digest
    -- must be byte-identical to what the unguarded implementation produced. Existing
    committed splits and staged indices are keyed by that digest.
    """
    store = _store(tmp_path, (128, 128), name="whole")
    spec = WindowSpec(length=64, stride=64)
    index = build_index(store, spec)
    assert index.starts.tolist() == [0, 64, 128, 192]
    assert index.digest == spec.digest
    assert window_discards(store, spec).discarded_rows == 0


def test_the_guard_leaves_the_spec_digest_alone() -> None:
    """Committed splits and staged indices are addressed by these exact strings.

    The policy is a keyword on the call, never a field on the spec, precisely so that
    these do not move: a spec that grew a field would rename every index on disk and
    orphan every split committed against one.
    """
    assert WindowSpec(length=64, stride=64).digest == "6d3891546387"
    assert WindowSpec(length=128, stride=64, drop_last_partial=False).digest == "53245c15c44d"


def test_a_trailing_remainder_is_reported_not_refused(tmp_path: Path) -> None:
    """`drop_last_partial` is documented, intentional and right for a waveform.

    A partial trailing window is meaningless on a signal, so truncating one is not an
    error -- but it is not nothing either, and a caller asking what it cost must get an
    answer rather than an empty one.
    """
    store = _store(tmp_path, (100,), name="one")
    spec = WindowSpec(length=64, stride=64)
    index = build_index(store, spec)  # must not raise
    assert len(index) == 1

    report = window_discards(store, spec)
    assert report.dropped == ()
    assert [entity.entity_id for entity in report.truncated] == ["v0"]
    assert report.truncated[0].discarded_rows == 36


def test_the_report_accounts_for_every_row_in_the_store(tmp_path: Path) -> None:
    """The spike's arithmetic: 320 of 634 rows kept, and the report must say so."""
    store = _store(tmp_path, SPIKE_SIZES)
    report = window_discards(store, WindowSpec(length=64, stride=64))

    assert report.n_rows == sum(SPIKE_SIZES) == 634
    assert report.covered_rows == 320
    assert report.discarded_rows == 634 - 320
    assert report.covered_rows + report.discarded_rows == report.n_rows
    assert [entity.entity_id for entity in report.dropped] == ["v3", "v4", "v7"]
    assert sum(entity.discarded_rows for entity in report.dropped) == 147


def test_overlapping_windows_cover_a_row_once(tmp_path: Path) -> None:
    """A stride below the window length re-reads rows; counting them twice would let a
    truncating view report more covered rows than the store has."""
    store = _store(tmp_path, (100,), name="overlap")
    report = window_discards(store, WindowSpec(length=64, stride=16))
    # starts 0, 16, 32 -> rows 0..96 covered, 4 left over.
    assert report.covered_rows == 96
    assert report.discarded_rows == 4


def test_a_dense_stride_fills_the_gaps_it_samples(tmp_path: Path) -> None:
    """The report is of the windows actually enumerated, not of the plain stride alone.

    A dense stride adds windows inside a marked region. Reporting coverage from the plain
    stride would then overstate what was discarded, and a caller would go looking for rows
    the index does hold.
    """
    store = _store(tmp_path, (400,), name="dense")
    spec = WindowSpec(length=64, stride=128, dense_stride=32)
    mask = np.zeros(store.n_rows, dtype=bool)
    mask[150:200] = True

    plain = window_discards(store, WindowSpec(length=64, stride=128))
    dense = window_discards(store, spec, dense_mask=mask)
    assert dense.covered_rows > plain.covered_rows


def test_load_or_build_refuses_a_cached_index_that_lost_entities(tmp_path: Path) -> None:
    """The cache must not launder the loss.

    The opt-out is a caller's decision, not a property of the index, and it is not part of
    the cache key -- it must not be, since the key is what binds an index to committed
    splits. So a caller who did not opt in has to be refused even when an earlier caller
    who did opt in already left an index on disk.
    """
    store = _store(tmp_path, SPIKE_SIZES)
    spec = WindowSpec(length=64, stride=64)
    root = tmp_path / "views"
    load_or_build(store, spec, root=root, on_dropped_entities="drop")
    with pytest.raises(ViewError):
        load_or_build(store, spec, root=root)


def test_an_empty_report_is_falsy_and_a_lossy_one_is_not(tmp_path: Path) -> None:
    """So `if window_discards(...):` reads as "did this cost anything"."""
    spec = WindowSpec(length=64, stride=64)
    assert not window_discards(_store(tmp_path, (128,), name="whole"), spec)
    assert window_discards(_store(tmp_path, (100,), name="part"), spec)


# --- metric floors: countable when asked, never refused ---------------------------------


def _graded_store(tmp_path: Path) -> tuple[SignalStore, np.ndarray]:
    """Two 64-row entities, one of which every window's purity floor will reject."""
    path = tmp_path / "graded"
    with SignalStore.builder(path, channels=1) as builder:
        for name in ("clean", "dirty"):
            builder.add(name, np.zeros((64, 1), "float32"), group=name)
    purity = np.concatenate([np.ones(64), np.zeros(64)])
    return SignalStore(path), purity


def test_a_metric_floor_that_empties_an_entity_is_counted_when_asked(
    tmp_path: Path,
) -> None:
    """The same loss as a too-short recording, by a different route.

    A caller writes ``min_metrics={'purity': 0.9}``; they do not write "and remove this
    subject from the study". The filter is deliberate, that consequence is not, and the
    damage is identical -- a group absent from every fold while every check passes. The
    spec digest records the filter, so this is reported rather than refused, but it must
    be *reportable*: before this, ``window_discards`` could not see it at all.
    """
    store, purity = _graded_store(tmp_path)
    spec = WindowSpec(length=64, stride=64, min_metrics={"purity": 0.9})

    report = window_discards(store, spec, row_metrics={"purity": purity})

    assert [e.entity_id for e in report.dropped] == ["dirty"]
    assert report.discarded_rows == 64


def test_metric_floors_are_invisible_without_the_metrics(tmp_path: Path) -> None:
    """``window_discards`` is a function of the store and the spec, and stays one.

    Without ``row_metrics`` it reports the geometry only, exactly as before -- the caller
    who has not supplied the metrics is not told about a filter that needs them.
    """
    store, _ = _graded_store(tmp_path)
    spec = WindowSpec(length=64, stride=64, min_metrics={"purity": 0.9})

    report = window_discards(store, spec)

    assert not report.dropped
    assert report.discarded_rows == 0


def test_a_metric_floor_never_refuses_a_build(tmp_path: Path) -> None:
    """Reporting, not refusing: the caller asked for the filter.

    Refusing would be a guess about how aggressively a project filters, and filtering a
    recording to nothing may be exactly what a purity floor is for. Reporting cannot be
    wrong in the way refusing can.
    """
    store, purity = _graded_store(tmp_path)
    spec = WindowSpec(length=64, stride=64, min_metrics={"purity": 0.9})

    index = build_index(store, spec, row_metrics={"purity": purity})

    assert len(index) == 1
    assert set(index.entity_ids) == {"clean"}


# --- one window per entity: the guarantee fixed-size item corpora rest on ----------------


def test_an_item_corpus_asserts_its_one_to_one_correspondence(tmp_path: Path) -> None:
    """Equal-length entities at ``length == stride == item length``: one window each."""
    store = _store(tmp_path, (64, 64, 64), name="items")
    index = build_index(store, WindowSpec(length=64, stride=64), one_window_per_entity=True)
    assert len(index) == 3


def test_an_oversized_item_is_refused_when_one_to_one_is_claimed(tmp_path: Path) -> None:
    """The hole the too-short guard leaves open, and the one that bites a vision corpus.

    An item *larger* than the window is not lost -- it becomes several windows that are
    not items. Nothing downstream can tell: the index is consistent, the loader yields
    tensors of the right shape, and every one of them is a crop of an image rather than
    an image. Only the caller knows a window was supposed to be a whole item.
    """
    store = _store(tmp_path, (64, 144, 64), name="ragged")
    with pytest.raises(ViewError, match="2 windows"):
        build_index(store, WindowSpec(length=64, stride=64), one_window_per_entity=True)


def test_the_refusal_names_the_offending_entities_and_the_remedy(tmp_path: Path) -> None:
    store = _store(tmp_path, (64, 144, 64), name="ragged")
    with pytest.raises(ViewError) as excinfo:
        build_index(store, WindowSpec(length=64, stride=64), one_window_per_entity=True)
    message = str(excinfo.value)
    assert "v1" in message
    assert "144" in message
    assert "resize" in message.lower()


def test_one_to_one_is_not_claimed_by_default(tmp_path: Path) -> None:
    """A waveform corpus wants many windows per recording; that is the normal case."""
    store = _store(tmp_path, (64, 144, 64), name="ragged")
    index = build_index(store, WindowSpec(length=64, stride=64))
    assert len(index) == 4


def test_an_entity_with_no_windows_fails_the_claim_even_when_dropping_is_allowed(
    tmp_path: Path,
) -> None:
    """``on_dropped_entities='drop'`` and ``one_window_per_entity=True`` contradict.

    Dropping says "recordings shorter than a window are meant to go"; one-to-one says
    "every entity is exactly one item". The stronger claim is the one that must hold.
    """
    store = _store(tmp_path, (64, 20, 64), name="short")
    with pytest.raises(ViewError, match="0 windows"):
        build_index(
            store,
            WindowSpec(length=64, stride=64),
            on_dropped_entities="drop",
            one_window_per_entity=True,
        )


def test_a_cached_index_cannot_launder_a_broken_correspondence(tmp_path: Path) -> None:
    """The same laundering `on_dropped_entities` already closes, for the same reason."""
    store = _store(tmp_path, (64, 144, 64), name="ragged")
    spec = WindowSpec(length=64, stride=64)
    load_or_build(store, spec, root=tmp_path / "views")

    with pytest.raises(ViewError, match="2 windows"):
        load_or_build(store, spec, root=tmp_path / "views", one_window_per_entity=True)


def test_the_claim_leaves_the_spec_digest_alone(tmp_path: Path) -> None:
    """It is a claim about the corpus, not a property of the view.

    A digest that moved would invalidate every cached index and every committed split the
    moment someone asserted a correspondence that already held.
    """
    store = _store(tmp_path, (64, 64), name="items")
    spec = WindowSpec(length=64, stride=64)
    plain = build_index(store, spec)
    claimed = build_index(store, spec, one_window_per_entity=True)
    assert plain.digest == claimed.digest == spec.digest
