"""Store and view invariants. Each test is named for the guarantee it protects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dsio.data.adapters import SignalExamples
from dsio.data.examples import ExamplesError
from dsio.data.format import (
    FORMAT_VERSION,
    HEADER_SIZE,
    MAGIC,
    DTypeCode,
    IndexFormatError,
    IndexHeader,
)
from dsio.data.store import Entity, SignalStore, StoreError
from dsio.data.views import (
    WindowSpec,
    assert_index_matches_store,
    build_index,
    load_or_build,
)


@pytest.fixture
def store(tmp_path: Path) -> SignalStore:
    """Three groups, two sessions each, 1200 rows per session."""
    path = tmp_path / "demo"
    with SignalStore.builder(path, channels=3, dtype="float32") as builder:
        for group in range(3):
            for session in range(2):
                signal = np.arange(1200 * 3, dtype="float32").reshape(1200, 3)
                builder.add(
                    f"p{group}_s{session}", signal + group * 1000, group=f"p{group}"
                )
    return SignalStore(path)


# --- format -------------------------------------------------------------------------


def test_header_round_trips() -> None:
    header = IndexHeader(
        version=FORMAT_VERSION, dtype=np.dtype("float32"), channels=3, n_entities=5, n_rows=99
    )
    assert IndexHeader.unpack(header.pack()) == header


def test_header_is_fixed_size() -> None:
    """The offset array starts at a known position; a variable header breaks that."""
    header = IndexHeader(
        version=FORMAT_VERSION, dtype=np.dtype("int16"), channels=1, n_entities=0, n_rows=0
    )
    assert len(header.pack()) == HEADER_SIZE


def test_foreign_file_is_rejected() -> None:
    with pytest.raises(IndexFormatError, match="not a dsio index"):
        IndexHeader.unpack(b"PARQUET1" + bytes(HEADER_SIZE))


def test_future_version_is_rejected_not_guessed() -> None:
    """A format without a version check cannot be changed safely later."""
    header = IndexHeader(
        version=FORMAT_VERSION, dtype=np.dtype("float32"), channels=3, n_entities=0, n_rows=0
    )
    raw = bytearray(header.pack())
    raw[8:12] = (FORMAT_VERSION + 1).to_bytes(4, "little")
    with pytest.raises(IndexFormatError, match="not supported"):
        IndexHeader.unpack(bytes(raw))


def test_dtype_is_recorded_not_assumed() -> None:
    """Reading float32 bytes as float16 yields plausible numbers, never an error."""
    for dtype in ("float32", "float64", "float16", "int16", "int8"):
        assert DTypeCode.from_dtype(dtype).to_dtype() == np.dtype(dtype)


def test_unsupported_dtype_fails_loudly() -> None:
    with pytest.raises(IndexFormatError, match="no dsio code"):
        DTypeCode.from_dtype("complex128")


def test_magic_is_stable() -> None:
    assert MAGIC == b"DSIOIDX\x00"


# --- store --------------------------------------------------------------------------


def test_store_reads_back_exactly(store: SignalStore) -> None:
    entity = store.entity("p1_s0")
    expected = np.arange(1200 * 3, dtype="float32").reshape(1200, 3) + 1000
    assert np.array_equal(store.read_entity("p1_s0"), expected)
    assert entity.group == "p1"


def test_read_refuses_to_cross_an_entity_boundary(store: SignalStore) -> None:
    """A window spanning two recordings splices sessions that never touched."""
    boundary = store.entity("p0_s1").start_row
    with pytest.raises(StoreError, match="crosses the end of entity"):
        store.read(boundary - 10, 100)


def test_read_rejects_out_of_range(store: SignalStore) -> None:
    with pytest.raises(StoreError, match="exceeds store length"):
        store.read(store.n_rows - 10, 100)


def test_entity_at_maps_rows_back(store: SignalStore) -> None:
    entity = store.entity("p2_s1")
    assert store.entity_at(entity.start_row).entity_id == "p2_s1"
    assert store.entity_at(entity.end_row - 1).entity_id == "p2_s1"


def test_duplicate_entity_id_is_rejected(tmp_path: Path) -> None:
    builder = SignalStore.builder(tmp_path / "dupe", channels=1)
    builder.add("a", np.zeros((10, 1), "float32"), group="g")
    with pytest.raises(StoreError, match="duplicate entity_id"):
        builder.add("a", np.zeros((10, 1), "float32"), group="g")


def test_wrong_channel_count_is_rejected(tmp_path: Path) -> None:
    builder = SignalStore.builder(tmp_path / "bad", channels=3)
    with pytest.raises(StoreError, match="expected \\(n, 3\\)"):
        builder.add("a", np.zeros((10, 2), "float32"), group="g")


def test_verify_fails_closed_on_corruption(store: SignalStore) -> None:
    """Existence is not integrity; a truncated payload must not read as valid."""
    store.verify()
    payload = store.path / "signal.bin"
    raw = bytearray(payload.read_bytes())
    raw[0] ^= 0xFF
    payload.write_bytes(bytes(raw))
    with pytest.raises(StoreError, match="modified or corrupted"):
        store.verify()


def test_manifest_records_group_count(store: SignalStore) -> None:
    manifest = store.manifest()
    assert manifest.n_entities == 6
    assert manifest.n_groups == 3
    assert manifest.channels == 3


def test_store_survives_pickling(store: SignalStore) -> None:
    """DataLoader workers receive the store by pickle under the spawn start method.

    A live np.memmap would be serialised by value — the whole array — so readers must be
    dropped on pickle and reopened per process.
    """
    import pickle

    revived = pickle.loads(pickle.dumps(store))
    assert np.array_equal(revived.read(0, 100), store.read(0, 100))


def test_entities_file_is_line_oriented(store: SignalStore) -> None:
    """JSONL streams and greps; a single JSON blob does neither at millions of entities."""
    lines = (store.path / "entities.jsonl").read_text().strip().splitlines()
    assert len(lines) == 6
    assert Entity.model_validate(json.loads(lines[0])).entity_id == "p0_s0"


# --- views --------------------------------------------------------------------------


def test_windows_never_cross_entities(store: SignalStore) -> None:
    index = build_index(store, WindowSpec(length=500, stride=200))
    for start in index.starts:
        entity = store.entity_at(int(start))
        assert int(start) + 500 <= entity.end_row


def test_every_window_carries_its_group(store: SignalStore) -> None:
    """Offsets alone make leakage invisible; provenance is what makes it checkable."""
    index = build_index(store, WindowSpec(length=500, stride=200))
    assert len(index.groups) == len(index)
    assert set(index.groups.tolist()) == {"p0", "p1", "p2"}


def test_changing_the_spec_changes_the_digest(store: SignalStore) -> None:
    a = WindowSpec(length=500, stride=200)
    b = WindowSpec(length=500, stride=100)
    assert a.digest != b.digest


def test_index_round_trips(store: SignalStore, tmp_path: Path) -> None:
    index = build_index(store, WindowSpec(length=400, stride=400))
    path = tmp_path / "idx.npz"
    index.save(path)
    from dsio.data.views import WindowIndex

    restored = WindowIndex.load(path)
    assert np.array_equal(restored.starts, index.starts)
    assert restored.spec == index.spec


def test_index_is_cached_by_spec(store: SignalStore, tmp_path: Path) -> None:
    spec = WindowSpec(length=500, stride=250)
    first = load_or_build(store, spec, root=tmp_path)
    second = load_or_build(store, spec, root=tmp_path)
    assert np.array_equal(first.starts, second.starts)


def test_dense_stride_oversamples_only_marked_regions(store: SignalStore) -> None:
    """A denser stride over rare regions: more windows there, not more bytes anywhere."""
    mask = np.zeros(store.n_rows, dtype=bool)
    entity = store.entity("p0_s0")
    mask[entity.start_row + 300 : entity.start_row + 800] = True

    plain = build_index(store, WindowSpec(length=200, stride=200))
    dense = build_index(
        store, WindowSpec(length=200, stride=200, dense_stride=50), dense_mask=mask
    )
    assert len(dense) > len(plain)

    extra = set(dense.starts.tolist()) - set(plain.starts.tolist())
    assert extra, "dense stride added no windows"
    for start in extra:
        assert mask[start : start + 200].any(), "dense window landed outside the mask"


def test_label_policies_differ(store: SignalStore) -> None:
    """``any_idx.labels.sum() >= maj_idx.labels.sum()`` alone holds for *any*
    implementation of ``majority`` -- including one that is always negative, or one that
    silently reimplements ``any`` -- because ``any`` can only report more positives than a
    correct majority threshold ever would, and a broken majority policy still cannot
    exceed it. It cannot tell "the threshold is 0.5" from "majority is broken".

    The first entity (``p0_s0``, rows 0..1200) gives windows at ``build_index``'s first
    two positions (length=500, stride=500: starts 0 and 500). Setting exactly 40% of the
    first window positive and exactly 60% of the second is the one case that actually
    distinguishes the two policies at a specific window: both windows have *some*
    positive rows, so ``any`` must label both 1 -- but only the 60% window clears the
    default 0.5 majority threshold, so ``majority`` must label the 40% window 0 and the
    60% window 1.
    """
    labels = np.zeros(store.n_rows, dtype=np.int8)
    labels[:200] = 1  # window 0 (rows 0:500): ratio 0.4 -- below the majority threshold
    labels[500:800] = 1  # window 1 (rows 500:1000): ratio 0.6 -- above it

    spec_any = WindowSpec(length=500, stride=500, label_policy="any")
    spec_maj = WindowSpec(length=500, stride=500, label_policy="majority")
    any_idx = build_index(store, spec_any, labels=labels)
    maj_idx = build_index(store, spec_maj, labels=labels)

    assert any_idx.labels is not None and maj_idx.labels is not None
    assert any_idx.starts[0] == 0 and any_idx.starts[1] == 500, (
        "the fixture's window layout changed; the hand-picked row ranges above no longer "
        "land on the windows this test means to check"
    )
    assert any_idx.labels[0] == 1 and any_idx.labels[1] == 1, (
        "'any' must report a positive window whenever it has any positive row at all"
    )
    assert maj_idx.labels[0] == 0, (
        "'majority' must not report a positive window below its own threshold, even "
        "though 'any' correctly does"
    )
    assert maj_idx.labels[1] == 1, (
        "'majority' must report a positive window once it clears its threshold"
    )
    assert any_idx.labels.sum() >= maj_idx.labels.sum()


# WindowView (a numpy-only, framework-free window reader over store + index) was merged
# into dsio.dataset.dataset.WindowDataset — the torch Dataset already took the same
# constructor arguments and was the only consumer of what WindowView read. Its
# read-matches-the-store property lives on there now:
# tests/dataset/test_dataset.py::test_the_window_matches_a_direct_store_read.
#
# The store-name guard is a data-layer invariant, not a torch one, so it did not move with
# WindowView: it is dsio.data.views.assert_index_matches_store, one canonical copy of the
# check and its message. Two call sites lean on it now -- WindowDataset.__init__ (tested
# again, torch-facing, as tests/dataset/test_dataset.py::test_a_foreign_index_is_rejected)
# and SignalExamples.__init__ below, which used to carry its own hand-written copy of the
# same comparison and message before this function existed to call instead.
def test_index_built_for_a_different_store_is_rejected(
    store: SignalStore, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    with SignalStore.builder(other, channels=3) as builder:
        builder.add("x", np.zeros((900, 3), "float32"), group="g")
    index = build_index(SignalStore(other), WindowSpec(length=500, stride=200))
    with pytest.raises(ValueError, match="was built for store"):
        assert_index_matches_store(store, index)


def test_signal_examples_rejects_a_foreign_index(store: SignalStore, tmp_path: Path) -> None:
    """SignalExamples calls the same guard, but must keep raising its own exception type --

    ExamplesError, not a bare ValueError -- since callers of the Examples protocol may
    depend on that.
    """
    other = tmp_path / "other"
    with SignalStore.builder(other, channels=3) as builder:
        builder.add("x", np.zeros((900, 3), "float32"), group="g")
    index = build_index(SignalStore(other), WindowSpec(length=500, stride=200))
    with pytest.raises(ExamplesError, match="was built for store"):
        SignalExamples(store, index)


def test_subset_keeps_arrays_aligned(store: SignalStore) -> None:
    index = build_index(store, WindowSpec(length=500, stride=200))
    mask = index.groups == "p1"
    subset = index.subset(mask)
    assert len(subset) == int(mask.sum())
    assert set(subset.groups.tolist()) == {"p1"}
    assert np.array_equal(subset.starts, index.starts[mask])


def test_windows_shorter_than_an_entity_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "short"
    with SignalStore.builder(path, channels=1) as builder:
        builder.add("tiny", np.zeros((10, 1), "float32"), group="g")
        builder.add("big", np.zeros((1000, 1), "float32"), group="g")
    index = build_index(SignalStore(path), WindowSpec(length=500, stride=500))
    assert set(index.entity_ids.tolist()) == {"big"}
