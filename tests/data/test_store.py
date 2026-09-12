"""Store and view invariants. Each test is named for the guarantee it protects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dsio.data.adapters import SignalExamples, entity_examples
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
    ViewError,
    WindowIndex,
    WindowSpec,
    assert_index_matches_store,
    build_index,
    index_path,
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


def test_builder_refuses_to_overwrite_a_store_without_changing_it(tmp_path: Path) -> None:
    path = tmp_path / "existing"
    with SignalStore.builder(path, channels=1) as builder:
        builder.add("a", np.arange(10, dtype="float32").reshape(10, 1), group="g")
    before = {
        item.relative_to(path): item.read_bytes() for item in path.rglob("*") if item.is_file()
    }

    with pytest.raises(StoreError, match="not empty"):
        SignalStore.builder(path, channels=1)

    after = {
        item.relative_to(path): item.read_bytes() for item in path.rglob("*") if item.is_file()
    }
    assert after == before


def test_builder_refuses_an_unrelated_non_empty_directory(tmp_path: Path) -> None:
    path = tmp_path / "occupied"
    path.mkdir()
    marker = path / "keep.txt"
    marker.write_bytes(b"do not touch")

    with pytest.raises(StoreError, match="not empty"):
        SignalStore.builder(path, channels=1)

    assert list(path.iterdir()) == [marker]
    assert marker.read_bytes() == b"do not touch"


def test_builder_accepts_an_existing_empty_directory(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.mkdir()

    with SignalStore.builder(path, channels=1) as builder:
        builder.add("a", np.zeros((4, 1), dtype="float32"), group="g")

    SignalStore(path).verify()


def test_competing_builders_cannot_open_the_same_destination(tmp_path: Path) -> None:
    path = tmp_path / "race"
    with SignalStore.builder(path, channels=1) as winner:
        with pytest.raises(StoreError, match="not empty"):
            SignalStore.builder(path, channels=1)
        winner.add("a", np.zeros((4, 1), dtype="float32"), group="g")

    SignalStore(path).verify()


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


def test_adapter_identities_require_a_manifest(store: SignalStore) -> None:
    examples = SignalExamples(store, build_index(store, WindowSpec(length=500, stride=500)))
    (store.path / "manifest.yaml").unlink()

    with pytest.raises(StoreError, match="manifest.yaml"):
        _ = examples.digest
    with pytest.raises(StoreError, match="manifest.yaml"):
        entity_examples(store)


def test_adapter_identities_reject_a_corrupt_manifest(store: SignalStore) -> None:
    examples = SignalExamples(store, build_index(store, WindowSpec(length=500, stride=500)))
    (store.path / "manifest.yaml").write_text("channels: [")

    with pytest.raises(StoreError, match="cannot read manifest"):
        _ = examples.digest
    with pytest.raises(StoreError, match="cannot read manifest"):
        entity_examples(store)


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
    restored = WindowIndex.load(path)
    assert np.array_equal(restored.starts, index.starts)
    assert restored.spec == index.spec
    assert restored.store_digest == index.store_digest


def test_index_is_cached_by_spec(store: SignalStore, tmp_path: Path) -> None:
    spec = WindowSpec(length=500, stride=250)
    first = load_or_build(store, spec, root=tmp_path)
    second = load_or_build(store, spec, root=tmp_path)
    assert np.array_equal(first.starts, second.starts)


def test_index_cache_key_includes_the_labels_array(store: SignalStore, tmp_path: Path) -> None:
    """Critical 2: `load_or_build` bakes `labels=` into the cached index, but the cache key
    used to be the `WindowSpec` digest alone. Two calls against the same store and spec that
    differ only in which label provider they passed must not collide on one cache file --
    the second caller would otherwise silently score against the first caller's labels."""
    spec = WindowSpec(length=500, stride=250, label_policy="majority")
    labels_a = np.zeros(store.n_rows, dtype=np.float32)
    labels_b = np.ones(store.n_rows, dtype=np.float32)

    built_a = load_or_build(store, spec, labels=labels_a, root=tmp_path)
    built_b = load_or_build(store, spec, labels=labels_b, root=tmp_path)

    assert index_path(store, spec, tmp_path, labels=labels_a) != index_path(
        store, spec, tmp_path, labels=labels_b
    )
    assert built_a.labels is not None and built_b.labels is not None
    assert np.all(built_a.labels == 0)
    assert np.all(built_b.labels == 1)


def test_index_cache_key_includes_the_dense_mask(store: SignalStore, tmp_path: Path) -> None:
    spec = WindowSpec(length=200, stride=200, dense_stride=50)
    sparse = np.zeros(store.n_rows, dtype=bool)
    dense = sparse.copy()
    dense[300:800] = True

    load_or_build(store, spec, dense_mask=sparse, root=tmp_path)
    cached = load_or_build(store, spec, dense_mask=dense, root=tmp_path)
    expected = build_index(store, spec, dense_mask=dense)

    assert index_path(store, spec, tmp_path, dense_mask=sparse) != index_path(
        store, spec, tmp_path, dense_mask=dense
    )
    assert np.array_equal(cached.starts, expected.starts)
    assert np.array_equal(cached.entity_codes, expected.entity_codes)


def test_index_cache_key_includes_row_metrics(store: SignalStore, tmp_path: Path) -> None:
    spec = WindowSpec(length=500, stride=500, min_metrics={"quality": 0.5})
    low = {"quality": np.zeros(store.n_rows)}
    high = {"quality": np.ones(store.n_rows)}

    load_or_build(store, spec, row_metrics=low, root=tmp_path)
    cached = load_or_build(store, spec, row_metrics=high, root=tmp_path)
    expected = build_index(store, spec, row_metrics=high)

    assert index_path(store, spec, tmp_path, row_metrics=low) != index_path(
        store, spec, tmp_path, row_metrics=high
    )
    assert np.array_equal(cached.starts, expected.starts)
    assert np.array_equal(cached.metrics["quality"], expected.metrics["quality"])


def test_index_cache_identity_is_semantic_and_order_independent(
    store: SignalStore, tmp_path: Path
) -> None:
    spec = WindowSpec(length=500, stride=500, label_policy="any", dense_stride=250)
    values = np.arange(store.n_rows * 2, dtype=np.float64).reshape(store.n_rows, 2)
    non_contiguous = values[:, 0]
    contiguous = non_contiguous.copy()
    labels = non_contiguous != 0
    mask = labels.copy()

    first = index_path(
        store,
        spec,
        tmp_path,
        dense_mask=mask,
        labels=labels,
        row_metrics={"second": contiguous, "first": non_contiguous},
    )
    second = index_path(
        store,
        spec,
        tmp_path,
        dense_mask=mask.copy(),
        labels=labels.copy(),
        row_metrics={"first": contiguous, "second": non_contiguous.copy()},
    )

    assert first == second


def test_inactive_inputs_do_not_change_the_cache_identity(
    store: SignalStore, tmp_path: Path
) -> None:
    spec = WindowSpec(length=500, stride=500)
    baseline = index_path(store, spec, tmp_path)
    with_inactive_inputs = index_path(
        store,
        spec,
        tmp_path,
        dense_mask=np.ones(store.n_rows, dtype=bool),
        labels=np.ones(store.n_rows),
    )
    assert baseline == with_inactive_inputs


def test_index_cache_key_includes_store_topology(tmp_path: Path) -> None:
    signal = np.arange(8, dtype=np.float32).reshape(8, 1)
    first_path = tmp_path / "first" / "same"
    with SignalStore.builder(first_path, channels=1) as builder:
        builder.add("whole", signal, group="one")
    second_path = tmp_path / "second" / "same"
    with SignalStore.builder(second_path, channels=1) as builder:
        builder.add("left", signal[:4], group="two")
        builder.add("right", signal[4:], group="two")

    first_store = SignalStore(first_path)
    second_store = SignalStore(second_path)
    spec = WindowSpec(length=4, stride=4)
    root = tmp_path / "views"
    first = load_or_build(first_store, spec, root=root)
    second = load_or_build(second_store, spec, root=root)

    assert first_store.manifest().signal_sha256 == second_store.manifest().signal_sha256
    assert index_path(first_store, spec, root) != index_path(second_store, spec, root)
    assert first.entity_names == ["whole"]
    assert second.entity_names == ["left", "right"]


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


def test_same_named_store_with_different_signal_is_rejected(tmp_path: Path) -> None:
    first_path = tmp_path / "first" / "same"
    with SignalStore.builder(first_path, channels=1) as builder:
        builder.add("item", np.zeros((8, 1), dtype=np.float32), group="group")
    second_path = tmp_path / "second" / "same"
    with SignalStore.builder(second_path, channels=1) as builder:
        builder.add("item", np.ones((8, 1), dtype=np.float32), group="group")

    index = build_index(SignalStore(first_path), WindowSpec(length=4, stride=4))
    with pytest.raises(ViewError, match="content"):
        assert_index_matches_store(SignalStore(second_path), index)


def test_store_entity_metadata_is_part_of_index_compatibility(tmp_path: Path) -> None:
    signal = np.zeros((8, 1), dtype=np.float32)
    first_path = tmp_path / "first" / "same"
    with SignalStore.builder(first_path, channels=1) as builder:
        builder.add("original", signal, group="first")
    second_path = tmp_path / "second" / "same"
    with SignalStore.builder(second_path, channels=1) as builder:
        builder.add("renamed", signal, group="second")

    first = SignalStore(first_path)
    second = SignalStore(second_path)
    assert first.manifest().signal_sha256 == second.manifest().signal_sha256
    assert first.manifest().index_sha256 == second.manifest().index_sha256

    index = build_index(first, WindowSpec(length=4, stride=4))
    with pytest.raises(ViewError, match="content"):
        assert_index_matches_store(second, index)


def test_index_remains_compatible_when_its_store_moves(tmp_path: Path) -> None:
    original = tmp_path / "original"
    with SignalStore.builder(original, channels=1) as builder:
        builder.add("item", np.zeros((8, 1), dtype=np.float32), group="group")
    index = build_index(SignalStore(original), WindowSpec(length=4, stride=4))

    moved = tmp_path / "renamed"
    original.rename(moved)
    assert_index_matches_store(SignalStore(moved), index)


def test_cache_hit_validates_the_index_store_binding(tmp_path: Path) -> None:
    expected_path = tmp_path / "expected"
    with SignalStore.builder(expected_path, channels=1) as builder:
        builder.add("item", np.zeros((8, 1), dtype=np.float32), group="group")
    foreign_path = tmp_path / "foreign"
    with SignalStore.builder(foreign_path, channels=1) as builder:
        builder.add("item", np.ones((8, 1), dtype=np.float32), group="group")

    expected_store = SignalStore(expected_path)
    spec = WindowSpec(length=4, stride=4)
    cache_path = index_path(expected_store, spec, tmp_path / "views")
    build_index(SignalStore(foreign_path), spec).save(cache_path)

    with pytest.raises(ViewError, match="content"):
        load_or_build(expected_store, spec, root=tmp_path / "views")


def test_loading_index_without_store_provenance_fails_clearly(
    store: SignalStore, tmp_path: Path
) -> None:
    path = tmp_path / "legacy.npz"
    build_index(store, WindowSpec(length=500, stride=500)).save(path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("store_digest")
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ViewError, match="store provenance.*rebuild"):
        WindowIndex.load(path)


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
    assert subset.store_digest == index.store_digest


def test_windows_shorter_than_an_entity_are_skipped_only_when_asked(tmp_path: Path) -> None:
    """This test used to assert the skip alone, which is how the silence got in.

    A recording too short for one window genuinely cannot be windowed, so ``tiny`` is
    absent from the index either way -- but the index cannot say so, and asserting only
    that ``big`` survived is agreeing not to ask. The behaviour is unchanged under the
    explicit opt-out; what changed is that it has to be asked for. The refusal and the
    accounting are covered in tests/data/test_discards.py.
    """
    path = tmp_path / "short"
    with SignalStore.builder(path, channels=1) as builder:
        builder.add("tiny", np.zeros((10, 1), "float32"), group="g")
        builder.add("big", np.zeros((1000, 1), "float32"), group="g")
    store = SignalStore(path)
    spec = WindowSpec(length=500, stride=500)

    with pytest.raises(ValueError, match="tiny"):
        build_index(store, spec)

    index = build_index(store, spec, on_dropped_entities="drop")
    assert set(index.entity_ids.tolist()) == {"big"}
