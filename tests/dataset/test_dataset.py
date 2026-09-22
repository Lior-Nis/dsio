"""Raw identity-preserving windows over the memory-mapped store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dsio.data.adapters import SignalExamples, entity_examples  # noqa: E402
from dsio.data.splits.folds import folds_from_splits  # noqa: E402
from dsio.data.splits.models import SplitFile, SplitFold  # noqa: E402
from dsio.data.store import SignalStore  # noqa: E402
from dsio.data.views import WindowSpec, build_index  # noqa: E402
from dsio.dataset.dataset import (  # noqa: E402
    WindowDataset,
    make_loader,
    train_dataset,
    val_dataset,
)


@pytest.fixture
def store(tmp_path: Path) -> SignalStore:
    path = tmp_path / "cohort"
    rng = np.random.default_rng(0)
    with SignalStore.builder(path, channels=3) as builder:
        for group in range(6):
            builder.add(
                f"p{group}",
                rng.standard_normal((1000, 3)).astype("float32"),
                group=f"p{group}",
            )
    return SignalStore(path)


@pytest.fixture
def index(store: SignalStore):  # type: ignore[no-untyped-def]
    return build_index(store, WindowSpec(length=100, stride=50))


def _kfold3(store: SignalStore) -> list[SplitFile]:
    digest = entity_examples(store).digest
    folds = [
        {"test": ["p0", "p1"], "val": ["p2"], "train": ["p3", "p4", "p5"]},
        {"test": ["p2", "p3"], "val": ["p4"], "train": ["p0", "p1", "p5"]},
        {"test": ["p4", "p5"], "val": ["p0"], "train": ["p1", "p2", "p3"]},
    ]
    return [
        SplitFile(
            store=store.path.name,
            store_manifest_sha256=digest,
            name="k3",
            folds=[
                SplitFold(
                    index=i,
                    counts={part: len(members) for part, members in parts.items()},
                    parts=parts,
                )
            ],
        )
        for i, parts in enumerate(folds)
    ]


def test_items_are_channels_first(store: SignalStore, index) -> None:  # type: ignore[no-untyped-def]
    assert WindowDataset(store, index)[0]["x"].shape == (3, 100)


def test_items_carry_the_position_they_came_from(store: SignalStore, index) -> None:  # type: ignore[no-untyped-def]
    dataset = WindowDataset(store, index, positions=np.array([5, 9, 2]))
    assert [dataset[i]["row"] for i in range(3)] == [5, 9, 2]


def test_window_items_use_the_governed_identity(store: SignalStore, index) -> None:  # type: ignore[no-untyped-def]
    governed = SignalExamples(store, index).sample_ids.tolist()
    positions = np.array([5, 2, 9])
    reordered = WindowDataset(store, index, positions=positions)
    assert [reordered[i]["sample_id"] for i in range(len(reordered))] == [
        governed[position] for position in positions
    ]


def test_the_window_matches_a_direct_store_read(store: SignalStore, index) -> None:  # type: ignore[no-untyped-def]
    dataset = WindowDataset(store, index, positions=np.array([7]))
    expected = store.read(int(index.starts[7]), 100)
    assert np.array_equal(dataset[0]["x"].numpy(), expected.T)


def test_a_fold_costs_positions_not_a_dataset(store: SignalStore, index) -> None:  # type: ignore[no-untyped-def]
    folds = folds_from_splits(SignalExamples(store, index), _kfold3(store))
    datasets = [WindowDataset(store, index, fold.train) for fold in folds]
    assert all(dataset.store is store and dataset.index is index for dataset in datasets)
    for fold in folds:
        parts = [fold.train, fold.test, fold.val]
        assert sum(part.size for part in parts if part is not None) == len(index)


def test_labels_are_indexed_by_whole_index_not_by_fold(store: SignalStore, index) -> None:  # type: ignore[no-untyped-def]
    labels = np.arange(len(index), dtype=np.float32)
    dataset = WindowDataset(store, index, positions=np.array([11, 4]), labels=labels)
    assert [float(dataset[i]["y"]) for i in range(2)] == [11.0, 4.0]


def test_fold_length_labels_are_rejected(store: SignalStore, index) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="aligned with the whole index"):
        WindowDataset(store, index, positions=np.array([1, 2]), labels=np.zeros(2))


def test_a_foreign_index_is_rejected(store: SignalStore, index, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    other = tmp_path / "other"
    with SignalStore.builder(other, channels=3) as builder:
        builder.add("q0", np.zeros((500, 3), "float32"), group="q0")
    with pytest.raises(ValueError, match="was built for store"):
        WindowDataset(SignalStore(other), index)


def test_groups_are_reachable_for_leak_checking(store: SignalStore, index) -> None:  # type: ignore[no-untyped-def]
    fold = folds_from_splits(SignalExamples(store, index), _kfold3(store))[0]
    train = WindowDataset(store, index, fold.train)
    test = WindowDataset(store, index, fold.test)
    assert not (set(train.groups) & set(test.groups))


@pytest.fixture
def token_store(tmp_path: Path) -> SignalStore:
    path = tmp_path / "corpus"
    rng = np.random.default_rng(0)
    with SignalStore.builder(path, channels=1, dtype="int32") as builder:
        for doc, length in enumerate((300, 240, 180)):
            ids = rng.integers(0, 64, size=(length, 1), dtype=np.int32)
            builder.add(f"doc{doc}", ids, group=f"doc{doc}")
    return SignalStore(path)


@pytest.fixture
def token_index(token_store: SignalStore):  # type: ignore[no-untyped-def]
    return build_index(token_store, WindowSpec(length=32, stride=16))


def test_the_payload_is_float32_by_default(token_store: SignalStore, token_index) -> None:  # type: ignore[no-untyped-def]
    assert WindowDataset(token_store, token_index)[0]["x"].dtype is torch.float32


def test_an_integer_payload_dtype_survives_to_the_item(
    token_store: SignalStore,
    token_index,  # type: ignore[no-untyped-def]
) -> None:
    item = WindowDataset(token_store, token_index, payload_dtype=torch.long)[0]
    assert item["x"].dtype is torch.int64
    assert np.array_equal(item["x"].numpy(), token_store.read(int(token_index.starts[0]), 32).T)


def test_an_integer_payload_survives_a_loader(token_store: SignalStore, token_index) -> None:  # type: ignore[no-untyped-def]
    dataset = WindowDataset(token_store, token_index, payload_dtype=torch.long)
    batch = next(iter(make_loader(dataset, batch_size=4)))
    assert batch["x"].dtype is torch.int64
    assert batch["x"].shape == (4, 1, 32)


def test_the_builders_pass_the_payload_dtype_through(
    token_store: SignalStore,
    token_index,  # type: ignore[no-untyped-def]
) -> None:
    train = train_dataset(token_store, token_index, payload_dtype=torch.long)
    val = val_dataset(token_store, token_index, payload_dtype=torch.long)
    assert train[0]["x"].dtype is torch.int64
    assert val[0]["x"].dtype is torch.int64


def test_loader_result_does_not_depend_on_worker_count(store: SignalStore, index) -> None:  # type: ignore[no-untyped-def]
    positions = np.arange(30)

    def load(workers: int) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        batches = list(
            make_loader(
                WindowDataset(store, index, positions),
                batch_size=7,
                shuffle=True,
                num_workers=workers,
                seed=42,
            )
        )
        return (
            [sample_id for batch in batches for sample_id in batch["sample_id"]],
            torch.cat([batch["x"] for batch in batches]),
            torch.cat([batch["row"] for batch in batches]),
        )

    zero, two = load(0), load(2)
    assert zero[0] == two[0]
    assert torch.equal(zero[1], two[1])
    assert torch.equal(zero[2], two[2])


def test_the_seed_changes_shuffle_order(store: SignalStore, index) -> None:  # type: ignore[no-untyped-def]
    dataset = WindowDataset(store, index, np.arange(30))
    first = torch.cat([batch["row"] for batch in make_loader(dataset, shuffle=True, seed=1)])
    second = torch.cat([batch["row"] for batch in make_loader(dataset, shuffle=True, seed=2)])
    assert not torch.equal(first, second)


def test_an_unshuffled_loader_covers_every_position_once(store: SignalStore, index) -> None:  # type: ignore[no-untyped-def]
    positions = np.array([7, 2, 11, 4, 19])
    rows = torch.cat(
        [batch["row"] for batch in make_loader(WindowDataset(store, index, positions))]
    )
    assert rows.tolist() == positions.tolist()


def test_workers_read_the_store_independently(store: SignalStore, index) -> None:  # type: ignore[no-untyped-def]
    loader = make_loader(WindowDataset(store, index, np.arange(20)), batch_size=4, num_workers=2)
    assert sum(batch["x"].shape[0] for batch in loader) == 20
