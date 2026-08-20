"""WindowDataset: fold positions over a memory-mapped store, without copying it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dsio.data.adapters import SignalExamples, entity_examples  # noqa: E402
from dsio.data.store import SignalStore  # noqa: E402
from dsio.data.views import WindowSpec, build_index  # noqa: E402
from dsio.nn.data import WindowDataset, make_loader  # noqa: E402
from dsio.splits.folds import folds_from_splits  # noqa: E402
from dsio.splits.models import SplitFile  # noqa: E402


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
def index(store: SignalStore):
    return build_index(store, WindowSpec(length=100, stride=50))


def _kfold3(store: SignalStore) -> list[SplitFile]:
    """Three hand-picked folds over the store's six groups, each a train/val/test split."""
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
            fold=i,
            counts={part: len(members) for part, members in parts.items()},
            parts=parts,
        )
        for i, parts in enumerate(folds)
    ]


def test_items_are_channels_first(store: SignalStore, index) -> None:
    """The store is [time, channels]; torch convolutions want [channels, time]."""
    item = WindowDataset(store, index)[0]
    assert item["x"].shape == (3, 100)


def test_items_carry_the_position_they_came_from(store: SignalStore, index) -> None:
    dataset = WindowDataset(store, index, positions=np.array([5, 9, 2]))
    assert [dataset[i]["row"] for i in range(3)] == [5, 9, 2]


def test_the_window_matches_a_direct_store_read(store: SignalStore, index) -> None:
    """The dataset must not quietly transform the signal on its way out."""
    dataset = WindowDataset(store, index, positions=np.array([7]))
    expected = store.read(int(index.starts[7]), 100)
    assert np.array_equal(dataset[0]["x"].numpy(), expected.T)


def test_a_fold_costs_positions_not_a_dataset(store: SignalStore, index) -> None:
    """The payoff of the index layer: five folds are five position arrays, not five copies."""
    splits = _kfold3(store)
    folds = folds_from_splits(SignalExamples(store, index), splits)
    datasets = [WindowDataset(store, index, fold.train) for fold in folds]
    assert all(dataset.store is store for dataset in datasets), "one store, shared"
    assert all(dataset.index is index for dataset in datasets), "one index, shared"
    # k=3 gives each fold a train, a val and a test bucket, so the three parts of any
    # single fold partition the index exactly.
    for fold in folds:
        parts = [fold.train, fold.test, fold.val]
        assert sum(part.size for part in parts if part is not None) == len(index)


def test_labels_are_indexed_by_whole_index_not_by_fold(store: SignalStore, index) -> None:
    """The subtle one. Labels align with the index; positions select into them.

    Passing fold-length labels would line up silently for fold 0 — whose positions often
    start at 0 — and be wrong for every other fold.
    """
    labels = np.arange(len(index), dtype=np.float32)
    dataset = WindowDataset(store, index, positions=np.array([11, 4]), labels=labels)
    assert [float(dataset[i]["y"]) for i in range(2)] == [11.0, 4.0]


def test_fold_length_labels_are_rejected(store: SignalStore, index) -> None:
    with pytest.raises(ValueError, match="aligned with the whole index"):
        WindowDataset(store, index, positions=np.array([1, 2]), labels=np.zeros(2))


def test_a_foreign_index_is_rejected(store: SignalStore, index, tmp_path: Path) -> None:
    other = tmp_path / "other"
    with SignalStore.builder(other, channels=3) as builder:
        builder.add("q0", np.zeros((500, 3), "float32"), group="q0")
    with pytest.raises(ValueError, match="was built for store"):
        WindowDataset(SignalStore(other), index)


def test_groups_are_reachable_for_leak_checking(store: SignalStore, index) -> None:
    splits = _kfold3(store)
    fold = folds_from_splits(SignalExamples(store, index), splits)[0]
    train = WindowDataset(store, index, fold.train)
    test = WindowDataset(store, index, fold.test)
    assert not (set(train.groups) & set(test.groups))


# --- loaders -------------------------------------------------------------------------


def test_loader_result_does_not_depend_on_worker_count(store: SignalStore, index) -> None:
    """A performance knob must never change an answer.

    Each worker gets its own RNG, so without an explicit generator and worker_init_fn the
    shuffle order varies with num_workers — putting the result at the mercy of how many
    cores the machine had.
    """
    dataset = WindowDataset(store, index)

    def order(workers: int) -> list[int]:
        loader = make_loader(dataset, batch_size=8, shuffle=True, num_workers=workers, seed=7)
        return [int(row) for batch in loader for row in batch["row"]]

    assert order(0) == order(2)


def test_the_seed_actually_changes_the_order(store: SignalStore, index) -> None:
    """Guards against a generator that is created and then never wired through."""
    dataset = WindowDataset(store, index)

    def order(seed: int) -> list[int]:
        loader = make_loader(dataset, batch_size=8, shuffle=True, seed=seed)
        return [int(row) for batch in loader for row in batch["row"]]

    assert order(1) != order(2)


def test_an_unshuffled_loader_covers_every_position_once(store: SignalStore, index) -> None:
    dataset = WindowDataset(store, index, positions=np.arange(20))
    loader = make_loader(dataset, batch_size=6)
    seen = [int(row) for batch in loader for row in batch["row"]]
    assert sorted(seen) == list(range(20))


def test_workers_read_the_store_independently(store: SignalStore, index) -> None:
    """The store drops live readers on pickling and rekeys by pid; this is that contract.

    Verified under real DataLoader concurrency in ADR 0005's worker-scaling benchmark
    before anything depended on it — this keeps it verified.
    """
    dataset = WindowDataset(store, index, positions=np.arange(16))
    loader = make_loader(dataset, batch_size=4, num_workers=2)
    batches = [batch["x"] for batch in loader]
    assert sum(batch.shape[0] for batch in batches) == 16
    assert all(torch.isfinite(batch).all() for batch in batches)
