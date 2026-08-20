"""WindowDataset: fold positions over a memory-mapped store, without copying it."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from dsio.data.adapters import SignalExamples, entity_examples  # noqa: E402
from dsio.data.store import SignalStore  # noqa: E402
from dsio.data.views import WindowSpec, build_index  # noqa: E402
from dsio.nn.components import Jitter  # noqa: E402
from dsio.nn.data import (  # noqa: E402
    TwoViewCollate,
    WindowDataset,
    make_loader,
    train_dataset,
    val_dataset,
)
from dsio.nn.masking import CausalMask, SpanMask  # noqa: E402
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


# --- pretext masking -------------------------------------------------------------------


def test_a_pretext_item_carries_the_masked_signal_and_a_sentinel_target(
    store: SignalStore, index
) -> None:
    """The target is not the untouched window. It is the original value at every position
    the mask hid, and NaN everywhere it did not -- the continuous analogue of MLM's -100,
    so a mask-aware loss knows what to score without a batch dict or a subclass.

    normalize_target=False here: this test is about the sentinel's shape, not about
    normalisation, which gets its own test below.
    """
    mask = CausalMask(ratio=0.5)
    dataset = WindowDataset(
        store, index, positions=np.array([7]), mask=mask, normalize_target=False
    )
    item = dataset[0]
    original = torch.from_numpy(store.read(int(index.starts[7]), 100).T).float()
    hidden = mask(original.unsqueeze(0)).squeeze(0)

    assert not np.array_equal(item["x"].numpy(), original.numpy()), "the masked view must differ"
    assert torch.isnan(item["y"][:, ~hidden]).all(), "visible positions must be NaN in the target"
    assert not torch.isnan(item["y"][:, hidden]).any(), "hidden positions must carry a real value"
    assert torch.equal(item["y"][:, hidden], original[:, hidden]), (
        "hidden positions must carry the original value, not the masked one"
    )


def test_a_pretext_item_still_carries_its_row(store: SignalStore, index) -> None:
    """Row survives either branch: the fold loop aligns by identity, not loader order."""
    dataset = WindowDataset(
        store, index, positions=np.array([5, 9, 2]), mask=CausalMask(ratio=0.5)
    )
    assert [dataset[i]["row"] for i in range(3)] == [5, 9, 2]


def test_a_pretext_target_lands_under_the_configured_key(store: SignalStore, index) -> None:
    """The target key is already configurable on the module; the dataset just has to use
    whichever one the module was built with, so no module-side change is needed."""
    dataset = WindowDataset(
        store, index, positions=np.array([3]), mask=CausalMask(ratio=0.5), target_key="orig"
    )
    item = dataset[0]
    assert "orig" in item and "y" not in item


def test_normalize_target_prevents_a_loud_channel_from_dominating(tmp_path: Path) -> None:
    """The property the deleted MaskedReconstruction.step()'s norm_target guarded, now
    enforced on the dataset side: reconstructing raw amplitude makes a loss dominated by
    whichever channel has the largest units, so normalising the target here is what keeps
    a downstream loss from spending all its gradient on the loudest sensor.

    To watch this fail: hardcode normalize_target=True on both datasets below (or compare
    the *raw* dataset's own loud-vs-quiet ratio against the same threshold) and the loud
    channel's target swamps the quiet one by ~100x either way, exactly like the assertion
    on ``raw`` two lines down demonstrates for the un-normalised case.
    """
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((400, 2)).astype("float32")
    signal[:, 0] *= 100.0  # channel 0 is two orders of magnitude louder than channel 1
    path = tmp_path / "loud"
    with SignalStore.builder(path, channels=2) as builder:
        builder.add("p0", signal, group="p0")
    loud_store = SignalStore(path)
    loud_index = build_index(loud_store, WindowSpec(length=100, stride=50))

    normalized = WindowDataset(
        loud_store,
        loud_index,
        positions=np.array([0]),
        mask=CausalMask(ratio=0.5),
        normalize_target=True,
    )[0]
    raw = WindowDataset(
        loud_store,
        loud_index,
        positions=np.array([0]),
        mask=CausalMask(ratio=0.5),
        normalize_target=False,
    )[0]

    hidden = ~torch.isnan(raw["y"])
    raw_loud = raw["y"][0][hidden[0]].abs().mean()
    raw_quiet = raw["y"][1][hidden[1]].abs().mean()
    norm_loud = normalized["y"][0][hidden[0]].abs().mean()
    norm_quiet = normalized["y"][1][hidden[1]].abs().mean()

    assert raw_loud > raw_quiet * 20, "the fixture must actually be louder on channel 0"
    assert norm_loud < norm_quiet * 5, (
        "normalisation must bring the loud channel's target back down to the quiet "
        "channel's scale, not leave it dominating"
    )


def test_validation_dataset_is_unmasked_by_construction(store: SignalStore, index) -> None:
    """The property the model's old ``self.training`` guard used to enforce, now enforced
    by which dataset a stage is handed rather than by a flag checked on every call.

    A ``WindowDataset`` has no notion of "training" or "eval" at all — there is no runtime
    branch here to get wrong. A training dataset masks because it was built with ``mask=``;
    a validation dataset does not because it was not, and that is the whole mechanism.
    """
    positions = np.array([3, 8])
    masked = WindowDataset(store, index, positions=positions, mask=CausalMask(ratio=0.5))
    # This is what a runner hands validation: the same store and index, no mask at all.
    unmasked = WindowDataset(store, index, positions=positions)

    original = store.read(int(index.starts[3]), 100).T
    assert np.array_equal(unmasked[0]["x"].numpy(), original), (
        "validation must see the untouched window"
    )
    assert not np.array_equal(masked[0]["x"].numpy(), original), (
        "training must see the masked window"
    )


def test_train_dataset_builder_can_mask(store: SignalStore, index) -> None:
    dataset = train_dataset(store, index, positions=np.array([7]), mask=CausalMask(ratio=0.5))
    item = dataset[0]
    original = store.read(int(index.starts[7]), 100).T
    assert not np.array_equal(item["x"].numpy(), original)
    assert torch.isnan(item["y"]).any(), "a masked train dataset must emit a sentinel target"


# --- mask_seed: a masked item as a function of position, not of call order -------------


def test_an_unseeded_mask_differs_across_repeated_fetches(store: SignalStore, index) -> None:
    """The property mask_seed exists to fix, demonstrated first: without it, the same
    position draws a different mask every time it is fetched -- exactly the failure the
    deleted self.training guard existed to prevent, one layer down in the dataset."""
    mask = SpanMask(ratio=0.5, span=16)
    dataset = train_dataset(store, index, positions=np.array([7]), mask=mask)
    draws = [dataset[0]["x"].clone() for _ in range(5)]
    assert not all(torch.equal(draws[0], draw) for draw in draws)


def test_a_seeded_mask_is_identical_across_repeated_fetches(store: SignalStore, index) -> None:
    """mask_seed makes the same position draw the same mask regardless of call order --
    what a validation loader needs, so its val/loss measures the model rather than the
    draw."""
    mask = SpanMask(ratio=0.5, span=16)
    dataset = train_dataset(store, index, positions=np.array([7]), mask=mask, mask_seed=42)
    draws = [dataset[0]["x"].clone() for _ in range(5)]
    assert all(torch.equal(draws[0], draw) for draw in draws)


def test_a_seeded_mask_still_differs_between_positions(store: SignalStore, index) -> None:
    """mask_seed must not collapse every item onto the same mask -- it derives a distinct
    generator per position (mask_seed XOR position), not one shared generator for the
    whole dataset."""
    mask = SpanMask(ratio=0.5, span=16)
    dataset = train_dataset(
        store, index, positions=np.array([3, 7, 11]), mask=mask, mask_seed=42
    )
    items = [dataset[i]["x"] for i in range(3)]
    assert not torch.equal(items[0], items[1])
    assert not torch.equal(items[1], items[2])


def test_a_seeded_mask_still_varies_with_the_seed(store: SignalStore, index) -> None:
    """Not a fixed mask baked into the position -- a different mask_seed genuinely changes
    which mask a position draws, so different validation runs (or a run and a rerun with a
    different seed) are not silently pinned to one specific draw forever."""
    mask = SpanMask(ratio=0.5, span=16)
    a = train_dataset(store, index, positions=np.array([7]), mask=mask, mask_seed=1)[0]
    b = train_dataset(store, index, positions=np.array([7]), mask=mask, mask_seed=2)[0]
    assert not torch.equal(a["x"], b["x"])


def test_val_dataset_builder_has_no_mask_parameter() -> None:
    """The structural version of "never mask a validation batch": there is no keyword to
    pass here at all, so the mistake is unrepresentable rather than merely unmade."""
    assert "mask" not in inspect.signature(val_dataset).parameters


def test_val_dataset_builder_is_never_masked(store: SignalStore, index) -> None:
    dataset = val_dataset(store, index, positions=np.array([7]))
    item = dataset[0]
    original = store.read(int(index.starts[7]), 100).T
    assert np.array_equal(item["x"].numpy(), original)
    assert "y" not in item, "no labels were supplied, so there is nothing to key under y"


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


def _collect_by_row(loader) -> torch.Tensor:  # type: ignore[no-untyped-def]
    """x, reordered by row so worker-to-worker batch reshuffling can't hide a real
    difference (or manufacture a fake one) behind which worker happened to read which
    position."""
    xs = torch.cat([batch["x"] for batch in loader])
    rows = torch.cat([torch.as_tensor(batch["row"]) for batch in loader])
    return xs[torch.argsort(rows)]


def test_an_unseeded_mask_does_depend_on_worker_count(store: SignalStore, index) -> None:
    """The honest boundary shuffle-order seeding does not cover, demonstrated directly
    rather than left implicit: `_seed_worker` seeds Python's and NumPy's per-worker RNGs,
    but torch's own global RNG is seeded to `base_seed + worker_id` — different per worker
    by construction — so a mask strategy drawing from it (no generator passed) genuinely
    varies with num_workers. This is exactly why `mask_seed` exists below."""
    positions = np.arange(16)
    mask = SpanMask(0.5, span=8)

    def read(workers: int) -> torch.Tensor:
        dataset = train_dataset(store, index, positions, mask=mask)
        loader = make_loader(dataset, batch_size=4, num_workers=workers, seed=7)
        return _collect_by_row(loader)

    assert not torch.equal(read(0), read(2))


def test_a_seeded_mask_does_not_depend_on_worker_count(store: SignalStore, index) -> None:
    """mask_seed closes the gap the test above demonstrates: a generator built fresh per
    item from an explicit seed does not care which worker constructed it."""
    positions = np.arange(16)
    mask = SpanMask(0.5, span=8)

    def read(workers: int) -> torch.Tensor:
        dataset = train_dataset(store, index, positions, mask=mask, mask_seed=42)
        loader = make_loader(dataset, batch_size=4, num_workers=workers, seed=7)
        return _collect_by_row(loader)

    assert torch.equal(read(0), read(2))


def test_an_unseeded_two_view_collate_does_depend_on_worker_count(
    store: SignalStore, index
) -> None:
    """The collate-side twin of the mask test above: PyTorch runs collate_fn inside the
    worker process, so an unseeded augmentor is exactly as exposed to per-worker torch RNG
    seeding as an unseeded mask strategy."""
    positions = np.arange(16)

    def read(workers: int) -> torch.Tensor:
        loader = make_loader(
            val_dataset(store, index, positions),
            batch_size=4,
            num_workers=workers,
            seed=7,
            collate_fn=TwoViewCollate(Jitter(0.3)),
        )
        return _collect_by_row(loader)

    assert not torch.equal(read(0), read(2))


def test_a_seeded_two_view_collate_does_not_depend_on_worker_count(
    store: SignalStore, index
) -> None:
    positions = np.arange(16)

    def read(workers: int) -> torch.Tensor:
        loader = make_loader(
            val_dataset(store, index, positions),
            batch_size=4,
            num_workers=workers,
            seed=7,
            collate_fn=TwoViewCollate(Jitter(0.3), seed=42),
        )
        return _collect_by_row(loader)

    assert torch.equal(read(0), read(2))


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


# --- TwoViewCollate --------------------------------------------------------------------
#
# Moved here from the deleted tests/ssl/test_methods.py: SimCLR and VICReg no longer build
# their two views inside a training step (dsio.ssl.methods.SimCLR/VICReg.step, both
# deleted). Task 6b moved that to collate time instead, so what those tests used to check
# about the pair-index target is now a property of TwoViewCollate, not of a pretext
# objective, and lives here next to the rest of the loader machinery.

class _Tag(nn.Module):
    """A deterministic stand-in for a stochastic augmentor: each call adds a distinct
    integer offset, so two independent calls are checkable rather than merely "different"."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x + self.calls


def _items(n: int, channels: int = 2, length: int = 8) -> list[dict[str, object]]:
    torch.manual_seed(0)
    return [{"x": torch.randn(channels, length), "row": i} for i in range(n)]


def test_two_view_collate_stacks_two_views_into_the_batch_dimension() -> None:
    items = _items(5)
    batch = TwoViewCollate(nn.Identity())(items)
    assert batch["x"].shape == (10, 2, 8)


def test_two_view_collate_calls_augment_twice_independently() -> None:
    """Each view is a separate, independent augmentation -- not the same call reused."""
    items = _items(3)
    tag = _Tag()
    batch = TwoViewCollate(tag)(items)
    assert tag.calls == 2
    first_view, second_view = batch["x"][:3], batch["x"][3:]
    raw = torch.stack([item["x"] for item in items])
    assert torch.equal(first_view, raw + 1)
    assert torch.equal(second_view, raw + 2)


def test_two_view_collate_pair_index_names_each_rows_partner() -> None:
    """window i's two views live at i and i + batch; target[i] must name the other one."""
    items = _items(4)
    batch = TwoViewCollate(nn.Identity())(items)
    target = batch["y"]
    assert torch.equal(target, torch.tensor([4, 5, 6, 7, 0, 1, 2, 3]))


def test_two_view_collate_pair_index_is_involutive() -> None:
    """The relation the recovered-halves loss (VICReg) depends on: target[target[i]] == i,
    regardless of batch size."""
    items = _items(7)
    batch = TwoViewCollate(nn.Identity())(items)
    target = batch["y"]
    assert torch.equal(target[target], torch.arange(target.shape[0]))


def test_two_view_collate_duplicates_the_row_for_each_view() -> None:
    items = _items(3)
    batch = TwoViewCollate(nn.Identity())(items)
    assert torch.equal(batch["row"], torch.tensor([0, 1, 2, 0, 1, 2]))


def test_two_view_collate_writes_under_the_configured_target_key() -> None:
    items = _items(2)
    batch = TwoViewCollate(nn.Identity(), target_key="pair")(items)
    assert "pair" in batch and "y" not in batch


def test_two_view_collate_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        TwoViewCollate(nn.Identity())([])


# --- TwoViewCollate.seed: two views as a function of the batch's rows, not call order --


class _RandomTag(nn.Module):
    """An augmentor that actually draws from torch's global RNG (unlike ``_Tag`` above,
    whose per-instance counter is deterministic by construction and so cannot tell a
    seeded fork_rng apart from an unseeded one)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn(())


def test_an_unseeded_collate_differs_across_repeated_calls() -> None:
    items = _items(4)
    collate = TwoViewCollate(_RandomTag())
    draws = [collate(items)["x"].clone() for _ in range(5)]
    assert not all(torch.equal(draws[0], draw) for draw in draws)


def test_a_seeded_collate_is_identical_across_repeated_calls() -> None:
    items = _items(4)
    collate = TwoViewCollate(_RandomTag(), seed=42)
    draws = [collate(items)["x"].clone() for _ in range(5)]
    assert all(torch.equal(draws[0], draw) for draw in draws)


def test_a_seeded_collate_still_gives_the_two_views_different_noise() -> None:
    """Determinism must not collapse the two views onto each other -- fork_rng reseeds
    once per batch and augment is still called twice against the continuing stream inside
    it, so the two calls draw different values."""
    items = _items(4)
    batch = TwoViewCollate(_RandomTag(), seed=42)(items)
    first_view, second_view = batch["x"][:4], batch["x"][4:]
    assert not torch.equal(first_view, second_view)


def test_a_seeded_collate_leaves_the_outer_rng_stream_untouched() -> None:
    """fork_rng's whole point: whatever a caller draws from torch's global RNG right after
    a seeded collate call must be exactly what it would have drawn had the collate call
    not happened at all."""
    items = _items(4)  # built once, outside either seeded region below: _items itself
    # reseeds the global RNG as a side effect, which would contaminate the comparison if
    # it ran between the two `torch.manual_seed(123)` calls instead of before both.

    torch.manual_seed(123)
    expected = torch.randn(4)

    torch.manual_seed(123)
    TwoViewCollate(_RandomTag(), seed=999)(items)
    after = torch.randn(4)

    assert torch.equal(after, expected)


def test_a_seeded_collate_varies_with_the_seed() -> None:
    items = _items(4)
    a = TwoViewCollate(_RandomTag(), seed=1)(items)["x"]
    b = TwoViewCollate(_RandomTag(), seed=2)(items)["x"]
    assert not torch.equal(a, b)
