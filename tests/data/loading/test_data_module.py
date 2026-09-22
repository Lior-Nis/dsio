"""The exact Lightning data path replays split identity without inventing another model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, default_collate

from dsio.data.adapters import entity_examples
from dsio.data.loading import DsioDataModule, LoadingError, collate_items, stored_samples
from dsio.data.splits import generate
from dsio.data.store import SignalStore


def _store(path: Path, *, offset: int = 0) -> SignalStore:
    with SignalStore.builder(path, channels=2, dtype="float32") as builder:
        for index in range(8):
            builder.add(
                f"sample-{index}",
                np.full((4, 2), index + offset, dtype=np.float32),
                group=f"group-{index // 2}",
                attrs={"target": index % 2},
            )
    return SignalStore(path)


def _inputs(path: Path) -> tuple[SignalStore, Any, Any]:
    store = _store(path)
    examples = entity_examples(store)
    split = generate(
        examples,
        "group_shuffle",
        name="holdout",
        seed=7,
        parameters={"test_size": 0.25},
    )
    return store, examples, split


def _ids(loader: Any) -> list[str]:
    return [sample_id for batch in loader for sample_id in batch["sample_id"]]


def test_setup_maps_exact_split_roles_to_lightning_phases(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")
    module = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"train": "train", "validate": "test"},
        dataset_factory=stored_samples,
        batch_size=2,
        seed=19,
    )

    module.setup("fit")

    fold = split.fold(0)
    assert type(module) is DsioDataModule
    assert sorted(_ids(module.train_dataloader())) == sorted(fold.assignments["train"])
    assert _ids(module.val_dataloader()) == fold.assignments["test"]
    batch = next(iter(module.train_dataloader()))
    assert batch["sample_id"] == [str(value) for value in batch["sample_id"]]
    assert batch["x"].shape == (2, 4, 2)


def test_setup_constructs_only_the_requested_stage(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")
    calls: list[tuple[str, ...]] = []

    def recording_factory(
        source: SignalStore,
        source_examples: Any,
        sample_ids: Sequence[str],
    ) -> Dataset[Mapping[str, Any]]:
        calls.append(tuple(sample_ids))
        return stored_samples(source, source_examples, sample_ids)

    module = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"train": "train", "test": "test"},
        dataset_factory=recording_factory,
    )

    module.setup("test")

    assert calls == [tuple(split.fold(0).assignments["test"])]
    assert _ids(module.test_dataloader()) == split.fold(0).assignments["test"]
    with pytest.raises(LoadingError, match="train.*not been set up"):
        module.train_dataloader()


def test_default_collation_preserves_identity_next_to_payload_and_target() -> None:
    items = [
        {"sample_id": "left", "x": np.array([1.0]), "y": 0},
        {"sample_id": "right", "x": np.array([2.0]), "y": 1},
    ]

    batch = collate_items(items)

    assert batch["sample_id"] == ["left", "right"]
    assert batch["x"].shape == (2, 1)
    assert batch["y"].tolist() == [0, 1]


def test_data_module_applies_custom_cpu_collation_behind_the_identity_guard(
    tmp_path: Path,
) -> None:
    store, examples, split = _inputs(tmp_path / "samples")

    def add_target(items: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        batch = default_collate(items)
        batch["y"] = batch["x"].mean(dim=(1, 2))
        return batch

    module = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"validate": "test"},
        dataset_factory=stored_samples,
        batch_size=2,
        collate_fn=add_target,
    )
    module.setup("validate")

    batch = next(iter(module.val_dataloader()))
    assert batch["sample_id"] == split.fold(0).assignments["test"]
    assert batch["y"].shape == (2,)


def test_custom_collation_must_preserve_exact_identity_order() -> None:
    items = [
        {"sample_id": "left", "x": np.array([1.0])},
        {"sample_id": "right", "x": np.array([2.0])},
    ]

    with pytest.raises(LoadingError, match="changed sample_id"):
        collate_items(
            items,
            collate_fn=lambda _: {"sample_id": ["right", "left"], "x": np.ones((2, 1))},
        )


@pytest.mark.parametrize(
    "items, collate_fn, message",
    [
        ([], None, "empty batch"),
        (
            [{"sample_id": "one", "x": np.ones(2)}],
            lambda _: np.ones(2),
            "must return a mapping",
        ),
        (
            [{"sample_id": "one", "x": np.ones(2)}],
            lambda _: {"x": np.ones((1, 2))},
            "must contain sample_id",
        ),
    ],
)
def test_malformed_batches_fail_at_collation(
    items: list[Mapping[str, Any]],
    collate_fn: Any,
    message: str,
) -> None:
    with pytest.raises(LoadingError, match=message):
        collate_items(items, collate_fn=collate_fn)


def test_collation_rejects_accelerator_side_processing() -> None:
    items = [{"sample_id": "one", "x": np.ones(2)}]

    with pytest.raises(LoadingError, match="keep tensors on CPU"):
        collate_items(
            items,
            collate_fn=lambda _: {
                "sample_id": ["one"],
                "x": torch.ones(1, 2, device="meta"),
            },
        )


@pytest.mark.parametrize(
    "item",
    [
        np.ones(2),
        {"x": np.ones(2)},
        {"sample_id": 7, "x": np.ones(2)},
        {"sample_id": "wrong", "x": np.ones(2)},
    ],
)
def test_factory_items_are_checked_at_the_identity_boundary(
    tmp_path: Path,
    item: object,
) -> None:
    store, examples, split = _inputs(tmp_path / "samples")

    class BrokenDataset(Dataset[Any]):
        def __len__(self) -> int:
            return len(split.fold(0).assignments["train"])

        def __getitem__(self, position: int) -> object:
            return item

    module = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"train": "train"},
        dataset_factory=lambda *_: BrokenDataset(),
    )
    module.setup("fit")

    with pytest.raises(LoadingError, match="mapping|sample_id"):
        next(iter(module.train_dataloader()))


def test_factory_length_must_match_the_role_assignment(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")

    class ShortDataset(Dataset[dict[str, object]]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, position: int) -> dict[str, object]:
            return {"sample_id": "sample-0", "x": np.ones(2)}

    module = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"train": "train"},
        dataset_factory=lambda *_: ShortDataset(),
    )

    with pytest.raises(LoadingError, match="returned 1 items.*assignment has"):
        module.setup("fit")


def test_seeded_shuffle_order_is_independent_of_worker_count(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")

    def order(workers: int) -> list[str]:
        module = DsioDataModule(
            store,
            examples,
            split,
            fold=0,
            roles={"train": "train"},
            dataset_factory=stored_samples,
            batch_size=2,
            num_workers=workers,
            seed=31,
        )
        module.setup("fit")
        return _ids(module.train_dataloader())

    assert order(0) == order(2)


def _worker_unsafe_factory(
    store: SignalStore,
    examples: Any,
    sample_ids: Sequence[str],
) -> Dataset[Mapping[str, Any]]:
    class LocalDataset(Dataset[Mapping[str, Any]]):
        def __len__(self) -> int:
            return len(sample_ids)

        def __getitem__(self, position: int) -> Mapping[str, Any]:
            return {"sample_id": sample_ids[position], "x": np.ones(2)}

    return LocalDataset()


def test_worker_unsafe_dataset_fails_during_setup(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")
    module = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"train": "train"},
        dataset_factory=_worker_unsafe_factory,
        num_workers=2,
    )

    with pytest.raises(LoadingError, match="cannot be used with DataLoader workers.*pickl"):
        module.setup("fit")


def test_worker_unsafe_collator_fails_during_setup(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")
    module = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"train": "train"},
        dataset_factory=stored_samples,
        collate_fn=lambda items: default_collate(items),
        num_workers=2,
    )

    with pytest.raises(LoadingError, match="collator cannot be used.*not picklable"):
        module.setup("fit")


def test_canonical_factory_rejects_a_different_store_with_the_same_sample_ids(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "source" / "samples")
    other = _store(tmp_path / "other" / "samples", offset=100)
    examples = entity_examples(store)

    with pytest.raises(LoadingError, match="content identity"):
        stored_samples(other, examples, examples.sample_ids.tolist())


def test_missing_role_and_invalid_stage_fail_before_factory_use(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")
    module = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"predict": "unassigned"},
        dataset_factory=stored_samples,
    )

    with pytest.raises(LoadingError, match="role 'unassigned'.*fold 0"):
        module.setup("predict")
    with pytest.raises(LoadingError, match="unsupported Lightning setup stage"):
        module.setup("serve")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"roles": {}}, "phase-to-role mapping"),
        ({"roles": {"fit": "train"}}, "unsupported Lightning phase"),
        ({"batch_size": 0}, "batch_size"),
        ({"num_workers": -1}, "num_workers"),
        ({"seed": -1}, "seed"),
        ({"shuffle": {"train": "yes"}}, "shuffle"),
        ({"shuffle": {"fit": True}}, "unsupported Lightning phase"),
    ],
)
def test_invalid_loader_configuration_fails_at_construction(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    store, examples, split = _inputs(tmp_path / "samples")
    arguments: dict[str, object] = {
        "fold": 0,
        "roles": {"train": "train"},
        "dataset_factory": stored_samples,
    }
    arguments.update(kwargs)

    with pytest.raises(LoadingError, match=message):
        DsioDataModule(store, examples, split, **arguments)  # type: ignore[arg-type]


def test_invalid_fold_and_factory_result_fail_during_setup(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")
    missing_fold = DsioDataModule(
        store,
        examples,
        split,
        fold=99,
        roles={"train": "train"},
        dataset_factory=stored_samples,
    )
    with pytest.raises(ValueError, match="no fold 99"):
        missing_fold.setup("fit")

    wrong_result = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"train": "train"},
        dataset_factory=lambda *_: [],  # type: ignore[arg-type,return-value]
    )
    with pytest.raises(LoadingError, match="must return a torch Dataset"):
        wrong_result.setup("fit")


def test_split_is_validated_before_the_dataset_factory_runs(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")
    mismatched = entity_examples(_store(tmp_path / "other"))
    called = False

    def factory(*_: object) -> Dataset[Mapping[str, Any]]:
        nonlocal called
        called = True
        raise AssertionError("must not run")

    module = DsioDataModule(
        store,
        mismatched,
        split,
        fold=0,
        roles={"train": "train"},
        dataset_factory=factory,
    )

    with pytest.raises(ValueError, match="dataset digest|built for dataset"):
        module.setup("fit")
    assert called is False
