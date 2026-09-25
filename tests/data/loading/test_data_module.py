"""The exact Lightning data path replays split identity without inventing another model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, IterableDataset, default_collate

from dsio.data.adapters import entity_examples
from dsio.data.loading import (
    DsioDataModule,
    LoadingError,
    build_loader,
    collate_items,
    stored_samples,
)
from dsio.data.splits import generate
from dsio.data.store import SignalStore


def _store(path: Path) -> SignalStore:
    with SignalStore.builder(path, channels=2, dtype="float32") as builder:
        for index in range(8):
            builder.add(
                f"sample-{index}",
                np.full((4, 2), index, dtype=np.float32),
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


def test_build_loader_drops_only_the_incomplete_final_batch() -> None:
    class Samples(Dataset[dict[str, Any]]):
        def __len__(self) -> int:
            return 5

        def __getitem__(self, position: int) -> dict[str, Any]:
            return {"sample_id": f"sample-{position}", "x": torch.tensor(position)}

    retained = build_loader(Samples(), batch_size=2)
    dropped = build_loader(Samples(), batch_size=2, drop_last=True)

    assert _ids(retained) == [f"sample-{index}" for index in range(5)]
    assert _ids(dropped) == [f"sample-{index}" for index in range(4)]
    assert retained.drop_last is False
    assert dropped.drop_last is True


def test_build_loader_requires_boolean_drop_last() -> None:
    class Samples(Dataset[dict[str, Any]]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, position: int) -> dict[str, Any]:
            return {"sample_id": str(position), "x": torch.tensor(position)}

    with pytest.raises(LoadingError, match="drop_last must be bool"):
        build_loader(Samples(), drop_last=1)  # type: ignore[arg-type]


def test_worker_loaders_do_not_fork_threaded_orchestrators(tmp_path: Path) -> None:
    store, examples, _ = _inputs(tmp_path / "samples")
    dataset = stored_samples(store, examples, examples.sample_ids.tolist())

    loader = build_loader(dataset, num_workers=1)

    assert loader.multiprocessing_context is not None
    assert loader.multiprocessing_context.get_start_method() == "spawn"


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


def test_data_module_defaults_to_retaining_every_phase_remainder(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")

    module = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"train": "train"},
        dataset_factory=stored_samples,
    )

    assert module.drop_last == {
        "train": False,
        "validate": False,
        "test": False,
        "predict": False,
    }


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


def test_collation_requires_batch_field_cardinality_to_match_identity() -> None:
    items = [
        {"sample_id": "left", "x": np.array([1.0])},
        {"sample_id": "right", "x": np.array([2.0])},
    ]

    with pytest.raises(LoadingError, match="1 rows for 2 sample_id"):
        collate_items(
            items,
            collate_fn=lambda _: {
                "sample_id": ["left", "right"],
                "x": torch.ones(1, 1),
            },
        )


@pytest.mark.parametrize("scalar", [torch.tensor(1.0), np.asarray(1.0)])
def test_collation_rejects_tensor_fields_without_a_batch_axis(scalar: object) -> None:
    items = [
        {"sample_id": "left", "x": np.array([1.0])},
        {"sample_id": "right", "x": np.array([2.0])},
    ]

    with pytest.raises(LoadingError, match="unbatched scalar"):
        collate_items(
            items,
            collate_fn=lambda _: {
                "sample_id": ["left", "right"],
                "x": scalar,
            },
        )


def test_collation_accepts_a_zero_width_batched_tensor() -> None:
    items = [
        {"sample_id": "left", "x": np.empty(0)},
        {"sample_id": "right", "x": np.empty(0)},
    ]

    assert collate_items(items)["x"].shape == (2, 0)


def test_collation_accepts_a_sample_major_ragged_tensor_batch() -> None:
    items = [
        {"sample_id": "left", "x": np.array([1.0])},
        {"sample_id": "right", "x": np.array([2.0, 3.0, 4.0])},
    ]

    batch = collate_items(
        items,
        collate_fn=lambda values: {
            "sample_id": [item["sample_id"] for item in values],
            "x": [torch.as_tensor(item["x"]) for item in values],
        },
    )

    assert [tensor.shape for tensor in batch["x"]] == [(1,), (3,)]


def test_collation_rejects_unbatched_opaque_and_device_hiding_fields() -> None:
    items = [
        {"sample_id": "left", "x": np.array([1.0])},
        {"sample_id": "right", "x": np.array([2.0])},
    ]

    with pytest.raises(LoadingError, match="unbatched scalar"):
        collate_items(
            items,
            collate_fn=lambda _: {
                "sample_id": ["left", "right"],
                "x": torch.ones(2, 1),
                "y": 1,
            },
        )
    with pytest.raises(LoadingError, match="1 rows for 2 sample_id|unbatched scalar"):
        collate_items(
            items,
            collate_fn=lambda _: {
                "sample_id": ["left", "right"],
                "x": [{"value": 1}],
            },
        )

    class HiddenTensor:
        def __init__(self) -> None:
            self.value = torch.ones(1, device="meta")

    with pytest.raises(LoadingError, match="unsupported type HiddenTensor"):
        collate_items(
            items,
            collate_fn=lambda _: {
                "sample_id": ["left", "right"],
                "x": torch.ones(2, 1),
                "hidden": HiddenTensor(),
            },
        )


def test_collation_rejects_object_arrays_and_accepts_zero_width_sequences() -> None:
    items = [
        {"sample_id": "left", "tokens": []},
        {"sample_id": "right", "tokens": []},
    ]
    hidden = np.empty(2, dtype=object)
    hidden[0] = torch.ones(1, device="meta")
    hidden[1] = torch.ones(1, device="meta")

    with pytest.raises(LoadingError, match="unsupported object array"):
        collate_items(
            items,
            collate_fn=lambda _: {
                "sample_id": ["left", "right"],
                "hidden": hidden,
            },
        )

    assert collate_items(items)["tokens"] == []


def test_collation_rejects_object_bearing_numpy_scalars() -> None:
    items = [
        {"sample_id": "left", "x": np.array([1.0])},
        {"sample_id": "right", "x": np.array([2.0])},
    ]
    hidden = np.empty(2, dtype=[("value", object)])
    hidden[0]["value"] = torch.ones(1, device="meta")
    hidden[1]["value"] = torch.ones(1, device="meta")

    with pytest.raises(LoadingError, match="unsupported object scalar"):
        collate_items(
            items,
            collate_fn=lambda _: {
                "sample_id": ["left", "right"],
                "hidden": [hidden[0], hidden[1]],
            },
        )


def test_collation_rejects_unordered_and_recursive_batch_containers() -> None:
    items = [{"sample_id": "one", "x": np.ones(2)}]

    with pytest.raises(LoadingError, match="unordered batch container"):
        collate_items(
            items,
            collate_fn=lambda _: {
                "sample_id": ["one"],
                "x": torch.ones(1, 2),
                "hidden": {torch.ones(1, device="meta")},
            },
        )

    cyclic: dict[str, Any] = {"sample_id": ["one"], "x": torch.ones(1, 2)}
    cyclic["self"] = cyclic
    with pytest.raises(LoadingError, match="recursive batch container"):
        collate_items(items, collate_fn=lambda _: cyclic)


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


def test_factory_must_return_a_map_style_dataset(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")

    class Stream(IterableDataset[Mapping[str, Any]]):
        def __iter__(self):
            yield from ()

        def __len__(self) -> int:
            return len(split.fold(0).assignments["train"])

    module = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"train": "train"},
        dataset_factory=lambda *_: Stream(),
    )

    with pytest.raises(LoadingError, match="map-style Dataset.*IterableDataset"):
        module.setup("fit")


def test_seeded_shuffle_order_is_independent_of_worker_count(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")

    def orders(workers: int) -> list[list[str]]:
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
        loader = module.train_dataloader()
        return [_ids(loader) for _ in range(3)]

    assert orders(0) == orders(2)


def test_train_drop_last_is_deterministic_and_preserves_complete_batch_identities(
    tmp_path: Path,
) -> None:
    store, examples, split = _inputs(tmp_path / "samples")

    def retained_ids() -> list[str]:
        module = DsioDataModule(
            store,
            examples,
            split,
            fold=0,
            roles={"train": "train"},
            dataset_factory=stored_samples,
            batch_size=4,
            seed=31,
            drop_last={"train": True},
        )
        module.setup("fit")
        loader = module.train_dataloader()
        assert loader.drop_last is True
        return _ids(loader)

    first = retained_ids()

    assert first == retained_ids()
    assert len(first) == 4
    assert len(set(first)) == 4
    assert set(first) < set(split.fold(0).assignments["train"])


@pytest.mark.parametrize("phase", ["validate", "test", "predict"])
def test_observation_phases_cannot_drop_assigned_samples(
    tmp_path: Path,
    phase: str,
) -> None:
    store, examples, split = _inputs(tmp_path / "samples")

    with pytest.raises(LoadingError, match=f"drop_last for phase {phase!r} must be false"):
        DsioDataModule(
            store,
            examples,
            split,
            fold=0,
            roles={"train": "train"},
            dataset_factory=stored_samples,
            drop_last={phase: True},
        )


def test_observation_phase_drop_last_is_revalidated_during_setup(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")
    module = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"test": "train"},
        dataset_factory=stored_samples,
        batch_size=4,
    )
    module.drop_last["test"] = True

    with pytest.raises(LoadingError, match="drop_last for phase 'test' must be false"):
        module.setup("test")


def test_drop_last_rejects_a_train_phase_without_one_full_batch(tmp_path: Path) -> None:
    store, examples, split = _inputs(tmp_path / "samples")
    module = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"train": "train"},
        dataset_factory=stored_samples,
        batch_size=7,
        drop_last={"train": True},
    )

    with pytest.raises(LoadingError, match="drop_last.*no full train batch.*6.*7"):
        module.setup("fit")


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
    def topology(path: Path, samples: list[tuple[str, float]]) -> SignalStore:
        with SignalStore.builder(path, channels=1, dtype="float32") as builder:
            for sample_id, value in samples:
                builder.add(
                    sample_id,
                    np.full((2, 1), value, dtype=np.float32),
                    group="same-group",
                )
        return SignalStore(path)

    store = topology(tmp_path / "source" / "samples", [("A", 0), ("B", 1)])
    other = topology(tmp_path / "other" / "samples", [("B", 0), ("A", 1)])
    examples = entity_examples(store)

    assert store.manifest().signal_sha256 == other.manifest().signal_sha256
    assert store.identity != other.identity
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
        ({"shuffle": ["train"]}, "phase-to-bool mapping"),
        ({"drop_last": {"train": "yes"}}, "drop_last"),
        ({"drop_last": {"fit": True}}, "unsupported Lightning phase"),
        ({"drop_last": ["train"]}, "phase-to-bool mapping"),
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

    wrong_collator = DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"train": "train"},
        dataset_factory=stored_samples,
        collate_fn="not-callable",  # type: ignore[arg-type]
    )
    with pytest.raises(LoadingError, match="collate_fn must be callable"):
        wrong_collator.setup("fit")


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
