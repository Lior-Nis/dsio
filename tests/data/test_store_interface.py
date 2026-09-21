from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from dsio.contracts import sha256_of_file
from dsio.data.store import ENTITIES_FILE, MANIFEST_FILE, SIGNAL_FILE, SignalStore, StoreError


def _build_store(path: Path, *, rows: int = 32) -> SignalStore:
    with SignalStore.builder(path, channels=2, dtype="float32", attrs={"kind": "test"}) as builder:
        builder.add(
            "alpha",
            np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
            group="subject-a",
            attrs={"label": 1},
        )
        builder.add(
            "beta",
            np.arange(12, dtype=np.float32).reshape(6, 2) + 100,
            group="subject-b",
            attrs={"label": 0},
        )
    return SignalStore(path)


class _StoredSamples(Dataset[dict[str, Any]]):
    def __init__(self, store: SignalStore) -> None:
        self.store = store

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, position: int) -> dict[str, Any]:
        return self.store.read_sample(position)


def test_samples_round_trip_by_identity_and_position(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "samples")

    by_id = store.read_sample("beta")
    by_position = store.read_sample(1)

    assert by_id.keys() == {"sample_id", "data", "group", "attrs"}
    assert by_id["sample_id"] == "beta"
    assert by_id["group"] == "subject-b"
    assert by_id["attrs"] == {"label": 0}
    assert by_id["data"].shape == (6, 2)
    assert by_id["data"].dtype == np.dtype("float32")
    np.testing.assert_array_equal(by_position["data"], by_id["data"])
    assert store.sample_ids == ("alpha", "beta")
    assert store.manifest().schema_version == 1
    assert all(entity.data_sha256 for entity in store.entities)


@pytest.mark.parametrize("key", [-1, 2, True, 1.5, "missing"])
def test_invalid_sample_keys_fail_at_the_public_boundary(
    tmp_path: Path,
    key: object,
) -> None:
    store = _build_store(tmp_path / "samples")

    with pytest.raises(StoreError, match="sample"):
        store.read_sample(key)  # type: ignore[arg-type]


def test_read_entity_uses_the_same_content_checked_path(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "samples")
    payload = store.path / SIGNAL_FILE
    with payload.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 0xFF]))

    with pytest.raises(StoreError, match="alpha.*digest"):
        store.read_sample("alpha")
    with pytest.raises(StoreError, match="alpha.*digest"):
        store.read_entity("alpha")


def test_open_rejects_a_truncated_payload(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "samples")
    payload = store.path / SIGNAL_FILE
    payload.write_bytes(payload.read_bytes()[:-4])

    with pytest.raises(StoreError, match="signal.bin.*bytes"):
        SignalStore(store.path)


def test_open_rejects_an_unsupported_store_schema(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "samples")
    manifest_path = store.path / MANIFEST_FILE
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["schema_version"] = 999
    manifest_path.write_text(yaml.safe_dump(manifest))

    with pytest.raises(StoreError, match="schema version 999"):
        SignalStore(store.path)


def test_open_rejects_malformed_entity_metadata(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "samples")
    (store.path / ENTITIES_FILE).write_text("not-json\n")

    with pytest.raises(StoreError, match="entities.jsonl"):
        SignalStore(store.path)


def test_open_rejects_entity_topology_that_disagrees_with_the_index(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "samples")
    entities_path = store.path / ENTITIES_FILE
    entities = [json.loads(line) for line in entities_path.read_text().splitlines()]
    entities[1]["start_row"] += 1
    entities_path.write_text("".join(json.dumps(entity) + "\n" for entity in entities))
    manifest_path = store.path / MANIFEST_FILE
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["entities_sha256"] = sha256_of_file(str(entities_path))
    manifest_path.write_text(yaml.safe_dump(manifest))

    with pytest.raises(StoreError, match="beta.*start_row"):
        SignalStore(store.path)


def test_spawned_loader_workers_reopen_without_serializing_payload(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "samples", rows=200_000)
    assert len(pickle.dumps(store)) < 64 * 1024

    loader = DataLoader(
        _StoredSamples(store),
        batch_size=None,
        num_workers=2,
        multiprocessing_context="spawn",
    )
    samples = list(loader)

    assert [sample["sample_id"] for sample in samples] == ["alpha", "beta"]
    assert isinstance(samples[0]["data"], torch.Tensor)
    np.testing.assert_array_equal(
        samples[0]["data"].numpy(),
        store.read_sample("alpha")["data"],
    )
