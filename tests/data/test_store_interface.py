from __future__ import annotations

import json
import pickle
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from dsio.contracts import sha256_of_file
from dsio.data.format import HEADER_SIZE
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


def _update_manifest_digest(store: SignalStore, filename: str) -> None:
    manifest_path = store.path / MANIFEST_FILE
    manifest = yaml.safe_load(manifest_path.read_text())
    field = {"signal.idx": "index_sha256", ENTITIES_FILE: "entities_sha256"}[filename]
    manifest[field] = sha256_of_file(str(store.path / filename))
    manifest_path.write_text(yaml.safe_dump(manifest))


def _replace_offsets(store: SignalStore, offsets: list[int]) -> None:
    path = store.path / "signal.idx"
    raw = bytearray(path.read_bytes())
    raw[HEADER_SIZE:] = np.asarray(offsets, dtype=np.int64).tobytes()
    path.write_bytes(raw)
    _update_manifest_digest(store, "signal.idx")


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


@pytest.mark.parametrize("version", [True, "1", 1.0])
def test_store_schema_version_is_not_coerced(tmp_path: Path, version: object) -> None:
    store = _build_store(tmp_path / "samples")
    manifest_path = store.path / MANIFEST_FILE
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["schema_version"] = version
    manifest_path.write_text(yaml.safe_dump(manifest))

    with pytest.raises(StoreError, match="schema_version"):
        SignalStore(store.path)


def test_legacy_store_schema_is_rejected_with_a_rebuild_instruction(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "samples")
    manifest_path = store.path / MANIFEST_FILE
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest.pop("schema_version")
    manifest_path.write_text(yaml.safe_dump(manifest))
    entities_path = store.path / ENTITIES_FILE
    entities = [json.loads(line) for line in entities_path.read_text().splitlines()]
    for entity in entities:
        entity.pop("data_sha256")
    entities_path.write_text("".join(json.dumps(entity) + "\n" for entity in entities))

    with pytest.raises(StoreError, match="predates.*rebuild"):
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


@pytest.mark.parametrize(
    "offsets",
    [
        [1, 32, 38],
        [0, 32, 37],
        [0, 0, 38],
        [0, 39, 38],
    ],
)
def test_open_rejects_noncanonical_index_topology(
    tmp_path: Path,
    offsets: list[int],
) -> None:
    store = _build_store(tmp_path / "samples")
    _replace_offsets(store, offsets)

    with pytest.raises(StoreError, match="signal.idx.*offset"):
        SignalStore(store.path)


@pytest.mark.parametrize("channels", [0, -1, True, 1.5])
def test_builder_rejects_invalid_channel_counts_before_touching_disk(
    tmp_path: Path,
    channels: object,
) -> None:
    path = tmp_path / "bad"
    with pytest.raises(StoreError, match="channels.*positive integer"):
        SignalStore.builder(path, channels=channels)  # type: ignore[arg-type]
    assert not path.exists()


@pytest.mark.parametrize("dtype", ["complex64", "uint16", "bool"])
def test_builder_rejects_unsupported_dtypes_before_touching_disk(
    tmp_path: Path,
    dtype: str,
) -> None:
    path = tmp_path / "bad"
    with pytest.raises(StoreError, match="dtype.*supported"):
        SignalStore.builder(path, channels=1, dtype=dtype)
    assert not path.exists()


def test_open_rejects_a_nonpositive_channel_header(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "samples")
    index_path = store.path / "signal.idx"
    raw = bytearray(index_path.read_bytes())
    raw[16:20] = struct.pack("<I", 0)
    index_path.write_bytes(raw)
    _update_manifest_digest(store, "signal.idx")

    with pytest.raises(StoreError, match="signal.idx.*channels"):
        SignalStore(store.path)


def test_empty_builder_closes_its_payload_on_failure(tmp_path: Path) -> None:
    builder = SignalStore.builder(tmp_path / "empty", channels=1)

    with pytest.raises(StoreError, match="no entities"):
        builder.close()

    assert builder._signal.closed


@pytest.mark.parametrize(
    "attrs",
    [
        {"bad": float("nan")},
        {"bad": (1, 2)},
        {1: "not a JSON object key"},
        {"bad": np.array([1])},
    ],
)
def test_builder_rejects_non_json_attrs_before_writing_sample(
    tmp_path: Path,
    attrs: dict[Any, Any],
) -> None:
    builder = SignalStore.builder(tmp_path / "samples", channels=1)

    with pytest.raises(StoreError, match="attrs.*JSON"):
        builder.add("bad", np.ones((2, 1), dtype=np.float32), group="g", attrs=attrs)

    assert (builder.path / SIGNAL_FILE).stat().st_size == 0
    builder.add("good", np.ones((2, 1), dtype=np.float32), group="g")
    builder.close()


def test_builder_rejects_non_json_store_attrs_before_touching_disk(tmp_path: Path) -> None:
    path = tmp_path / "bad"
    with pytest.raises(StoreError, match="store attrs.*JSON"):
        SignalStore.builder(path, channels=1, attrs={"bad": float("inf")})
    assert not path.exists()


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
