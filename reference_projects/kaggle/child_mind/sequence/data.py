"""Stage raw CMI actigraphy once and derive complete participant-safe windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dsio.data.adapters import SignalExamples
from dsio.data.store import SignalStore
from dsio.data.views import WindowSpec, build_index
from reference_projects.kaggle.child_mind.data import (
    SENSOR_PARQUET_COLUMNS,
    TABULAR_MASK_SLICE,
    TABULAR_VALUE_SLICE,
    load_competition_data,
    pack_participant,
)

RAW_SENSOR_FEATURES = (
    "X",
    "Y",
    "Z",
    "enmo",
    "anglez",
    "non-wear_flag",
    "light",
    "battery_voltage",
)


def stage_sequence_store(
    data_dir: str | Path, path: str | Path, *, window_length: int
) -> SignalStore:
    """Write every labelled participant as raw rows or one explicit missing window."""
    if isinstance(window_length, bool) or not isinstance(window_length, int) or window_length < 1:
        raise ValueError("window_length must be a positive integer")
    loaded = load_competition_data(data_dir)
    destination = Path(path)
    with SignalStore.builder(
        destination,
        channels=len(RAW_SENSOR_FEATURES),
        dtype="float32",
        source=str(Path(data_dir).resolve()),
        attrs={"schema": "kaggle-child-mind-sequence-v1", "window_length": window_length},
    ) as builder:
        for row in loaded["train"]:
            participant_id = str(row["id"])
            series_path = row["series_path"]
            present = series_path is not None
            signal = (
                _read_partition(Path(series_path), participant_id)
                if present
                else np.zeros((window_length, len(RAW_SENSOR_FEATURES)), dtype=np.float32)
            )
            source_rows = len(signal)
            if source_rows < window_length:
                padding = np.full(
                    (window_length - source_rows, len(RAW_SENSOR_FEATURES)),
                    np.nan,
                    dtype=np.float32,
                )
                signal = np.concatenate((signal, padding), axis=0)
            packed = pack_participant(row, None)[0]
            tabular = np.concatenate((packed[TABULAR_VALUE_SLICE], packed[TABULAR_MASK_SLICE]))
            builder.add(
                participant_id,
                signal,
                group=participant_id,
                attrs={
                    "label": int(row["target"]),
                    "sensor_present": present,
                    "source_rows": source_rows if present else 0,
                    "tabular": tabular.tolist(),
                },
            )
    store = SignalStore(destination)
    store.verify()
    return store


def sequence_examples(store: SignalStore, *, window_length: int) -> SignalExamples:
    """Expose complete non-overlapping windows with one final overlap for every tail."""
    spec = WindowSpec(
        length=window_length,
        stride=window_length,
        drop_last_partial=False,
    )
    return SignalExamples(store, build_index(store, spec))


def _read_partition(path: Path, participant_id: str) -> np.ndarray[Any, np.dtype[np.float32]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("CMI sequence ingestion requires the DSIO 'data' extra") from error

    try:
        parquet = pq.ParquetFile(path)
    except Exception as error:
        raise ValueError(
            f"sensor partition {participant_id!r} cannot be opened: {error}"
        ) from error
    expected = [name for name, _ in SENSOR_PARQUET_COLUMNS]
    if parquet.schema_arrow.names != expected:
        raise ValueError(f"sensor partition {participant_id!r} columns must be {expected}")
    expected_types = {
        name: pa.type_for_alias(field_type) for name, field_type in SENSOR_PARQUET_COLUMNS
    }
    for field in parquet.schema_arrow:
        if field.type != expected_types[field.name]:
            raise ValueError(
                f"sensor partition {participant_id!r} field {field.name!r} has "
                f"Arrow type {field.type}, expected {expected_types[field.name]}"
            )
    table = parquet.read(columns=["step", *RAW_SENSOR_FEATURES])
    steps = np.asarray(table["step"].to_numpy(zero_copy_only=False), dtype=np.int64)
    if not len(steps):
        raise ValueError(f"sensor partition {participant_id!r} is empty")
    if table["step"].null_count or bool((np.diff(steps) <= 0).any()):
        raise ValueError(
            f"sensor partition {participant_id!r} step must be non-null, increasing, and unique"
        )
    signal = np.column_stack(
        [
            np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=np.float32)
            for name in RAW_SENSOR_FEATURES
        ]
    )
    if bool(np.isinf(signal).any()):
        raise ValueError(f"sensor partition {participant_id!r} contains infinity")
    return np.ascontiguousarray(signal, dtype=np.float32)


__all__ = ["RAW_SENSOR_FEATURES", "sequence_examples", "stage_sequence_store"]
