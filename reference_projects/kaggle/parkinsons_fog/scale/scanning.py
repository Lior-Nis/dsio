"""Bounded, strict iteration of Parkinson non-supervised source lanes."""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

NON_TASK_COLUMNS = (
    "Time",
    "AccV",
    "AccML",
    "AccAP",
    "Event",
    "Valid",
    "Task",
    "StartHesitation",
    "Turn",
    "Walking",
)
DAILY_COLUMNS = ("Time", "AccV", "AccML", "AccAP")


def scan_non_supervised(
    inventory: Mapping[str, Any], *, batch_rows: int = 65_536
) -> dict[str, Any]:
    """Read every non-supervised row while retaining at most ``batch_rows`` rows."""
    if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
        raise ValueError(f"batch_rows must be a positive integer, got {batch_rows!r}")
    hasher = hashlib.sha256()
    lanes: dict[str, dict[str, int]] = {}
    maximum = 0
    for lane in ("notype", "daily"):
        files = inventory.get(lane)
        if not isinstance(files, list):
            raise ValueError(f"inventory lane {lane!r} must be a list")
        if not files:
            raise ValueError(f"inventory lane {lane!r} must not be empty")
        identities = [str(item["recording_id"]) for item in files]
        duplicates = sorted(
            identity for identity in set(identities) if identities.count(identity) > 1
        )
        if duplicates:
            raise ValueError(f"inventory lane {lane!r} has duplicate identity {duplicates[0]!r}")
        rows = 0
        source_bytes = 0
        for item in files:
            identity = str(item["recording_id"])
            path = Path(str(item["path"]))
            if not path.is_file():
                raise ValueError(f"{lane} source {identity!r} is missing: {path}")
            actual_bytes = _validate_source_fingerprint(item, path, lane, identity, "before")
            _hash_bytes(hasher, lane.encode())
            _hash_bytes(hasher, identity.encode())
            if lane == "notype":
                observed, observed_max = _scan_csv(path, identity, batch_rows, hasher)
            else:
                observed, observed_max = _scan_parquet(path, identity, batch_rows, hasher)
            _validate_source_fingerprint(item, path, lane, identity, "after")
            rows += observed
            source_bytes += actual_bytes
            maximum = max(maximum, observed_max)
        lanes[lane] = {"files": len(files), "rows": rows, "source_bytes": source_bytes}
    return {
        "files": sum(value["files"] for value in lanes.values()),
        "rows": sum(value["rows"] for value in lanes.values()),
        "source_bytes": sum(value["source_bytes"] for value in lanes.values()),
        "max_batch_rows": maximum,
        "batch_rows": batch_rows,
        "checksum": hasher.hexdigest(),
        "lanes": lanes,
    }


def _scan_csv(
    path: Path, identity: str, batch_rows: int, hasher: Any
) -> tuple[int, int]:
    rows = 0
    maximum = 0
    batch: list[bytes] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != NON_TASK_COLUMNS:
            raise ValueError(
                f"non-task source {identity!r} columns must be {list(NON_TASK_COLUMNS)}"
            )
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"non-task source {identity!r} row {row_number} is ragged")
            time = _integer(row["Time"], identity, row_number, "Time")
            if time != rows:
                raise ValueError(f"non-task source {identity!r} Time must be consecutive from zero")
            for name in ("AccV", "AccML", "AccAP"):
                _number(row[name], identity, row_number, name)
            for name in ("Event", "StartHesitation", "Turn", "Walking"):
                _binary(row[name], identity, row_number, name)
            for name in ("Valid", "Task"):
                if row[name].lower() not in ("true", "false"):
                    raise ValueError(
                        f"non-task source {identity!r} row {row_number} has invalid {name}"
                    )
            encoded_row = b"".join(
                _length_prefixed(row[name].encode()) for name in NON_TASK_COLUMNS
            )
            batch.append(encoded_row)
            rows += 1
            if len(batch) == batch_rows:
                for value in batch:
                    hasher.update(value)
                maximum = max(maximum, len(batch))
                batch.clear()
    for value in batch:
        hasher.update(value)
    maximum = max(maximum, len(batch))
    if rows == 0:
        raise ValueError(f"non-task source {identity!r} is empty")
    return rows, maximum


def _scan_parquet(
    path: Path, identity: str, batch_rows: int, hasher: Any
) -> tuple[int, int]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Parkinson scale scanning requires the optional pyarrow data extra"
        ) from error

    try:
        parquet = pq.ParquetFile(path)
    except Exception as error:
        raise ValueError(f"daily source {identity!r} cannot be opened: {error}") from error
    schema = parquet.schema_arrow
    if tuple(schema.names) != DAILY_COLUMNS:
        raise ValueError(f"daily source {identity!r} columns must be {list(DAILY_COLUMNS)}")
    if schema.field("Time").type != pa.int64() or any(
        schema.field(name).type != pa.float64() for name in DAILY_COLUMNS[1:]
    ):
        raise ValueError(f"daily source {identity!r} has unsupported column types: {schema}")

    dtype = np.dtype(
        [("Time", "<i8"), ("AccV", "<f8"), ("AccML", "<f8"), ("AccAP", "<f8")]
    )
    rows = 0
    maximum = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=list(DAILY_COLUMNS)):
            if batch.num_rows == 0:
                continue
            if any(column.null_count for column in batch.columns):
                raise ValueError(f"daily source {identity!r} contains null values")
            packed = np.empty(batch.num_rows, dtype=dtype)
            for index, name in enumerate(DAILY_COLUMNS):
                values = batch.column(index).to_numpy(zero_copy_only=False)
                if name != "Time" and not bool(np.isfinite(values).all()):
                    raise ValueError(f"daily source {identity!r} contains non-finite {name}")
                packed[name] = values
            expected = np.arange(rows, rows + batch.num_rows, dtype=np.int64)
            if not np.array_equal(packed["Time"], expected):
                raise ValueError(f"daily source {identity!r} Time must be consecutive from zero")
            hasher.update(packed.tobytes())
            rows += batch.num_rows
            maximum = max(maximum, batch.num_rows)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"daily source {identity!r} cannot be read: {error}") from error
    if rows == 0:
        raise ValueError(f"daily source {identity!r} is empty")
    return rows, maximum


def _number(value: str, identity: str, row: int, field: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(
            f"non-task source {identity!r} row {row} has invalid {field}={value!r}"
        ) from error
    if not math.isfinite(result):
        raise ValueError(f"non-task source {identity!r} row {row} has invalid {field}={value!r}")
    return result


def _integer(value: str, identity: str, row: int, field: str) -> int:
    result = _number(value, identity, row, field)
    if not result.is_integer():
        raise ValueError(
            f"non-task source {identity!r} row {row} has non-integer {field}={value!r}"
        )
    return int(result)


def _binary(value: str, identity: str, row: int, field: str) -> None:
    if _integer(value, identity, row, field) not in (0, 1):
        raise ValueError(
            f"non-task source {identity!r} row {row} has non-binary {field}={value!r}"
        )


def _validate_source_fingerprint(
    item: Mapping[str, Any], path: Path, lane: str, identity: str, phase: str
) -> int:
    expected_bytes = item.get("bytes")
    expected_mtime_ns = item.get("mtime_ns")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or isinstance(expected_mtime_ns, bool)
        or not isinstance(expected_mtime_ns, int)
        or expected_mtime_ns < 0
    ):
        raise ValueError(f"{lane} source {identity!r} has an invalid stat fingerprint")
    try:
        stat = path.stat()
    except OSError as error:
        raise ValueError(f"{lane} source {identity!r} cannot be stat-ed: {path}") from error
    if stat.st_size != expected_bytes or stat.st_mtime_ns != expected_mtime_ns:
        raise ValueError(
            f"{lane} source {identity!r} changed since inventory {phase} scan: "
            f"expected ({expected_bytes}, {expected_mtime_ns}), observed "
            f"({stat.st_size}, {stat.st_mtime_ns})"
        )
    return stat.st_size


def _length_prefixed(value: bytes) -> bytes:
    return len(value).to_bytes(8, "little") + value


def _hash_bytes(hasher: Any, value: bytes) -> None:
    hasher.update(_length_prefixed(value))
