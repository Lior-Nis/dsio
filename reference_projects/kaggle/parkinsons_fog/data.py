"""Bounded, strict CSV ingestion for Parkinson's Freezing of Gait."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

WINDOW_SIZE = 512
FEATURES = ("AccV", "AccML", "AccAP")
TARGETS = ("StartHesitation", "Turn", "Walking")
BASE_COLUMNS = ("Time", *FEATURES)
TRAIN_COLUMNS = (*BASE_COLUMNS, *TARGETS)
DEFOG_COLUMNS = (*TRAIN_COLUMNS, "Valid", "Task")
METADATA_COLUMNS = {
    "defog": ("Id", "Subject", "Visit", "Medication"),
    "tdcsfog": ("Id", "Subject", "Visit", "Test", "Medication"),
}


def load_competition_data(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    metadata = {
        kind: _metadata(root / f"{kind}_metadata.csv", columns)
        for kind, columns in METADATA_COLUMNS.items()
    }
    train = _recordings(root / "train", metadata, labelled=True)
    test = _recordings(root / "test", metadata, labelled=False)
    test_subjects = sorted({str(recording["subject"]) for recording in test})
    excluded = sorted(
        str(recording["recording_id"])
        for recording in train
        if recording["subject"] in set(test_subjects)
    )
    test_counts = {
        str(recording["recording_id"]): _row_count(Path(recording["path"])) for recording in test
    }
    submission_ids = _submission(root / "sample_submission.csv", test_counts)
    return {
        "train": train,
        "test": test,
        "test_subjects": test_subjects,
        "excluded_train_ids": excluded,
        "submission_ids": submission_ids,
    }


def iter_recording_windows(
    recording: Mapping[str, Any], *, window_size: int = WINDOW_SIZE
) -> Iterator[dict[str, Any]]:
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
        raise ValueError(f"window_size must be a positive integer, got {window_size!r}")
    path = Path(str(recording["path"]))
    labelled = bool(recording["labelled"])
    kind = str(recording["kind"])
    expected = DEFOG_COLUMNS if labelled and kind == "defog" else TRAIN_COLUMNS
    if not labelled:
        expected = BASE_COLUMNS
    buffer: list[list[float]] = []
    times: list[int] = []
    rows = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path.name} columns must be {list(expected)}")
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name} row {row_number} is ragged")
            time = _integer(row["Time"], path.name, row_number, "Time")
            if time != rows:
                raise ValueError(f"{path.name} Time must be consecutive from zero")
            features = [_number(row[name], path.name, row_number, name) for name in FEATURES]
            targets = (
                [_binary(row[name], path.name, row_number, name) for name in TARGETS]
                if labelled
                else [0.0] * len(TARGETS)
            )
            valid = 1.0
            if labelled and kind == "defog":
                valid = float(
                    _boolean(row["Valid"], path.name, row_number, "Valid")
                    and _boolean(row["Task"], path.name, row_number, "Task")
                )
            buffer.append([*features, *targets, valid])
            times.append(time)
            rows += 1
            if len(buffer) == window_size:
                yield {"start": times[0], "times": times, "data": np.asarray(buffer, np.float32)}
                buffer, times = [], []
    if buffer:
        yield {"start": times[0], "times": times, "data": np.asarray(buffer, np.float32)}
    if rows == 0:
        raise ValueError(f"{path.name} must contain at least one row")


def _recordings(
    directory: Path,
    metadata: Mapping[str, Mapping[str, str]],
    *,
    labelled: bool,
) -> list[dict[str, Any]]:
    if not directory.is_dir():
        raise ValueError(f"missing Parkinson's directory: {directory}")
    result: list[dict[str, Any]] = []
    for kind in METADATA_COLUMNS:
        for path in sorted((directory / kind).glob("*.csv")):
            recording_id = path.stem
            try:
                subject = metadata[kind][recording_id]
            except KeyError:
                raise ValueError(f"{path.name} has no {kind} metadata") from None
            result.append(
                {
                    "path": str(path),
                    "kind": kind,
                    "recording_id": recording_id,
                    "subject": subject,
                    "labelled": labelled,
                }
            )
    if not result:
        raise ValueError(f"{directory} contains no supported recordings")
    return result


def _metadata(path: Path, expected: tuple[str, ...]) -> dict[str, str]:
    rows = _rows(path, expected)
    result: dict[str, str] = {}
    for row in rows:
        recording_id, subject = row["Id"], row["Subject"]
        if not recording_id or not subject or recording_id != recording_id.strip():
            raise ValueError(f"{path.name} contains invalid metadata identity")
        if recording_id in result:
            raise ValueError(f"{path.name} contains duplicate Id {recording_id!r}")
        result[recording_id] = subject
    return result


def _submission(path: Path, test_counts: Mapping[str, int]) -> list[str]:
    rows = _rows(path, ("Id", *TARGETS))
    identifiers = [row["Id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("sample_submission.csv contains duplicate identifiers")
    observed: dict[str, set[int]] = {recording_id: set() for recording_id in test_counts}
    for identifier in identifiers:
        try:
            recording_id, raw_time = identifier.rsplit("_", 1)
            time = int(raw_time)
            count = test_counts[recording_id]
        except (KeyError, ValueError):
            raise ValueError(f"sample_submission.csv has unknown Id {identifier!r}") from None
        if not 0 <= time < count:
            raise ValueError(f"sample_submission.csv has out-of-range Id {identifier!r}")
        observed[recording_id].add(time)
    if any(len(observed[key]) != count for key, count in test_counts.items()):
        raise ValueError("sample_submission.csv does not cover every test row exactly once")
    return identifiers


def _row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        count = sum(1 for _ in reader)
    if count == 0:
        raise ValueError(f"{path.name} must contain at least one row")
    return count


def _rows(path: Path, expected: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing required Parkinson's metadata: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path.name} columns must be {list(expected)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} must contain at least one row")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name} contains ragged CSV rows")
    return rows


def _number(value: str, source: str, row: int, field: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{source} row {row} has invalid {field}={value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{source} row {row} has invalid {field}={value!r}")
    return result


def _integer(value: str, source: str, row: int, field: str) -> int:
    result = _number(value, source, row, field)
    if not result.is_integer():
        raise ValueError(f"{source} row {row} has non-integer {field}={value!r}")
    return int(result)


def _binary(value: str, source: str, row: int, field: str) -> float:
    result = _integer(value, source, row, field)
    if result not in (0, 1):
        raise ValueError(f"{source} row {row} has non-binary {field}={value!r}")
    return float(result)


def _boolean(value: str, source: str, row: int, field: str) -> bool:
    normalized = value.lower()
    if normalized not in ("true", "false"):
        raise ValueError(f"{source} row {row} has invalid {field}={value!r}")
    return normalized == "true"
