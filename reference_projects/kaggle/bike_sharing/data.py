"""Strict source-schema validation for Bike Sharing Demand."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any

FEATURE_COLUMNS = (
    "datetime",
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
)
TRAIN_COLUMNS = (*FEATURE_COLUMNS, "casual", "registered", "count")


def load_competition_data(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    train = _read(root / "train.csv", TRAIN_COLUMNS)
    test = _read(root / "test.csv", FEATURE_COLUMNS)
    train_times = _validate(train, labelled=True, source="train.csv")
    test_times = _validate(test, labelled=False, source="test.csv")
    if set(train_times) & set(test_times):
        raise ValueError("train.csv and test.csv contain overlapping datetime identifiers")
    return {
        "train": train,
        "test": test,
        "train_ids": train_times,
        "test_ids": test_times,
        "test_order": list(test_times),
    }


def _read(path: Path, expected: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing required Bike Sharing CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(
                f"{path.name} columns must be {list(expected)}, got {reader.fieldnames or []}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} must contain at least one row")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name} rows must contain exactly {len(expected)} values")
    return rows


def _validate(rows: list[dict[str, str]], *, labelled: bool, source: str) -> list[str]:
    times: list[str] = []
    parsed_times: list[datetime] = []
    for number, row in enumerate(rows, start=2):
        try:
            parsed = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
        except ValueError as error:
            raise ValueError(
                f"{source} row {number} has invalid datetime {row['datetime']!r}"
            ) from error
        if parsed.strftime("%Y-%m-%d %H:%M:%S") != row["datetime"]:
            raise ValueError(f"{source} row {number} has noncanonical datetime {row['datetime']!r}")
        times.append(row["datetime"])
        parsed_times.append(parsed)
        for field, low, high in (
            ("season", 1, 4),
            ("holiday", 0, 1),
            ("workingday", 0, 1),
            ("weather", 1, 4),
        ):
            value = _integer(row[field], source, number, field)
            if not low <= value <= high:
                raise ValueError(f"{source} row {number} has invalid {field}={row[field]!r}")
        for field in ("temp", "atemp", "windspeed"):
            _number(row[field], source, number, field, minimum=0.0)
        humidity = _number(row["humidity"], source, number, "humidity", minimum=0.0)
        if humidity > 100:
            raise ValueError(f"{source} row {number} has invalid humidity={row['humidity']!r}")
        if labelled:
            casual = _integer(row["casual"], source, number, "casual")
            registered = _integer(row["registered"], source, number, "registered")
            count = _integer(row["count"], source, number, "count")
            if min(casual, registered, count) < 0:
                raise ValueError(f"{source} row {number} has a negative count")
            if count != casual + registered:
                raise ValueError(f"{source} row {number} count must equal casual + registered")
    if len(times) != len(set(times)):
        raise ValueError(f"{source} contains duplicate datetime identifiers")
    if parsed_times != sorted(parsed_times):
        raise ValueError(f"{source} datetime rows must be ordered chronologically")
    return times


def _integer(value: str, source: str, row: int, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{source} row {row} has invalid integer {field}={value!r}") from error
    if str(parsed) != value.strip():
        raise ValueError(f"{source} row {row} has invalid integer {field}={value!r}")
    return parsed


def _number(value: str, source: str, row: int, field: str, *, minimum: float) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{source} row {row} has invalid number {field}={value!r}") from error
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{source} row {row} has invalid {field}={value!r}")
    return parsed
