"""Strict source-schema validation for Kaggle Digit Recognizer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

PIXELS = tuple(f"pixel{index}" for index in range(784))
TRAIN_COLUMNS = ("label", *PIXELS)


def load_competition_data(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    train = _read(root / "train.csv", TRAIN_COLUMNS)
    test = _read(root / "test.csv", PIXELS)
    _validate(train, labelled=True, source="train.csv")
    _validate(test, labelled=False, source="test.csv")
    train_ids = [f"train-{index}" for index in range(1, len(train) + 1)]
    test_ids = [str(index) for index in range(1, len(test) + 1)]
    return {
        "train": train,
        "test": test,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "test_order": list(test_ids),
    }


def _read(path: Path, expected: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing required Digit Recognizer CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path.name} must contain exactly {len(expected)} official columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} must contain at least one row")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name} rows must contain exactly {len(expected)} values")
    return rows


def _validate(rows: list[dict[str, str]], *, labelled: bool, source: str) -> None:
    for number, row in enumerate(rows, start=2):
        if labelled:
            label = _integer(row["label"], source, number, "label")
            if not 0 <= label <= 9:
                raise ValueError(f"{source} row {number} has invalid label {row['label']!r}")
        for field in PIXELS:
            value = _integer(row[field], source, number, field)
            if not 0 <= value <= 255:
                raise ValueError(f"{source} row {number} has invalid {field}={row[field]!r}")


def _integer(value: str, source: str, row: int, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{source} row {row} has invalid integer {field}={value!r}") from error
    if str(parsed) != value.strip():
        raise ValueError(f"{source} row {row} has invalid integer {field}={value!r}")
    return parsed
