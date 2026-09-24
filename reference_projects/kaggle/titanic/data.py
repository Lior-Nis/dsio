"""Strict CSV ingestion for the Kaggle Titanic source schema."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

TRAIN_COLUMNS = (
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
)
TEST_COLUMNS = tuple(column for column in TRAIN_COLUMNS if column != "Survived")


def load_competition_data(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    train = _read(root / "train.csv", TRAIN_COLUMNS)
    test = _read(root / "test.csv", TEST_COLUMNS)
    train_ids = _validate_rows(train, labelled=True, source="train.csv")
    test_ids = _validate_rows(test, labelled=False, source="test.csv")
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"PassengerId appears in both train.csv and test.csv: {overlap[0]}")
    return {
        "train": train,
        "test": test,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "test_order": list(test_ids),
    }


def _read(path: Path, expected: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing required Titanic CSV: {path}")
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


def _validate_rows(rows: list[dict[str, str]], *, labelled: bool, source: str) -> list[str]:
    ids: list[str] = []
    for number, row in enumerate(rows, start=2):
        passenger_id = _integer(row["PassengerId"], source, number, "PassengerId", minimum=1)
        ids.append(str(passenger_id))
        _integer(row["Pclass"], source, number, "Pclass", minimum=1, maximum=3)
        _integer(row["SibSp"], source, number, "SibSp", minimum=0)
        _integer(row["Parch"], source, number, "Parch", minimum=0)
        _number(row["Age"], source, number, "Age", minimum=0.0, allow_empty=True)
        _number(row["Fare"], source, number, "Fare", minimum=0.0, allow_empty=True)
        if row["Sex"] not in {"male", "female"}:
            raise ValueError(f"{source} row {number} has invalid Sex {row['Sex']!r}")
        if row["Embarked"] not in {"", "C", "Q", "S"}:
            raise ValueError(f"{source} row {number} has invalid Embarked {row['Embarked']!r}")
        if not row["Ticket"]:
            raise ValueError(f"{source} row {number} has empty Ticket")
        if labelled:
            _integer(row["Survived"], source, number, "Survived", minimum=0, maximum=1)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source} contains duplicate PassengerId values")
    return ids


def _integer(
    value: str,
    source: str,
    row: int,
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{source} row {row} has invalid integer {field}={value!r}") from error
    if (
        str(parsed) != value.strip()
        or parsed < minimum
        or (maximum is not None and parsed > maximum)
    ):
        raise ValueError(f"{source} row {row} has invalid {field}={value!r}")
    return parsed


def _number(
    value: str,
    source: str,
    row: int,
    field: str,
    *,
    minimum: float,
    allow_empty: bool,
) -> float | None:
    if allow_empty and value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{source} row {row} has invalid number {field}={value!r}") from error
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{source} row {row} has invalid {field}={value!r}")
    return parsed
