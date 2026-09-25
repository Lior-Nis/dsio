"""Strict, bounded ingestion for Kaggle Store Sales forecasting."""

from __future__ import annotations

import csv
import math
from collections import deque
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

CONTEXT_DAYS = 30
HORIZON_DAYS = 16
TRAIN_ORIGINS = 5
HISTORY_DAYS = CONTEXT_DAYS + HORIZON_DAYS * TRAIN_ORIGINS
TRAIN_COLUMNS = ("id", "date", "store_nbr", "family", "sales", "onpromotion")
TEST_COLUMNS = ("id", "date", "store_nbr", "family", "onpromotion")


def load_competition_data(data_dir: str | Path) -> dict[str, Any]:
    """Read only the bounded history needed by the reference forecaster.

    The official train file has millions of rows. Keeping a fixed tail per series makes the
    experiment representative without turning source parsing into an in-memory dataframe.
    """
    root = Path(data_dir)
    train, train_count, train_id_range = _read_train(root / "train.csv")
    test, test_ids = _read_test(root / "test.csv")
    if set(train) != set(test):
        missing = sorted(set(train) ^ set(test))[:3]
        raise ValueError(f"train/test series differ; examples: {missing}")
    for series_id in sorted(train):
        history = list(train[series_id])
        future = test[series_id]
        if len(history) != HISTORY_DAYS:
            raise ValueError(
                f"series {series_id!r} has {len(history)} retained train days; "
                f"expected {HISTORY_DAYS}"
            )
        if len(future) != HORIZON_DAYS:
            raise ValueError(
                f"series {series_id!r} has {len(future)} test days; expected {HORIZON_DAYS}"
            )
        _require_daily(history, series_id, "train")
        _require_daily(future, series_id, "test")
        if history[-1]["date"] + timedelta(days=1) != future[0]["date"]:
            raise ValueError(f"series {series_id!r} train/test dates are not consecutive")
    if not (train_id_range[1] < test_ids[0] or test_ids[-1] < train_id_range[0]):
        raise ValueError("train.csv and test.csv contain overlapping row identifiers")
    return {
        "series": {
            series_id: {"train": list(train[series_id]), "test": test[series_id]}
            for series_id in sorted(train)
        },
        "train_count": train_count,
        "test_ids": test_ids,
        "test_order": test_ids,
    }


def _read_train(
    path: Path,
) -> tuple[dict[str, deque[dict[str, Any]]], int, tuple[int, int]]:
    series: dict[str, deque[dict[str, Any]]] = {}
    first_identifier: int | None = None
    previous_identifier: int | None = None
    count = 0
    for row, number in _rows(path, TRAIN_COLUMNS):
        parsed = _parse_common(row, path.name, number)
        identifier = int(parsed["id"])
        if previous_identifier is not None and identifier <= previous_identifier:
            raise ValueError(f"{path.name} row identifiers must be unique and ordered")
        first_identifier = identifier if first_identifier is None else first_identifier
        previous_identifier = identifier
        sales = _number(row["sales"], path.name, number, "sales")
        parsed["sales"] = sales
        series.setdefault(parsed["series_id"], deque(maxlen=HISTORY_DAYS)).append(parsed)
        count += 1
    assert first_identifier is not None and previous_identifier is not None
    return series, count, (first_identifier, previous_identifier)


def _read_test(path: Path) -> tuple[dict[str, list[dict[str, Any]]], list[int]]:
    series: dict[str, list[dict[str, Any]]] = {}
    identifiers: list[int] = []
    for row, number in _rows(path, TEST_COLUMNS):
        parsed = _parse_common(row, path.name, number)
        series.setdefault(parsed["series_id"], []).append(parsed)
        identifiers.append(parsed["id"])
    _require_identifiers(identifiers, path.name)
    return series, identifiers


def _rows(path: Path, expected: tuple[str, ...]) -> Iterator[tuple[dict[str, str], int]]:
    if not path.is_file():
        raise ValueError(f"missing required Store Sales CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(
                f"{path.name} columns must be {list(expected)}, got {reader.fieldnames or []}"
            )
        seen = False
        for number, row in enumerate(reader, start=2):
            seen = True
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name} rows must contain exactly {len(expected)} values")
            yield row, number
    if not seen:
        raise ValueError(f"{path.name} must contain at least one row")


def _parse_common(row: dict[str, str], source: str, number: int) -> dict[str, Any]:
    identifier = _integer(row["id"], source, number, "id", minimum=0)
    store = _integer(row["store_nbr"], source, number, "store_nbr", minimum=1)
    promotion = _integer(row["onpromotion"], source, number, "onpromotion", minimum=0)
    family = row["family"].strip()
    if not family or family != row["family"]:
        raise ValueError(f"{source} row {number} has invalid family={row['family']!r}")
    try:
        parsed_date = date.fromisoformat(row["date"])
    except ValueError as error:
        raise ValueError(f"{source} row {number} has invalid date={row['date']!r}") from error
    if parsed_date.isoformat() != row["date"]:
        raise ValueError(f"{source} row {number} has noncanonical date={row['date']!r}")
    return {
        "id": identifier,
        "date": parsed_date,
        "store_nbr": store,
        "family": family,
        "series_id": f"store={store}|family={family}",
        "onpromotion": promotion,
    }


def _require_daily(rows: list[dict[str, Any]], series_id: str, source: str) -> None:
    dates = [row["date"] for row in rows]
    if len(dates) != len(set(dates)):
        raise ValueError(f"series {series_id!r} contains duplicate {source} dates")
    expected = [dates[0] + timedelta(days=offset) for offset in range(len(dates))]
    if dates != expected:
        raise ValueError(f"series {series_id!r} {source} dates must be consecutive and ordered")


def _require_identifiers(values: list[int], source: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{source} contains duplicate row identifiers")
    if values != sorted(values):
        raise ValueError(f"{source} row identifiers must be ordered")


def _integer(value: str, source: str, row: int, field: str, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{source} row {row} has invalid integer {field}={value!r}") from error
    if str(parsed) != value.strip() or parsed < minimum:
        raise ValueError(f"{source} row {row} has invalid {field}={value!r}")
    return parsed


def _number(value: str, source: str, row: int, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{source} row {row} has invalid number {field}={value!r}") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{source} row {row} has invalid {field}={value!r}")
    return parsed
