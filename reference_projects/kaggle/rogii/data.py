"""Strict paired-file ingestion for ROGII Wellbore Geology."""

from __future__ import annotations

import csv
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

TRAIN_HORIZONTAL = (
    "MD",
    "X",
    "Y",
    "Z",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
    "TVT",
    "GR",
    "TVT_input",
)
TEST_HORIZONTAL = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TRAIN_TYPEWELL = ("TVT", "GR", "Geology")
TEST_TYPEWELL = ("TVT", "GR")


def load_competition_data(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    train = _split(root / "train", labelled=True)
    test = _split(root / "test", labelled=False)
    test_ids = [well["well_id"] for well in test]
    excluded = sorted(set(test_ids) & {well["well_id"] for well in train})
    submission_ids = _submission(root / "sample_submission.csv")
    expected = [f"{well['well_id']}_{row['row_index']}" for well in test for row in well["tail"]]
    if submission_ids != expected:
        raise ValueError("sample_submission.csv must match test hidden-tail rows in order")
    return {
        "train": train,
        "test": test,
        "test_ids": test_ids,
        "excluded_train_ids": excluded,
        "submission_ids": submission_ids,
    }


def _split(directory: Path, *, labelled: bool) -> list[dict[str, Any]]:
    if not directory.is_dir():
        raise ValueError(f"missing ROGII directory: {directory}")
    horizontal = {
        path.name.removesuffix("__horizontal_well.csv"): path
        for path in directory.glob("*__horizontal_well.csv")
    }
    typewells = {
        path.name.removesuffix("__typewell.csv"): path for path in directory.glob("*__typewell.csv")
    }
    if not horizontal:
        raise ValueError(f"{directory} contains no horizontal wells")
    if set(horizontal) != set(typewells):
        missing = sorted(set(horizontal) ^ set(typewells))[:3]
        raise ValueError(f"every horizontal well requires one paired typewell; examples: {missing}")
    return [
        _well(well_id, horizontal[well_id], typewells[well_id], labelled=labelled)
        for well_id in sorted(horizontal)
    ]


def _well(well_id: str, horizontal: Path, typewell: Path, *, labelled: bool) -> dict[str, Any]:
    expected = TRAIN_HORIZONTAL if labelled else TEST_HORIZONTAL
    rows = _rows(horizontal, expected)
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = {
            "row_index": index,
            **{
                field: _number(row[field], horizontal.name, index + 2, field)
                for field in ("MD", "X", "Y", "Z")
            },
            "GR": _optional_number(row["GR"], horizontal.name, index + 2, "GR"),
            "TVT_input": _optional_number(
                row["TVT_input"], horizontal.name, index + 2, "TVT_input"
            ),
        }
        if labelled:
            item["TVT"] = _number(row["TVT"], horizontal.name, index + 2, "TVT")
        parsed.append(item)
    measured_depth = [row["MD"] for row in parsed]
    if any(right <= left for left, right in pairwise(measured_depth)):
        raise ValueError(f"{horizontal.name} MD must be strictly increasing")
    missing = [row["TVT_input"] is None for row in parsed]
    if not any(missing) or missing[0]:
        raise ValueError(f"{horizontal.name} requires a known TVT prefix and hidden tail")
    start = missing.index(True)
    if not all(missing[start:]):
        raise ValueError(f"{horizontal.name} TVT_input must be a contiguous prefix")
    if labelled and any(
        not math.isclose(float(row["TVT_input"]), float(row["TVT"]), abs_tol=1e-6)
        for row in parsed[:start]
    ):
        raise ValueError(f"{horizontal.name} TVT_input prefix must match TVT")
    type_rows = _rows(typewell, TRAIN_TYPEWELL if labelled else TEST_TYPEWELL)
    type_tvt = [
        _number(row["TVT"], typewell.name, index + 2, "TVT") for index, row in enumerate(type_rows)
    ]
    type_gr = [
        _number(row["GR"], typewell.name, index + 2, "GR") for index, row in enumerate(type_rows)
    ]
    known = [float(row["TVT_input"]) for row in parsed[:start]]
    return {
        "well_id": well_id,
        "tail": parsed[start:],
        "last_known_tvt": known[-1],
        "last_tvt_slope": known[-1] - known[-2] if len(known) > 1 else 0.0,
        "typewell": {"TVT": type_tvt, "GR": type_gr},
    }


def _submission(path: Path) -> list[str]:
    rows = _rows(path, ("id", "tvt"))
    identifiers = [row["id"] for row in rows]
    if any(not value or value != value.strip() for value in identifiers):
        raise ValueError("sample_submission.csv contains invalid identifiers")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("sample_submission.csv contains duplicate identifiers")
    return identifiers


def _rows(path: Path, expected: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing required ROGII CSV: {path}")
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


def _optional_number(value: str, source: str, row: int, field: str) -> float | None:
    return None if value == "" else _number(value, source, row, field)
