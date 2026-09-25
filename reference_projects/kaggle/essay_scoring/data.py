"""Strict CSV ingestion for Automated Essay Scoring 2.0."""

from __future__ import annotations

import csv
from pathlib import Path

TRAIN_COLUMNS = ("essay_id", "full_text", "score")
TEST_COLUMNS = ("essay_id", "full_text")
SUBMISSION_COLUMNS = ("essay_id", "score")


def load_competition_data(data_dir: str | Path) -> dict[str, object]:
    root = Path(data_dir)
    train = _essays(root / "train.csv", labelled=True)
    test = _essays(root / "test.csv", labelled=False)
    train_ids = [row["essay_id"] for row in train]
    test_ids = [row["essay_id"] for row in test]
    submission = _submission_ids(root / "sample_submission.csv")
    if submission != test_ids:
        raise ValueError("sample_submission.csv essay order must match test.csv")
    return {
        "train": train,
        "test": test,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }


def _essays(path: Path, *, labelled: bool) -> list[dict[str, object]]:
    expected = TRAIN_COLUMNS if labelled else TEST_COLUMNS
    rows: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for row, number in _rows(path, expected):
        essay_id = _identifier(row["essay_id"], path.name, number)
        full_text = _essay(row["full_text"], path.name, number)
        if essay_id in identifiers:
            raise ValueError(f"{path.name} contains duplicate essay_id={essay_id!r}")
        identifiers.add(essay_id)
        parsed: dict[str, object] = {"essay_id": essay_id, "full_text": full_text}
        if labelled:
            parsed["score"] = _score(row["score"], path.name, number)
        rows.append(parsed)
    return rows


def _submission_ids(path: Path) -> list[str]:
    return [
        _identifier(row["essay_id"], path.name, number)
        for row, number in _rows(path, SUBMISSION_COLUMNS)
    ]


def _rows(path: Path, expected: tuple[str, ...]) -> list[tuple[dict[str, str], int]]:
    if not path.is_file():
        raise ValueError(f"missing required Essay Scoring CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(
                f"{path.name} columns must be {list(expected)}, got {reader.fieldnames or []}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} must contain at least one row")
    result: list[tuple[dict[str, str], int]] = []
    for number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path.name} rows must contain exactly {len(expected)} values")
        result.append((row, number))
    return result


def _identifier(value: str, source: str, row: int) -> str:
    if not value.strip() or value != value.strip():
        raise ValueError(f"{source} row {row} has invalid essay_id")
    return value


def _essay(value: str, source: str, row: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{source} row {row} has invalid full_text")
    return normalized


def _score(value: str, source: str, row: int) -> int:
    try:
        score = int(value)
    except ValueError as error:
        raise ValueError(f"{source} row {row} has invalid score={value!r}") from error
    if str(score) != value.strip() or not 1 <= score <= 6:
        raise ValueError(f"{source} row {row} has invalid score={value!r}")
    return score
