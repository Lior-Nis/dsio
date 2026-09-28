"""Resolve the official Parkinson corpus across the centralized source roots."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

OFFICIAL_COUNTS = {
    "train_defog": 91,
    "train_tdcsfog": 833,
    "notype": 46,
    "daily": 65,
    "test": 2,
}
SUPPORTED_LANES = ("train_defog", "train_tdcsfog", "notype", "daily", "test")


def resolve_official_inventory(
    metadata_root: str | Path,
    labelled_root: str | Path,
    daily_root: str | Path,
    *,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Resolve every official source identity before any timed work begins."""
    expected_counts = OFFICIAL_COUNTS if expected_counts is None else expected_counts
    _validate_expected_counts(expected_counts)
    metadata = Path(metadata_root)
    labelled = Path(labelled_root)
    daily = Path(daily_root)
    defog_ids = _metadata_ids(
        metadata / "defog_metadata.csv", ("Id", "Subject", "Visit", "Medication")
    )
    tdcs_ids = _metadata_ids(
        metadata / "tdcsfog_metadata.csv",
        ("Id", "Subject", "Visit", "Test", "Medication"),
    )
    daily_ids = _metadata_ids(
        metadata / "daily_metadata.csv",
        ("Id", "Subject", "Visit", "Beginning of recording [00:00-23:59]"),
    )

    defog_files = _files(labelled / "defog" / "sessions", ".csv", "train_defog")
    tdcs_files = _files(labelled / "tdcsfog" / "sessions", ".csv", "train_tdcsfog")
    notype_files = _files(labelled / "notype" / "sessions", ".csv", "notype")
    daily_files = _files(daily, ".parquet", "daily")

    ambiguous = sorted(defog_ids & set(defog_files) & set(notype_files))
    if ambiguous:
        raise ValueError(
            f"defog identities are ambiguous across labelled and non-task lanes: {ambiguous}"
        )
    _require_exact_members("defog", defog_ids, set(defog_files) | set(notype_files))
    _require_exact_members("tdcsfog", tdcs_ids, set(tdcs_files))
    _require_exact_members("daily", daily_ids, set(daily_files))

    test: list[dict[str, Any]] = []
    for kind, official_ids in (("defog", defog_ids), ("tdcsfog", tdcs_ids)):
        paths = _files(metadata / "test" / kind, ".csv", f"test_{kind}")
        unknown = sorted(set(paths) - official_ids)
        if unknown:
            raise ValueError(f"test {kind} identities have no official metadata: {unknown}")
        test.extend(_entries(f"test_{kind}", paths, set(paths)))
    test_kinds = {str(item["kind"]) for item in test}
    required_test_kinds = {"defog", "tdcsfog"}
    if test_kinds != required_test_kinds:
        missing = sorted(required_test_kinds - test_kinds)
        raise ValueError(f"test source kinds must include defog and tdcsfog; missing {missing}")

    result: dict[str, Any] = {
        "train_defog": _entries("train_defog", defog_files, defog_ids),
        "train_tdcsfog": _entries("train_tdcsfog", tdcs_files, tdcs_ids),
        "notype": _entries("notype", notype_files, defog_ids),
        "daily": _entries("daily", daily_files, daily_ids),
        "test": sorted(test, key=lambda item: (item["kind"], item["recording_id"])),
        "extras": {
            "train_defog": sorted(set(defog_files) - defog_ids),
            "train_tdcsfog": sorted(set(tdcs_files) - tdcs_ids),
            "notype": sorted(set(notype_files) - defog_ids),
            "daily": sorted(set(daily_files) - daily_ids),
        },
    }
    for lane, expected in expected_counts.items():
        actual = len(result[lane])
        if actual != expected:
            raise ValueError(f"{lane} resolved {actual} files, expected {expected}")
    digest_input = {
        lane: [
            {
                "recording_id": item["recording_id"],
                "bytes": item["bytes"],
                "mtime_ns": item["mtime_ns"],
            }
            for item in result[lane]
        ]
        for lane in ("train_defog", "train_tdcsfog", "notype", "daily", "test")
    }
    result["manifest_digest"] = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result["source_files"] = sum(len(result[lane]) for lane in digest_input)
    result["source_bytes"] = sum(
        item["bytes"] for lane in digest_input for item in result[lane]
    )
    return result


def _validate_expected_counts(expected_counts: Mapping[str, int]) -> None:
    expected_lanes = set(SUPPORTED_LANES)
    actual_lanes = set(expected_counts)
    if actual_lanes != expected_lanes:
        raise ValueError(
            "expected_counts must define exactly the supported lanes: "
            f"{list(SUPPORTED_LANES)}"
        )
    for lane in SUPPORTED_LANES:
        value = expected_counts[lane]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"expected_counts must contain a positive integer for {lane}, got {value!r}"
            )


def _metadata_ids(path: Path, expected: tuple[str, ...]) -> set[str]:
    if not path.is_file():
        raise ValueError(f"missing official metadata: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path.name} columns must be {list(expected)}")
        identifiers: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name} row {row_number} is ragged")
            for column in expected:
                value = row[column]
                if not value or value != value.strip():
                    raise ValueError(
                        f"{path.name} row {row_number} has invalid {column}={value!r}"
                    )
            identifiers.append(row["Id"])
    if not identifiers:
        raise ValueError(f"{path.name} contains no identities")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{path.name} contains a duplicate identity")
    return set(identifiers)


def _files(directory: Path, suffix: str, lane: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"missing {lane} source directory: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.glob(f"*{suffix}")):
        identity = path.stem
        if identity in result:
            raise ValueError(f"{lane} identity {identity!r} is duplicated")
        result[identity] = path
    return result


def _require_exact_members(lane: str, expected: set[str], observed: set[str]) -> None:
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"{lane} identity {missing[0]} is missing ({len(missing)} missing)")


def _entries(
    lane: str, paths: Mapping[str, Path], official_ids: set[str]
) -> list[dict[str, Any]]:
    kind = "tdcsfog" if "tdcsfog" in lane else "defog" if "defog" in lane else lane
    result: list[dict[str, Any]] = []
    for identity in sorted(set(paths) & official_ids):
        path = paths[identity]
        stat = path.stat()
        result.append(
            {
                "lane": lane,
                "kind": kind,
                "recording_id": identity,
                "path": str(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return result
