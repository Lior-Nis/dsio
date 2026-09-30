"""Strict CMI boundary and bounded participant-level sensor summaries."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

SAFE_COLUMNS = (
    "id",
    "Basic_Demos-Enroll_Season",
    "Basic_Demos-Age",
    "Basic_Demos-Sex",
    "CGAS-Season",
    "CGAS-CGAS_Score",
    "Physical-Season",
    "Physical-BMI",
    "Physical-Height",
    "Physical-Weight",
    "Physical-Waist_Circumference",
    "Physical-Diastolic_BP",
    "Physical-HeartRate",
    "Physical-Systolic_BP",
    "Fitness_Endurance-Season",
    "Fitness_Endurance-Max_Stage",
    "Fitness_Endurance-Time_Mins",
    "Fitness_Endurance-Time_Sec",
    "FGC-Season",
    "FGC-FGC_CU",
    "FGC-FGC_CU_Zone",
    "FGC-FGC_GSND",
    "FGC-FGC_GSND_Zone",
    "FGC-FGC_GSD",
    "FGC-FGC_GSD_Zone",
    "FGC-FGC_PU",
    "FGC-FGC_PU_Zone",
    "FGC-FGC_SRL",
    "FGC-FGC_SRL_Zone",
    "FGC-FGC_SRR",
    "FGC-FGC_SRR_Zone",
    "FGC-FGC_TL",
    "FGC-FGC_TL_Zone",
    "BIA-Season",
    "BIA-BIA_Activity_Level_num",
    "BIA-BIA_BMC",
    "BIA-BIA_BMI",
    "BIA-BIA_BMR",
    "BIA-BIA_DEE",
    "BIA-BIA_ECW",
    "BIA-BIA_FFM",
    "BIA-BIA_FFMI",
    "BIA-BIA_FMI",
    "BIA-BIA_Fat",
    "BIA-BIA_Frame_num",
    "BIA-BIA_ICW",
    "BIA-BIA_LDM",
    "BIA-BIA_LST",
    "BIA-BIA_SMM",
    "BIA-BIA_TBW",
    "PAQ_A-Season",
    "PAQ_A-PAQ_A_Total",
    "PAQ_C-Season",
    "PAQ_C-PAQ_C_Total",
    "SDS-Season",
    "SDS-SDS_Total_Raw",
    "SDS-SDS_Total_T",
    "PreInt_EduHx-Season",
    "PreInt_EduHx-computerinternet_hoursday",
)
SAFE_FEATURE_COLUMNS = SAFE_COLUMNS[1:]
PCIAT_COLUMNS = (
    "PCIAT-Season",
    *(f"PCIAT-PCIAT_{index:02d}" for index in range(1, 21)),
    "PCIAT-PCIAT_Total",
)
_PCIAT_POSITION = SAFE_COLUMNS.index("SDS-Season")
TRAIN_COLUMNS = (
    *SAFE_COLUMNS[:_PCIAT_POSITION],
    *PCIAT_COLUMNS,
    *SAFE_COLUMNS[_PCIAT_POSITION:],
    "sii",
)

SENSOR_PARQUET_COLUMNS = (
    ("step", "uint32"),
    ("X", "float32"),
    ("Y", "float32"),
    ("Z", "float32"),
    ("enmo", "float32"),
    ("anglez", "float32"),
    ("non-wear_flag", "float32"),
    ("light", "float32"),
    ("battery_voltage", "float32"),
    ("time_of_day", "int64"),
    ("weekday", "int8"),
    ("quarter", "int8"),
    ("relative_date_PCIAT", "float32"),
)
SENSOR_COLUMNS = tuple(name for name, _ in SENSOR_PARQUET_COLUMNS[1:])
_SEASONS = {"Spring": 0.0, "Summer": 1.0, "Fall": 2.0, "Winter": 3.0}
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_SENSOR_STATS = 5
TABULAR_FEATURES = len(SAFE_FEATURE_COLUMNS)
SENSOR_FEATURES = 1 + len(SENSOR_COLUMNS) * _SENSOR_STATS
TABULAR_VALUE_SLICE = slice(0, TABULAR_FEATURES)
TABULAR_MASK_SLICE = slice(TABULAR_VALUE_SLICE.stop, TABULAR_VALUE_SLICE.stop + TABULAR_FEATURES)
SENSOR_VALUE_SLICE = slice(TABULAR_MASK_SLICE.stop, TABULAR_MASK_SLICE.stop + SENSOR_FEATURES)
SENSOR_MASK_SLICE = slice(SENSOR_VALUE_SLICE.stop, SENSOR_VALUE_SLICE.stop + SENSOR_FEATURES)
SENSOR_PRESENT_INDEX = SENSOR_MASK_SLICE.stop
PACKED_FEATURES = SENSOR_PRESENT_INDEX + 1


class SensorSummary(NamedTuple):
    values: np.ndarray[Any, np.dtype[np.float32]]
    observed: np.ndarray[Any, np.dtype[np.float32]]
    rows: int
    source_bytes: int


def load_competition_data(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    train_rows = _rows(root / "train.csv", TRAIN_COLUMNS)
    test_rows = _rows(root / "test.csv", SAFE_COLUMNS)
    _unique_ids(train_rows, "train.csv")
    _unique_ids(test_rows, "test.csv")

    train_series = _partitions(root / "series_train.parquet", train_rows, "train")
    test_series = _partitions(root / "series_test.parquet", test_rows, "test")
    labelled: list[dict[str, Any]] = []
    dropped = 0
    for row in train_rows:
        target = _target(row["sii"], str(row["id"]))
        if target is None:
            dropped += 1
            continue
        labelled.append(
            {
                **row,
                "target": target,
                "series_path": train_series.get(str(row["id"])),
            }
        )
    test = [{**row, "series_path": test_series.get(str(row["id"]))} for row in test_rows]
    submission = _rows(root / "sample_submission.csv", ("id", "sii"))
    submission_ids = [str(row["id"]) for row in submission]
    test_ids = [str(row["id"]) for row in test]
    if submission_ids != test_ids:
        raise ValueError("sample_submission.csv IDs must match test.csv order exactly")
    return {
        "train": labelled,
        "test": test,
        "test_ids": test_ids,
        "dropped_missing_targets": dropped,
        "train_series_count": len(train_series),
        "test_series_count": len(test_series),
    }


def summarize_partition(path: Path, participant_id: str) -> SensorSummary:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "CMI sensor ingestion requires the DSIO 'data' extra (pyarrow)"
        ) from error

    try:
        parquet = pq.ParquetFile(path)
    except Exception as error:
        raise ValueError(
            f"sensor partition {participant_id!r} cannot be opened: {error}"
        ) from error
    expected = [name for name, _ in SENSOR_PARQUET_COLUMNS]
    if parquet.schema_arrow.names != expected:
        raise ValueError(f"sensor partition {participant_id!r} columns must be {expected}")
    declared_types = dict(SENSOR_PARQUET_COLUMNS)
    expected_types = {
        name: pa.type_for_alias(field_type) for name, field_type in SENSOR_PARQUET_COLUMNS
    }
    for field in parquet.schema_arrow:
        if field.type != expected_types[field.name]:
            raise ValueError(
                f"sensor partition {participant_id!r} field {field.name!r} must have "
                f"Arrow type {declared_types[field.name]}, found {field.type}"
            )

    count = np.zeros(len(SENSOR_COLUMNS), dtype=np.int64)
    mean = np.zeros(len(SENSOR_COLUMNS), dtype=np.float64)
    sum_squared_deviations = np.zeros(len(SENSOR_COLUMNS), dtype=np.float64)
    minimum = np.full(len(SENSOR_COLUMNS), np.inf, dtype=np.float64)
    maximum = np.full(len(SENSOR_COLUMNS), -np.inf, dtype=np.float64)
    rows = 0
    previous_step: int | None = None
    try:
        for batch in parquet.iter_batches(batch_size=65_536, columns=["step", *SENSOR_COLUMNS]):
            rows += batch.num_rows
            step_column = batch.column(0)
            if step_column.null_count:
                raise ValueError("step must be non-null")
            steps = np.asarray(step_column.to_numpy(zero_copy_only=False), dtype=np.int64)
            if steps.size and (
                (previous_step is not None and int(steps[0]) <= previous_step)
                or bool((np.diff(steps) <= 0).any())
            ):
                raise ValueError("step must be strictly increasing and unique")
            if steps.size:
                previous_step = int(steps[-1])
            matrix = np.column_stack(
                [
                    np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
                    for column in batch.columns[1:]
                ]
            )
            infinite = np.argwhere(np.isinf(matrix))
            if infinite.size:
                raise ValueError(f"field {SENSOR_COLUMNS[int(infinite[0, 1])]!r} contains infinity")
            valid = np.isfinite(matrix)
            batch_count = valid.sum(axis=0, dtype=np.int64)
            safe = np.where(valid, matrix, 0.0)
            batch_mean = np.divide(
                safe.sum(axis=0, dtype=np.float64),
                batch_count,
                out=np.zeros_like(mean),
                where=batch_count > 0,
            )
            deviations = np.where(valid, matrix - batch_mean, 0.0)
            batch_squared_deviations = np.square(deviations).sum(axis=0, dtype=np.float64)
            combined_count = count + batch_count
            delta = batch_mean - mean
            merge_weight = np.divide(
                batch_count,
                combined_count,
                out=np.zeros_like(mean),
                where=combined_count > 0,
            )
            sum_squared_deviations += batch_squared_deviations + np.divide(
                np.square(delta) * count * batch_count,
                combined_count,
                out=np.zeros_like(mean),
                where=combined_count > 0,
            )
            mean += delta * merge_weight
            count = combined_count
            minimum = np.minimum(minimum, np.min(np.where(valid, matrix, np.inf), axis=0))
            maximum = np.maximum(maximum, np.max(np.where(valid, matrix, -np.inf), axis=0))
    except ValueError as error:
        raise ValueError(f"sensor partition {participant_id!r} is malformed: {error}") from error
    except Exception as error:
        raise ValueError(f"sensor partition {participant_id!r} cannot be read: {error}") from error
    if rows == 0:
        raise ValueError(f"sensor partition {participant_id!r} is empty")

    values = [math.log1p(rows)]
    observed = [1.0]
    for index in range(len(SENSOR_COLUMNS)):
        available = int(count[index])
        if available:
            variance = max(sum_squared_deviations[index] / available, 0.0)
            values.extend(
                [
                    mean[index],
                    math.sqrt(variance),
                    minimum[index],
                    maximum[index],
                    1 - available / rows,
                ]
            )
            observed.extend([1.0] * _SENSOR_STATS)
        else:
            values.extend([0.0, 0.0, 0.0, 0.0, 1.0])
            observed.extend([0.0, 0.0, 0.0, 0.0, 1.0])
    with np.errstate(over="ignore", invalid="ignore"):
        packed_values = np.asarray(values, dtype=np.float32)
    if not bool(np.isfinite(packed_values).all()):
        raise ValueError(
            f"sensor partition {participant_id!r} summary cannot be represented as float32"
        )
    return SensorSummary(
        packed_values,
        np.asarray(observed, dtype=np.float32),
        rows,
        path.stat().st_size,
    )


def pack_participant(
    row: Mapping[str, Any], summary: SensorSummary | None
) -> np.ndarray[Any, np.dtype[np.float32]]:
    tabular = np.zeros(TABULAR_FEATURES, dtype=np.float32)
    tabular_mask = np.zeros(TABULAR_FEATURES, dtype=np.float32)
    for index, field in enumerate(SAFE_FEATURE_COLUMNS):
        raw = row[field]
        if raw == "" or raw is None:
            continue
        if field.endswith("Season"):
            try:
                value = _SEASONS[str(raw)]
            except KeyError:
                raise ValueError(f"participant {row['id']!r} has invalid {field}={raw!r}") from None
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"participant {row['id']!r} has non-numeric {field}={raw!r}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"participant {row['id']!r} has non-finite {field}={raw!r}")
        if abs(value) > _FLOAT32_MAX:
            raise ValueError(
                f"participant {row['id']!r} field {field!r} is outside float32 range: {raw!r}"
            )
        tabular[index] = value
        tabular_mask[index] = 1.0

    sensor = np.zeros(SENSOR_FEATURES, dtype=np.float32)
    sensor_mask = np.zeros(SENSOR_FEATURES, dtype=np.float32)
    present = 0.0
    if summary is not None:
        sensor[:] = summary.values
        sensor_mask[:] = summary.observed
        present = 1.0
    packed = np.concatenate(
        [tabular, tabular_mask, sensor, sensor_mask, np.asarray([present], dtype=np.float32)]
    )
    return packed.reshape(1, -1)


def _rows(path: Path, expected: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing required CMI file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path.name} columns must be {list(expected)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} must contain at least one row")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name} contains ragged rows")
    return rows


def _unique_ids(rows: list[dict[str, str]], source: str) -> None:
    identifiers = [row["id"] for row in rows]
    if any(not value or value != value.strip() for value in identifiers):
        raise ValueError(f"{source} contains invalid participant identity")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{source} contains duplicate participant identity")


def _partitions(directory: Path, rows: list[dict[str, str]], source: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"missing CMI sensor directory: {directory}")
    known = {row["id"] for row in rows}
    result: dict[str, Path] = {}
    for partition in sorted(directory.glob("id=*")):
        participant_id = partition.name.removeprefix("id=")
        if not participant_id or participant_id not in known:
            raise ValueError(
                f"series_{source}.parquet contains unknown participant {participant_id!r}"
            )
        files = list(partition.glob("*.parquet"))
        if len(files) != 1 or files[0].name != "part-0.parquet":
            raise ValueError(
                f"sensor participant {participant_id!r} must contain only part-0.parquet"
            )
        result[participant_id] = files[0]
    return result


def _target(value: str, participant_id: str) -> int | None:
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"participant {participant_id!r} has invalid sii={value!r}") from error
    if not number.is_integer() or int(number) not in range(4):
        raise ValueError(f"participant {participant_id!r} has invalid sii={value!r}")
    return int(number)


__all__ = [
    "PACKED_FEATURES",
    "PCIAT_COLUMNS",
    "SAFE_COLUMNS",
    "SAFE_FEATURE_COLUMNS",
    "SENSOR_COLUMNS",
    "SENSOR_FEATURES",
    "SENSOR_MASK_SLICE",
    "SENSOR_PARQUET_COLUMNS",
    "SENSOR_PRESENT_INDEX",
    "SENSOR_VALUE_SLICE",
    "TABULAR_FEATURES",
    "TABULAR_MASK_SLICE",
    "TABULAR_VALUE_SLICE",
    "TRAIN_COLUMNS",
    "SensorSummary",
    "load_competition_data",
    "pack_participant",
    "summarize_partition",
]
