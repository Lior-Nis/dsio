from __future__ import annotations

import csv
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import mlflow
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))


@pytest.fixture
def kaggle_services(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith(("PREFECT_", "MLFLOW_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))
    monkeypatch.setenv("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    monkeypatch.setenv("PREFECT_LOGGING_LEVEL", "ERROR")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    tracking_uri = (tmp_path / "mlruns").resolve().as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def titanic_csvs(tmp_path: Path) -> Path:
    root = tmp_path / "titanic"
    train_fields = [
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
    ]
    test_fields = [field for field in train_fields if field != "Survived"]
    rows: list[dict[str, object]] = []
    for index in range(12):
        rows.append(
            {
                "PassengerId": index + 1,
                "Survived": index % 2,
                "Pclass": index % 3 + 1,
                "Name": f"Person {index}",
                "Sex": "female" if index % 2 else "male",
                "Age": 18 + index,
                "SibSp": index % 2,
                "Parch": 0,
                "Ticket": f"T-{index // 2}",
                "Fare": 10 + index,
                "Cabin": "",
                "Embarked": ("S", "C", "Q")[index % 3],
            }
        )
    write_csv(root / "train.csv", train_fields, rows)
    write_csv(
        root / "test.csv",
        test_fields,
        [
            {
                key: value
                for key, value in {**rows[index], "PassengerId": 101 + index}.items()
                if key != "Survived"
            }
            for index in range(4)
        ],
    )
    return root


@pytest.fixture
def bike_csvs(tmp_path: Path) -> Path:
    root = tmp_path / "bike"
    features = [
        "datetime",
        "season",
        "holiday",
        "workingday",
        "weather",
        "temp",
        "atemp",
        "humidity",
        "windspeed",
    ]
    train_fields = [*features, "casual", "registered", "count"]
    rows = []
    for index in range(24):
        casual, registered = index % 5, 5 + index
        rows.append(
            {
                "datetime": f"2011-01-{index // 24 + 1:02d} {index % 24:02d}:00:00",
                "season": 1,
                "holiday": 0,
                "workingday": int(index % 7 not in (5, 6)),
                "weather": index % 4 + 1,
                "temp": 8.0 + index / 2,
                "atemp": 9.0 + index / 2,
                "humidity": 40 + index,
                "windspeed": index / 3,
                "casual": casual,
                "registered": registered,
                "count": casual + registered,
            }
        )
    write_csv(root / "train.csv", train_fields, rows)
    write_csv(
        root / "test.csv",
        features,
        [
            {
                key: value
                for key, value in {
                    **rows[index],
                    "datetime": f"2011-02-01 {index:02d}:00:00",
                }.items()
                if key in features
            }
            for index in range(4)
        ],
    )
    return root


@pytest.fixture
def store_sales_csvs(tmp_path: Path) -> Path:
    root = tmp_path / "store-sales"
    train_fields = ["id", "date", "store_nbr", "family", "sales", "onpromotion"]
    test_fields = ["id", "date", "store_nbr", "family", "onpromotion"]
    families = ("BEVERAGES", "PRODUCE")
    start = date(2020, 1, 1)
    train: list[dict[str, object]] = []
    test: list[dict[str, object]] = []
    row_id = 0
    for offset in range(110):
        day = start + timedelta(days=offset)
        for store in (1, 2):
            for family_index, family in enumerate(families):
                train.append(
                    {
                        "id": row_id,
                        "date": day.isoformat(),
                        "store_nbr": store,
                        "family": family,
                        "sales": float(10 * store + family_index + offset % 7),
                        "onpromotion": (offset + family_index) % 4,
                    }
                )
                row_id += 1
    for offset in range(110, 126):
        day = start + timedelta(days=offset)
        for store in (1, 2):
            for family_index, family in enumerate(families):
                test.append(
                    {
                        "id": row_id,
                        "date": day.isoformat(),
                        "store_nbr": store,
                        "family": family,
                        "onpromotion": (offset + family_index) % 4,
                    }
                )
                row_id += 1
    write_csv(root / "train.csv", train_fields, train)
    write_csv(root / "test.csv", test_fields, test)
    return root


@pytest.fixture
def digit_csvs(tmp_path: Path) -> Path:
    root = tmp_path / "digits"
    pixels = [f"pixel{index}" for index in range(784)]
    train = []
    for row in range(20):
        values: dict[str, object] = {"label": row % 10}
        values.update({name: (row * 13 + index) % 256 for index, name in enumerate(pixels)})
        train.append(values)
    test = []
    for row in range(4):
        test.append({name: (row * 17 + index) % 256 for index, name in enumerate(pixels)})
    write_csv(root / "train.csv", ["label", *pixels], train)
    write_csv(root / "test.csv", pixels, test)
    return root
