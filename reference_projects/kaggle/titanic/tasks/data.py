"""CSV-to-store ingestion and ticket-group splitting for Titanic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mlflow import MlflowClient
from prefect import task

from dsio.data.adapters import TableExamples
from dsio.data.splits import generate
from dsio.data.store import SignalStore
from dsio.tracking import attempt, record_provenance, record_split_evidence
from reference_projects.kaggle.titanic.data import load_competition_data


def labelled_examples(store: SignalStore) -> TableExamples:
    labelled = [entity for entity in store.entities if entity.attrs["source"] == "train"]
    return TableExamples(
        name=store.path.name,
        sample_ids=[entity.entity_id for entity in labelled],
        groups=[str(entity.attrs["ticket"]) for entity in labelled],
        attributes={"label": [int(entity.attrs["target"]) for entity in labelled]},
        digest=store.identity,
    )


def _features(row: dict[str, str]) -> np.ndarray[Any, np.dtype[np.float32]]:
    age_missing, fare_missing = row["Age"] == "", row["Fare"] == ""
    return np.asarray(
        [
            [
                float(row["Pclass"]),
                float(row["Sex"] == "female"),
                0.0 if age_missing else float(row["Age"]),
                float(age_missing),
                float(row["SibSp"]),
                float(row["Parch"]),
                0.0 if fare_missing else float(row["Fare"]),
                float(fare_missing),
                {"": 0.0, "S": 1.0, "C": 2.0, "Q": 3.0}[row["Embarked"]],
                float(bool(row["Cabin"])),
            ]
        ],
        dtype=np.float32,
    )


@task(persist_result=False)
def ingest(data_dir: str, workspace: str, experiment_id: str) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        loaded = load_competition_data(data_dir)
        path = Path(workspace) / run.info.run_id / "titanic"
        path.parent.mkdir(parents=True, exist_ok=True)
        with SignalStore.builder(path, channels=10, dtype="float32") as builder:
            for source, rows in (("train", loaded["train"]), ("test", loaded["test"])):
                for order, row in enumerate(rows):
                    attrs: dict[str, Any] = {
                        "source": source,
                        "source_order": order,
                        "ticket": row["Ticket"],
                    }
                    if source == "train":
                        attrs["target"] = int(row["Survived"])
                    builder.add(
                        row["PassengerId"], _features(row), group=row["Ticket"], attrs=attrs
                    )
        store = SignalStore(path)
        store.verify()
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": store.identity,
                "train_rows": len(loaded["train"]),
                "test_rows": len(loaded["test"]),
                "schema": "kaggle-titanic-v1",
            },
            components={"loader": "reference_projects.kaggle.titanic.data:load_competition_data"},
        )
        MlflowClient().log_dict(
            run.info.run_id,
            {
                "dataset_digest": store.identity,
                "train_ids": loaded["train_ids"],
                "test_ids": loaded["test_ids"],
            },
            "outputs/dataset.json",
        )
        return {
            "store_path": str(path),
            "dataset_digest": store.identity,
            "data_run_id": run.info.run_id,
            "identity": identity,
            "test_ids": loaded["test_ids"],
            "tickets": {row["PassengerId"]: row["Ticket"] for row in loaded["train"]},
        }


@task(persist_result=False)
def split_data(data: dict[str, Any], experiment_id: str, seed: int) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = generate(
            examples,
            "group_shuffle",
            name="titanic-ticket-holdout",
            seed=seed,
            roles=("train", "validate"),
            parameters={"test_size": 0.34},
        )
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "algorithm": "group_shuffle",
                "seed": seed,
                "parameters": {"test_size": 0.34},
            },
            components={"splitter": "dsio.data.splits.generate:generate"},
        )
        uri = record_split_evidence(run.info.run_id, examples, manifest, source=store.path.name)
        return {
            "split_run_id": run.info.run_id,
            "split_uri": uri,
            "split_digest": manifest.digest,
            "assignments": manifest.fold(0).assignments,
            "identity": identity,
        }
