"""CSV-to-store ingestion and final causal splitting for Bike Sharing."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from mlflow import MlflowClient
from prefect import task

from dsio.data.adapters import TableExamples
from dsio.data.splits import generate
from dsio.data.store import SignalStore
from dsio.tracking import attempt, record_provenance, record_split_evidence
from reference_projects.kaggle.bike_sharing.data import load_competition_data

SPLIT_PARAMETERS = {
    "n_splits": 1,
    "test_fraction": 0.25,
    "label_horizon": 3600.0,
    "embargo_fraction": 0.05,
    "time_unit": "epoch_s",
}


def labelled_examples(store: SignalStore) -> TableExamples:
    labelled = [entity for entity in store.entities if entity.attrs["source"] == "train"]
    starts = np.asarray([float(entity.attrs["time"]) for entity in labelled])
    return TableExamples(
        name=store.path.name,
        sample_ids=[entity.entity_id for entity in labelled],
        groups=["timeline"] * len(labelled),
        times=(starts, starts + 3600.0),
        digest=store.identity,
    )


def _timestamp(value: str) -> float:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()


def _features(row: dict[str, str]) -> np.ndarray[Any, np.dtype[np.float32]]:
    parsed = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
    return np.asarray(
        [
            [
                float(row["season"]),
                float(row["holiday"]),
                float(row["workingday"]),
                float(row["weather"]),
                float(row["temp"]),
                float(row["atemp"]),
                float(row["humidity"]),
                float(row["windspeed"]),
                float(parsed.hour),
            ]
        ],
        dtype=np.float32,
    )


@task(persist_result=False)
def ingest(data_dir: str, workspace: str, experiment_id: str) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        loaded = load_competition_data(data_dir)
        path = Path(workspace) / run.info.run_id / "bike-sharing"
        path.parent.mkdir(parents=True, exist_ok=True)
        with SignalStore.builder(path, channels=9, dtype="float32") as builder:
            for source, rows in (("train", loaded["train"]), ("test", loaded["test"])):
                for order, row in enumerate(rows):
                    attrs: dict[str, Any] = {
                        "source": source,
                        "source_order": order,
                        "time": _timestamp(row["datetime"]),
                    }
                    if source == "train":
                        attrs["target"] = int(row["count"])
                    builder.add(row["datetime"], _features(row), group="timeline", attrs=attrs)
        store = SignalStore(path)
        store.verify()
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": store.identity,
                "train_rows": len(loaded["train"]),
                "test_rows": len(loaded["test"]),
                "schema": "kaggle-bike-v1",
            },
            components={
                "loader": "reference_projects.kaggle.bike_sharing.data:load_competition_data"
            },
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
        }


@task(persist_result=False)
def split_data(data: dict[str, Any], experiment_id: str, seed: int) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = generate(
            examples,
            "purged_walk_forward",
            name="bike-final-purged-holdout",
            seed=seed,
            roles=("train", "validate"),
            parameters=SPLIT_PARAMETERS,
        )
        fold = manifest.fold(0)
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "algorithm": "purged_walk_forward",
                "parameters": SPLIT_PARAMETERS,
                "discarded_count": fold.counts["discarded"],
                "seed": seed,
            },
            components={"splitter": "dsio.data.splits.generate:generate"},
        )
        uri = record_split_evidence(run.info.run_id, examples, manifest, source=store.path.name)
        times = {entity.entity_id: float(entity.attrs["time"]) for entity in store.entities}
        return {
            "split_run_id": run.info.run_id,
            "split_uri": uri,
            "split_digest": manifest.digest,
            "assignments": fold.assignments,
            "discarded_count": fold.counts["discarded"],
            "train_times": [times[value] for value in fold.assignments["train"]],
            "validate_times": [times[value] for value in fold.assignments["validate"]],
            "identity": identity,
        }
