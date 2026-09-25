"""Bounded Store Sales staging and rolling-origin split generation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from math import cos, log1p, pi, sin
from pathlib import Path
from typing import Any

import numpy as np
from mlflow import MlflowClient
from prefect import task

from dsio.data.adapters import TableExamples
from dsio.data.splits import generate
from dsio.data.store import SignalStore
from dsio.tracking import attempt, record_provenance, record_split_evidence
from reference_projects.kaggle.store_sales.data import (
    CONTEXT_DAYS,
    HORIZON_DAYS,
    TRAIN_ORIGINS,
    load_competition_data,
)

SPLIT_PARAMETERS = {
    "n_splits": 2,
    "test_fraction": 0.2,
    "label_horizon": 0.0,
    "embargo_fraction": 0.0,
    "mode": "expanding",
    "time_unit": "epoch_s",
}


def labelled_examples(store: SignalStore) -> TableExamples:
    labelled = [entity for entity in store.entities if entity.attrs["source"] == "train"]
    return TableExamples(
        name=store.path.name,
        sample_ids=[entity.entity_id for entity in labelled],
        groups=[entity.group for entity in labelled],
        times=(
            np.asarray([float(entity.attrs["target_start"]) for entity in labelled]),
            np.asarray([float(entity.attrs["target_end"]) for entity in labelled]),
        ),
        digest=store.identity,
    )


def _timestamp(value: date) -> float:
    return datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp()


def _features(context: list[dict[str, Any]], future: list[dict[str, Any]]) -> np.ndarray:
    rows: list[list[float]] = []
    for item in context:
        angle = 2 * pi * item["date"].weekday() / 7
        rows.append(
            [
                log1p(float(item["sales"])),
                log1p(float(item["onpromotion"])),
                sin(angle),
                cos(angle),
                0,
            ]
        )
    for item in future:
        angle = 2 * pi * item["date"].weekday() / 7
        rows.append([0, log1p(float(item["onpromotion"])), sin(angle), cos(angle), 1])
    return np.asarray(rows, dtype=np.float32)


@task(persist_result=False)
def ingest(data_dir: str, workspace: str, experiment_id: str) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        loaded = load_competition_data(data_dir)
        path = Path(workspace) / run.info.run_id / "store-sales"
        path.parent.mkdir(parents=True, exist_ok=True)
        train_sample_ids: list[str] = []
        test_sample_ids: list[str] = []
        with SignalStore.builder(path, channels=5, dtype="float32") as builder:
            for series_id, series in loaded["series"].items():
                history = series["train"]
                for number in range(TRAIN_ORIGINS):
                    origin = CONTEXT_DAYS + number * HORIZON_DAYS
                    context = history[origin - CONTEXT_DAYS : origin]
                    target_rows = history[origin : origin + HORIZON_DAYS]
                    sample_id = f"{series_id}|origin={target_rows[0]['date'].isoformat()}"
                    train_sample_ids.append(sample_id)
                    builder.add(
                        sample_id,
                        _features(context, target_rows),
                        group=series_id,
                        attrs={
                            "source": "train",
                            "series_id": series_id,
                            "target": [float(row["sales"]) for row in target_rows],
                            "target_start": _timestamp(target_rows[0]["date"]),
                            "target_end": _timestamp(target_rows[-1]["date"] + timedelta(days=1)),
                        },
                    )
                future = series["test"]
                sample_id = f"{series_id}|forecast={future[0]['date'].isoformat()}"
                test_sample_ids.append(sample_id)
                builder.add(
                    sample_id,
                    _features(history[-CONTEXT_DAYS:], future),
                    group=series_id,
                    attrs={
                        "source": "test",
                        "series_id": series_id,
                        "submission_ids": [int(row["id"]) for row in future],
                    },
                )
        store = SignalStore(path)
        store.verify()
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": store.identity,
                "source_train_rows": loaded["train_count"],
                "source_test_rows": len(loaded["test_ids"]),
                "series_count": len(loaded["series"]),
                "context_days": CONTEXT_DAYS,
                "horizon_days": HORIZON_DAYS,
                "train_origins": TRAIN_ORIGINS,
                "schema": "kaggle-store-sales-v1",
            },
            components={
                "loader": "reference_projects.kaggle.store_sales.data:load_competition_data"
            },
        )
        MlflowClient().log_dict(
            run.info.run_id,
            {
                "dataset_digest": store.identity,
                "train_sample_ids": train_sample_ids,
                "test_sample_ids": test_sample_ids,
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
            "test_sample_ids": test_sample_ids,
        }


@task(persist_result=False)
def split_data(data: dict[str, Any], experiment_id: str, seed: int) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = generate(
            examples,
            "purged_walk_forward",
            name="store-sales-rolling-origins",
            seed=seed,
            roles=("train", "validate"),
            parameters=SPLIT_PARAMETERS,
        )
        entities = {entity.entity_id: entity for entity in store.entities}
        assignments = [fold.assignments for fold in manifest.folds]
        boundaries = []
        for fold in manifest.folds:
            boundaries.append(
                {
                    "max_train_target_end": max(
                        float(entities[value].attrs["target_end"])
                        for value in fold.assignments["train"]
                    ),
                    "min_validate_target_start": min(
                        float(entities[value].attrs["target_start"])
                        for value in fold.assignments["validate"]
                    ),
                }
            )
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "algorithm": "purged_walk_forward",
                "parameters": SPLIT_PARAMETERS,
                "seed": seed,
                "fold_assignments": assignments,
            },
            components={"splitter": "dsio.data.splits.generate:generate"},
        )
        uri = record_split_evidence(run.info.run_id, examples, manifest, source=store.path.name)
        return {
            "split_run_id": run.info.run_id,
            "split_uri": uri,
            "split_digest": manifest.digest,
            "fold_assignments": assignments,
            "fold_boundaries": boundaries,
            "identity": identity,
        }
