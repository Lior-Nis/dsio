"""Bounded window staging and subject-disjoint split generation."""

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
from reference_projects.kaggle.parkinsons_fog.components import CHANNEL_COUNT
from reference_projects.kaggle.parkinsons_fog.data import (
    WINDOW_SIZE,
    iter_recording_windows,
    load_competition_data,
)

SPLIT_PARAMETERS = {"test_size": 0.34}


def labelled_examples(store: SignalStore) -> TableExamples:
    labelled = [entity for entity in store.entities if entity.attrs["source"] == "train"]
    return TableExamples(
        name=store.path.name,
        sample_ids=[entity.entity_id for entity in labelled],
        groups=[entity.group for entity in labelled],
        digest=store.identity,
    )


@task(persist_result=False)
def ingest(data_dir: str, workspace: str, experiment_id: str) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        loaded = load_competition_data(data_dir)
        excluded = set(loaded["excluded_train_ids"])
        path = Path(workspace) / run.info.run_id / "parkinsons-fog"
        path.parent.mkdir(parents=True, exist_ok=True)
        test_sample_ids: list[str] = []
        ignored_points = 0
        staged_points = 0
        dropped_windows = 0
        with SignalStore.builder(path, channels=CHANNEL_COUNT, dtype="float32") as builder:
            for source in ("train", "test"):
                for recording in loaded[source]:
                    recording_id = str(recording["recording_id"])
                    if source == "train" and recording_id in excluded:
                        continue
                    for window in iter_recording_windows(recording):
                        values = np.asarray(window["data"], dtype=np.float32)
                        valid = values[:, -1].astype(bool)
                        ignored_points += int((~valid).sum()) if source == "train" else 0
                        if source == "train" and not bool(valid.any()):
                            dropped_windows += 1
                            continue
                        start = int(window["start"])
                        sample_id = f"{source}:{recording['kind']}:{recording_id}:{start}"
                        attrs: dict[str, Any] = {
                            "source": source,
                            "kind": str(recording["kind"]),
                            "recording_id": recording_id,
                            "subject": str(recording["subject"]),
                            "start": start,
                        }
                        if source == "test":
                            attrs["submission_ids"] = [
                                f"{recording_id}_{time}" for time in window["times"]
                            ]
                            test_sample_ids.append(sample_id)
                        builder.add(
                            sample_id,
                            values,
                            group=str(recording["subject"]),
                            attrs=attrs,
                        )
                        staged_points += len(values)
        store = SignalStore(path)
        store.verify()
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": store.identity,
                "window_size": WINDOW_SIZE,
                "excluded_test_subjects": loaded["test_subjects"],
                "excluded_train_recordings": sorted(excluded),
                "ignored_points": ignored_points,
                "dropped_windows": dropped_windows,
                "staged_points": staged_points,
                "schema": "kaggle-parkinsons-fog-v1",
            },
            components={
                "loader": ("reference_projects.kaggle.parkinsons_fog.data:iter_recording_windows")
            },
        )
        MlflowClient().log_dict(
            run.info.run_id,
            {
                "dataset_digest": store.identity,
                "test_sample_ids": test_sample_ids,
                "submission_ids": loaded["submission_ids"],
            },
            "outputs/dataset.json",
        )
        return {
            "store_path": str(path),
            "dataset_digest": store.identity,
            "data_run_id": run.info.run_id,
            "identity": identity,
            "test_sample_ids": test_sample_ids,
            "submission_ids": loaded["submission_ids"],
            "ignored_points": ignored_points,
            "dropped_windows": dropped_windows,
            "staged_points": staged_points,
        }


@task(persist_result=False)
def split_data(data: dict[str, Any], experiment_id: str, seed: int) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = generate(
            examples,
            "group_shuffle",
            name="parkinsons-subject-holdout",
            seed=seed,
            roles=("train", "validate"),
            parameters=SPLIT_PARAMETERS,
        )
        fold = manifest.fold(0)
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "algorithm": "group_shuffle",
                "parameters": SPLIT_PARAMETERS,
                "seed": seed,
            },
            components={"splitter": "dsio.data.splits.generate:generate"},
        )
        uri = record_split_evidence(run.info.run_id, examples, manifest, source=store.path.name)
        return {
            "split_run_id": run.info.run_id,
            "split_uri": uri,
            "split_digest": manifest.digest,
            "assignments": fold.assignments,
            "split_groups": fold.parts,
            "identity": identity,
        }
