"""ROGII paired-file staging and well-disjoint split generation."""

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
from reference_projects.kaggle.rogii.components import FEATURES, well_arrays
from reference_projects.kaggle.rogii.data import load_competition_data

SPLIT_PARAMETERS = {"test_size": 0.25}


def labelled_examples(store: SignalStore) -> TableExamples:
    labelled = [entity for entity in store.entities if entity.attrs["source"] == "train"]
    return TableExamples(
        name=store.path.name,
        sample_ids=[entity.entity_id for entity in labelled],
        groups=[str(entity.attrs["well_id"]) for entity in labelled],
        digest=store.identity,
    )


@task(persist_result=False)
def ingest(data_dir: str, workspace: str, experiment_id: str) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        loaded = load_competition_data(data_dir)
        excluded = set(loaded["excluded_train_ids"])
        path = Path(workspace) / run.info.run_id / "rogii"
        path.parent.mkdir(parents=True, exist_ok=True)
        test_sample_ids: list[str] = []
        tail_lengths: dict[str, int] = {}
        with SignalStore.builder(path, channels=FEATURES + 1, dtype="float32") as builder:
            for source in ("train", "test"):
                for well in loaded[source]:
                    well_id = str(well["well_id"])
                    if source == "train" and well_id in excluded:
                        continue
                    sample_id = f"{source}:{well_id}"
                    x, y = well_arrays(well)
                    tail_lengths[sample_id] = len(x)
                    attrs: dict[str, Any] = {
                        "source": source,
                        "well_id": well_id,
                        "last_known_tvt": float(well["last_known_tvt"]),
                    }
                    if source == "test":
                        test_sample_ids.append(sample_id)
                    builder.add(
                        sample_id,
                        np.column_stack((x, y)),
                        group=well_id,
                        attrs=attrs,
                    )
        store = SignalStore(path)
        store.verify()
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": store.identity,
                "train_wells": len(loaded["train"]) - len(excluded),
                "test_wells": len(loaded["test"]),
                "excluded_test_wells": sorted(excluded),
                "schema": "kaggle-rogii-v1",
            },
            components={"loader": "reference_projects.kaggle.rogii.data:load_competition_data"},
        )
        MlflowClient().log_dict(
            run.info.run_id,
            {
                "dataset_digest": store.identity,
                "test_sample_ids": test_sample_ids,
                "test_ids": loaded["test_ids"],
                "submission_ids": loaded["submission_ids"],
                "tail_lengths": tail_lengths,
            },
            "outputs/dataset.json",
        )
        return {
            "store_path": str(path),
            "dataset_digest": store.identity,
            "data_run_id": run.info.run_id,
            "identity": identity,
            "test_sample_ids": test_sample_ids,
            "test_ids": loaded["test_ids"],
            "submission_ids": loaded["submission_ids"],
            "tail_lengths": tail_lengths,
            "max_tail_points": max(tail_lengths.values()),
        }


@task(persist_result=False)
def split_data(data: dict[str, Any], experiment_id: str, seed: int) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = generate(
            examples,
            "group_shuffle",
            name="rogii-well-holdout",
            seed=seed,
            roles=("train", "validate"),
            parameters=SPLIT_PARAMETERS,
        )
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
            "assignments": manifest.fold(0).assignments,
            "identity": identity,
        }
