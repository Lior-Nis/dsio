"""CSV-to-store ingestion and governed splitting for Digit Recognizer."""

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
from reference_projects.kaggle.digit_recognizer.data import PIXELS, load_competition_data


def labelled_examples(store: SignalStore) -> TableExamples:
    labelled = [entity for entity in store.entities if entity.attrs["source"] == "train"]
    return TableExamples(
        name=store.path.name,
        sample_ids=[entity.entity_id for entity in labelled],
        groups=[entity.entity_id for entity in labelled],
        digest=store.identity,
    )


def _pixels(row: dict[str, str]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return np.asarray([int(row[name]) for name in PIXELS], dtype=np.uint8).reshape(28, 28)


@task(persist_result=False)
def ingest(data_dir: str, workspace: str, parent_run_id: str) -> dict[str, Any]:
    with attempt(parent_run_id) as child:
        loaded = load_competition_data(data_dir)
        path = Path(workspace) / child.info.run_id / "digit-recognizer"
        path.parent.mkdir(parents=True, exist_ok=True)
        with SignalStore.builder(path, channels=28, dtype="uint8") as builder:
            for sample_id, row in zip(loaded["train_ids"], loaded["train"], strict=True):
                builder.add(
                    sample_id,
                    _pixels(row),
                    group=sample_id,
                    attrs={"source": "train", "target": int(row["label"])},
                )
            for order, (sample_id, row) in enumerate(
                zip(loaded["test_ids"], loaded["test"], strict=True)
            ):
                builder.add(
                    sample_id,
                    _pixels(row),
                    group=sample_id,
                    attrs={"source": "test", "source_order": order},
                )
        store = SignalStore(path)
        store.verify()
        identity = record_provenance(
            child.info.run_id,
            {
                "dataset_digest": store.identity,
                "train_rows": len(loaded["train"]),
                "test_rows": len(loaded["test"]),
                "pixels": 784,
                "schema": "kaggle-digit-recognizer-v1",
            },
            components={
                "loader": "reference_projects.kaggle.digit_recognizer.data:load_competition_data"
            },
        )
        MlflowClient().log_dict(
            child.info.run_id,
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
            "data_run_id": child.info.run_id,
            "identity": identity,
            "test_ids": loaded["test_ids"],
        }


@task(persist_result=False)
def split_data(data: dict[str, Any], parent_run_id: str, seed: int) -> dict[str, Any]:
    with attempt(parent_run_id) as child:
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = generate(
            examples,
            "group_shuffle",
            name="digit-holdout",
            seed=seed,
            roles=("train", "validate"),
            parameters={"test_size": 0.2},
        )
        identity = record_provenance(
            child.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "algorithm": "group_shuffle",
                "parameters": {"test_size": 0.2},
                "seed": seed,
            },
            components={"splitter": "dsio.data.splits.generate:generate"},
        )
        uri = record_split_evidence(child.info.run_id, examples, manifest, source=store.path.name)
        return {
            "split_run_id": child.info.run_id,
            "split_uri": uri,
            "split_digest": manifest.digest,
            "assignments": manifest.fold(0).assignments,
            "identity": identity,
        }
