"""Data construction and split generation tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from prefect import task

from dsio.data.adapters import entity_examples
from dsio.data.splits import generate
from dsio.data.store import SignalStore
from dsio.tracking import (
    attempt,
    evidence_uri,
    record_provenance,
    record_split_evidence,
)
from reference_projects.supervised.components import build_synthetic_store


@task(persist_result=False)
def build_data(workspace: str, parent_run_id: str, seed: int) -> dict[str, Any]:
    with attempt(parent_run_id) as child:
        identity = record_provenance(
            child.info.run_id,
            {"seed": seed, "samples": 8, "rows": 4, "channels": 1},
            components={
                "builder": "reference_projects.supervised.components:build_synthetic_store"
            },
        )
        path = Path(workspace) / child.info.run_id / "synthetic-supervised"
        path.parent.mkdir(parents=True, exist_ok=True)
        store = build_synthetic_store(path, seed)
        store.verify()
        examples = entity_examples(store)
        MlflowClient().log_dict(
            child.info.run_id,
            {
                "dataset_digest": examples.digest,
                "sample_ids": examples.sample_ids.tolist(),
            },
            "outputs/dataset.json",
        )
        return {
            "store_path": str(path),
            "data_run_id": child.info.run_id,
            "dataset_digest": examples.digest,
            "identity": identity,
        }


@task(persist_result=False)
def split_data(
    data: dict[str, Any],
    parent_run_id: str,
    seed: int,
    *,
    split_name: str = "supervised-holdout",
) -> dict[str, Any]:
    with attempt(parent_run_id) as child:
        store = SignalStore(data["store_path"])
        examples = entity_examples(store)
        identity = record_provenance(
            child.info.run_id,
            {
                "dataset_digest": examples.digest,
                "algorithm": "group_shuffle",
                "name": split_name,
                "test_size": 0.25,
                "seed": seed,
            },
            components={"splitter": "dsio.data.splits.generate:generate"},
        )
        manifest = generate(
            examples,
            "group_shuffle",
            name=split_name,
            seed=seed,
            parameters={"test_size": 0.25},
        )
        uri = record_split_evidence(
            child.info.run_id,
            examples,
            manifest,
            source=store.path.name,
        )
        MlflowClient().set_tag(
            child.info.run_id,
            "dsio.source.dataset_uri",
            evidence_uri(data["data_run_id"], "outputs/dataset.json"),
        )
        return {
            "split_run_id": child.info.run_id,
            "split_uri": uri,
            "split_digest": manifest.digest,
            "assignments": manifest.fold(0).assignments,
            "identity": identity,
        }
