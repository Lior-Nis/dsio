"""Raw CMI staging and participant-disjoint split evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from prefect import task

from dsio.data.splits import generate
from dsio.data.store import SignalStore
from dsio.tracking import attempt, record_provenance, record_split_evidence
from reference_projects.kaggle.child_mind.sequence.data import (
    sequence_examples,
    stage_sequence_store,
)

SPLIT_PARAMETERS = {"n_splits": 3, "target": "label"}


@task(persist_result=False)
def ingest(
    data_dir: str,
    workspace: str,
    experiment_id: str,
    window_length: int,
    existing_store_path: str | None = None,
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        if existing_store_path is None:
            path = Path(workspace) / run.info.run_id / "child-mind-sequences"
            path.parent.mkdir(parents=True, exist_ok=True)
            store = stage_sequence_store(data_dir, path, window_length=window_length)
        else:
            path = Path(existing_store_path)
            store = SignalStore(path)
            store.verify()
            attrs = store.manifest().attrs
            if attrs.get("schema") != "kaggle-child-mind-sequence-v1":
                raise ValueError(f"existing store {path} is not a CMI sequence store")
            if attrs.get("window_length") != window_length:
                raise ValueError(
                    f"existing store uses window_length={attrs.get('window_length')}, "
                    f"requested {window_length}"
                )
        examples = sequence_examples(store, window_length=window_length)
        present = sum(bool(entity.attrs["sensor_present"]) for entity in store.entities)
        raw_rows = sum(int(entity.attrs["source_rows"]) for entity in store.entities)
        configuration = {
            "dataset_digest": store.identity,
            "schema": "kaggle-child-mind-sequence-v1",
            "participant_count": len(store.entities),
            "sensor_present_participants": present,
            "sensor_missing_participants": len(store.entities) - present,
            "raw_sensor_rows": raw_rows,
            "stored_rows": store.n_rows,
            "window_count": len(examples),
            "window_length": window_length,
            "reused_store": existing_store_path is not None,
        }
        identity = record_provenance(
            run.info.run_id,
            configuration,
            components={
                "staging": (
                    "reference_projects.kaggle.child_mind.sequence.data:stage_sequence_store"
                ),
                "view": "reference_projects.kaggle.child_mind.sequence.data:sequence_examples",
            },
        )
        client = MlflowClient()
        for name, value in configuration.items():
            if isinstance(value, int) and not isinstance(value, bool):
                client.log_metric(run.info.run_id, name, float(value))
        client.log_dict(
            run.info.run_id,
            {"dataset_digest": store.identity, "store_path": str(path)},
            "outputs/dataset.json",
        )
        return {
            "store_path": str(path),
            "data_run_id": run.info.run_id,
            "identity": identity,
            **configuration,
        }


@task(persist_result=False)
def split_data(data: dict[str, Any], experiment_id: str, seed: int) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        store = SignalStore(data["store_path"])
        examples = sequence_examples(store, window_length=int(data["window_length"]))
        manifest = generate(
            examples,
            "stratified_group_kfold",
            name="child-mind-sequence-participants",
            seed=seed,
            roles=("train", "validate"),
            parameters=SPLIT_PARAMETERS,
        )
        fold = manifest.fold(0)
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "algorithm": "stratified_group_kfold",
                "parameters": SPLIT_PARAMETERS,
                "seed": seed,
                "fold": 0,
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


__all__ = ["SPLIT_PARAMETERS", "ingest", "split_data"]
