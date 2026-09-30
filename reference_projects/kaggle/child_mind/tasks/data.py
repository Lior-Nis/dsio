"""Participant staging and stratified split evidence for CMI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from prefect import task

from dsio.data.adapters import TableExamples
from dsio.data.splits import generate
from dsio.data.store import SignalStore
from dsio.tracking import attempt, record_provenance, record_split_evidence
from reference_projects.kaggle.child_mind.data import (
    PACKED_FEATURES,
    load_competition_data,
    pack_participant,
    summarize_partition,
)

SPLIT_PARAMETERS = {"n_splits": 3, "target": "label"}


def labelled_examples(store: SignalStore) -> TableExamples:
    labelled = [entity for entity in store.entities if entity.attrs["source"] == "train"]
    return TableExamples(
        name=store.path.name,
        sample_ids=[entity.entity_id for entity in labelled],
        groups=[entity.group for entity in labelled],
        attributes={"label": [int(entity.attrs["target"]) for entity in labelled]},
        digest=store.identity,
    )


@task(persist_result=False)
def ingest(data_dir: str, workspace: str, experiment_id: str) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        loaded = load_competition_data(data_dir)
        path = Path(workspace) / run.info.run_id / "child-mind"
        path.parent.mkdir(parents=True, exist_ok=True)
        sensor_rows = 0
        sensor_bytes = 0
        train_sensor_participants = 0
        test_sensor_participants = 0
        test_sample_ids: list[str] = []
        with SignalStore.builder(
            path,
            channels=PACKED_FEATURES,
            dtype="float32",
            source=str(Path(data_dir).resolve()),
        ) as builder:
            for source in ("train", "test"):
                for order, row in enumerate(loaded[source]):
                    series_path = row["series_path"]
                    summary = (
                        None
                        if series_path is None
                        else summarize_partition(Path(series_path), str(row["id"]))
                    )
                    if summary is not None:
                        sensor_rows += summary.rows
                        sensor_bytes += summary.source_bytes
                        if source == "train":
                            train_sensor_participants += 1
                        else:
                            test_sensor_participants += 1
                    sample_id = f"{source}:{row['id']}"
                    attrs: dict[str, Any] = {
                        "source": source,
                        "source_id": str(row["id"]),
                        "source_order": order,
                        "sensor_present": summary is not None,
                    }
                    if source == "train":
                        attrs["target"] = int(row["target"])
                    else:
                        test_sample_ids.append(sample_id)
                    builder.add(
                        sample_id,
                        pack_participant(row, summary),
                        group=str(row["id"]),
                        attrs=attrs,
                    )
        store = SignalStore(path)
        store.verify()
        configuration = {
            "dataset_digest": store.identity,
            "schema": "kaggle-child-mind-v1",
            "labelled_participants": len(loaded["train"]),
            "test_participants": len(loaded["test"]),
            "dropped_missing_targets": loaded["dropped_missing_targets"],
            "train_sensor_participants": train_sensor_participants,
            "test_sensor_participants": test_sensor_participants,
            "sensor_rows": sensor_rows,
            "sensor_source_bytes": sensor_bytes,
        }
        identity = record_provenance(
            run.info.run_id,
            configuration,
            components={
                "loader": "reference_projects.kaggle.child_mind.data:load_competition_data",
                "sensor_summary": ("reference_projects.kaggle.child_mind.data:summarize_partition"),
            },
        )
        client = MlflowClient()
        for name in (
            "labelled_participants",
            "test_participants",
            "dropped_missing_targets",
            "train_sensor_participants",
            "test_sensor_participants",
            "sensor_rows",
            "sensor_source_bytes",
        ):
            client.log_metric(run.info.run_id, name, float(configuration[name]))
        client.log_dict(
            run.info.run_id,
            {
                "dataset_digest": store.identity,
                "test_ids": loaded["test_ids"],
                "test_sample_ids": test_sample_ids,
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
            **{key: configuration[key] for key in configuration if key != "dataset_digest"},
        }


@task(persist_result=False)
def split_data(data: dict[str, Any], experiment_id: str, seed: int) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = generate(
            examples,
            "stratified_group_kfold",
            name="child-mind-stratified-participants",
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
