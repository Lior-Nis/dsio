"""Essay token staging and stratified split generation."""

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
from reference_projects.kaggle.essay_scoring.components import tokenize
from reference_projects.kaggle.essay_scoring.data import load_competition_data

SPLIT_PARAMETERS = {"n_splits": 3, "target": "score"}


def labelled_examples(store: SignalStore) -> TableExamples:
    labelled = [entity for entity in store.entities if entity.attrs["source"] == "train"]
    return TableExamples(
        name=store.path.name,
        sample_ids=[entity.entity_id for entity in labelled],
        groups=[entity.entity_id for entity in labelled],
        attributes={"score": [int(entity.attrs["target"]) for entity in labelled]},
        digest=store.identity,
    )


def _tokens(text: str) -> np.ndarray[Any, np.dtype[np.int32]]:
    values = np.asarray(tokenize(text), dtype=np.int32)
    return np.column_stack((values, np.ones_like(values)))


@task(persist_result=False)
def ingest(data_dir: str, workspace: str, experiment_id: str) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        loaded = load_competition_data(data_dir)
        path = Path(workspace) / run.info.run_id / "essay-scoring"
        path.parent.mkdir(parents=True, exist_ok=True)
        test_sample_ids: list[str] = []
        with SignalStore.builder(path, channels=2, dtype="int32") as builder:
            for source in ("train", "test"):
                for order, row in enumerate(loaded[source]):
                    sample_id = f"{source}:{row['essay_id']}"
                    attrs: dict[str, Any] = {"source": source, "source_order": order}
                    if source == "train":
                        attrs["target"] = int(row["score"])
                    else:
                        test_sample_ids.append(sample_id)
                    builder.add(
                        sample_id,
                        _tokens(str(row["full_text"])),
                        group=str(row["essay_id"]),
                        attrs=attrs,
                    )
        store = SignalStore(path)
        store.verify()
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": store.identity,
                "train_rows": len(loaded["train"]),
                "test_rows": len(loaded["test"]),
                "schema": "kaggle-essay-scoring-v1",
            },
            components={
                "loader": ("reference_projects.kaggle.essay_scoring.data:load_competition_data"),
                "tokenizer": "reference_projects.kaggle.essay_scoring.components:tokenize",
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
            "test_sample_ids": test_sample_ids,
        }


@task(persist_result=False)
def split_data(data: dict[str, Any], experiment_id: str, seed: int) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = generate(
            examples,
            "stratified_group_kfold",
            name="essay-score-stratified",
            seed=seed,
            roles=("train", "validate"),
            parameters=SPLIT_PARAMETERS,
        )
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "algorithm": "stratified_group_kfold",
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
