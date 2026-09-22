"""Ordinary project-owned Prefect flow for self-supervised training."""

from __future__ import annotations

from typing import Any

from prefect import flow

from dsio.tracking import experiment
from reference_projects.self_supervised.tasks import export_model, train_model
from reference_projects.supervised.tasks import (
    build_data,
    evaluate_model,
    infer,
    split_data,
)


@flow(name="synthetic-self-supervised-reference", persist_result=False)
def self_supervised_flow(workspace: str, *, seed: int = 23) -> dict[str, Any]:
    with experiment("dsio-self-supervised-reference") as parent:
        parent_run_id = parent.info.run_id
        data = build_data(workspace, parent_run_id, seed)
        split = split_data(data, parent_run_id, seed)
        training = train_model(data, split, parent_run_id, seed)
        exported = export_model(data, split, training, parent_run_id)
        sample_ids = list(split["assignments"]["test"])
        evaluation = evaluate_model(
            store_path=data["store_path"],
            sample_ids=sample_ids,
            model_uri=exported["model_uri"],
            dataset_run_id=split["split_run_id"],
            dataset_digest=data["dataset_digest"],
            checkpoint_digest=exported["checkpoint_digest"],
            parent_run_id=parent_run_id,
            metrics=("mae", "rmse"),
        )
        inference = infer(
            store_path=data["store_path"],
            sample_ids=sample_ids,
            model_uri=exported["model_uri"],
            dataset_run_id=split["split_run_id"],
            dataset_digest=data["dataset_digest"],
            checkpoint_digest=exported["checkpoint_digest"],
            parent_run_id=parent_run_id,
        )
        return {
            "parent_run_id": parent_run_id,
            "data_run_id": data["data_run_id"],
            "split_run_id": split["split_run_id"],
            "train_run_id": training["train_run_id"],
            "export_run_id": exported["export_run_id"],
            "evaluation_run_id": evaluation["evaluation_run_id"],
            "inference_run_id": inference["inference_run_id"],
            "store_path": data["store_path"],
            "dataset_digest": data["dataset_digest"],
            "split_digest": split["split_digest"],
            "assignments": split["assignments"],
            "test_sample_id": sample_ids,
            "model_uri": exported["model_uri"],
            "checkpoint_digest": exported["checkpoint_digest"],
            "metrics": evaluation["metrics"],
            "prediction": inference["prediction"],
            "inference_sample_id": inference["sample_id"],
            "identities": {
                "data": data["identity"],
                "split": split["identity"],
                "train": training["identity"],
                "export": exported["identity"],
                "evaluation": evaluation["identity"],
                "inference": inference["identity"],
            },
        }
