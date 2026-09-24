"""Project-owned Prefect DAG for Kaggle Bike Sharing Demand."""

from __future__ import annotations

from typing import Any

from prefect import flow

from dsio.tracking import experiment
from reference_projects.kaggle.bike_sharing.tasks import (
    SPLIT_PARAMETERS,
    evaluate_model,
    export,
    infer_and_submit,
    ingest,
    split_data,
    train,
)


@flow(name="kaggle-bike-sharing-consumer", persist_result=False)
def bike_sharing_flow(data_dir: str, workspace: str, *, seed: int = 19) -> dict[str, Any]:
    with experiment("dsio-kaggle-bike-sharing") as parent:
        parent_id = parent.info.run_id
        data = ingest(data_dir, workspace, parent_id)
        split = split_data(data, parent_id, seed)
        training = train(data, split, parent_id, seed)
        exported = export(data, split, training, parent_id)
        evaluation = evaluate_model(data, split, exported, parent_id)
        inference = infer_and_submit(data, exported, workspace, parent_id)
        return {
            "parent_run_id": parent_id,
            "data_run_id": data["data_run_id"],
            "split_run_id": split["split_run_id"],
            "train_run_id": training["train_run_id"],
            "export_run_id": exported["export_run_id"],
            "evaluation_run_id": evaluation["evaluation_run_id"],
            "inference_run_id": inference["inference_run_id"],
            "dataset_digest": data["dataset_digest"],
            "split_digest": split["split_digest"],
            "assignments": split["assignments"],
            "split_algorithm": "purged_walk_forward",
            "split_parameters": SPLIT_PARAMETERS,
            "discarded_count": split["discarded_count"],
            "train_times": split["train_times"],
            "validate_times": split["validate_times"],
            "scaler_fit_ids": training["scaler_fit_ids"],
            "model_uri": exported["model_uri"],
            "checkpoint_digest": exported["checkpoint_digest"],
            "metrics": evaluation["metrics"],
            "prediction": inference["prediction"],
            "submission_path": inference["submission_path"],
            "submission_bytes": inference["submission_bytes"],
            "submission_digest": inference["submission_digest"],
            "identities": {
                "data": data["identity"],
                "split": split["identity"],
                "train": training["identity"],
                "export": exported["identity"],
                "evaluation": evaluation["identity"],
                "inference": inference["identity"],
            },
        }
