"""Project-owned Prefect DAG for Kaggle Digit Recognizer."""

from __future__ import annotations

from typing import Any

from prefect import flow

from dsio.tracking import resolve_experiment
from reference_projects.kaggle.digit_recognizer.tasks import (
    evaluate_model,
    export,
    infer_and_submit,
    ingest,
    pretrain_encoder,
    split_data,
    train_classifier,
)


@flow(name="kaggle-digit-recognizer-consumer", persist_result=False)
def digit_recognizer_flow(data_dir: str, workspace: str, *, seed: int = 19) -> dict[str, Any]:
    experiment_id = resolve_experiment("dsio-kaggle-digit-recognizer").experiment_id
    data = ingest(data_dir, workspace, experiment_id)
    split = split_data(data, experiment_id, seed)
    pretraining = pretrain_encoder(data, split, experiment_id, seed)
    training = train_classifier(data, split, pretraining, experiment_id, seed)
    exported = export(data, split, training, experiment_id)
    evaluation = evaluate_model(data, split, exported, experiment_id)
    inference = infer_and_submit(data, exported, workspace, experiment_id)
    return {
        "experiment_id": experiment_id,
        "data_run_id": data["data_run_id"],
        "split_run_id": split["split_run_id"],
        "pretrain_run_id": pretraining["pretrain_run_id"],
        "train_run_id": training["train_run_id"],
        "export_run_id": exported["export_run_id"],
        "evaluation_run_id": evaluation["evaluation_run_id"],
        "inference_run_id": inference["inference_run_id"],
        "dataset_digest": data["dataset_digest"],
        "split_digest": split["split_digest"],
        "assignments": split["assignments"],
        "pretraining_label_fields": pretraining["label_fields"],
        "encoder_digest": training["encoder_digest"],
        "encoder_verified": training["encoder_verified"],
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
            "pretrain": pretraining["identity"],
            "train": training["identity"],
            "export": exported["identity"],
            "evaluation": evaluation["identity"],
            "inference": inference["identity"],
        },
    }
