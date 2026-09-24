"""Ordinary project-owned Prefect flows for full training and downstream reevaluation."""

from __future__ import annotations

from typing import Any

from prefect import flow

from dsio.tracking import resolve_experiment
from reference_projects.supervised.tasks import (
    build_data,
    evaluate_model,
    export_model,
    infer,
    split_data,
    train_model,
)


@flow(name="synthetic-supervised-reference", persist_result=False)
def supervised_flow(workspace: str, *, seed: int = 19) -> dict[str, Any]:
    experiment_id = resolve_experiment("dsio-supervised-reference").experiment_id
    data = build_data(workspace, experiment_id, seed)
    split = split_data(data, experiment_id, seed)
    training = train_model(data, split, experiment_id, seed)
    exported = export_model(data, split, training, experiment_id)
    sample_ids = list(split["assignments"]["test"])
    evaluation = evaluate_model(
        store_path=data["store_path"],
        sample_ids=sample_ids,
        model_uri=exported["model_uri"],
        dataset_run_id=split["split_run_id"],
        dataset_digest=data["dataset_digest"],
        checkpoint_digest=exported["checkpoint_digest"],
        experiment_id=experiment_id,
        metrics=("mae", "rmse"),
    )
    inference = infer(
        store_path=data["store_path"],
        sample_ids=sample_ids,
        model_uri=exported["model_uri"],
        dataset_run_id=split["split_run_id"],
        dataset_digest=data["dataset_digest"],
        checkpoint_digest=exported["checkpoint_digest"],
        experiment_id=experiment_id,
    )
    return {
        "experiment_id": experiment_id,
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


@flow(name="synthetic-supervised-reevaluation", persist_result=False)
def reevaluate(sources: dict[str, Any], *, metrics: tuple[str, ...]) -> dict[str, Any]:
    experiment_id = resolve_experiment("dsio-supervised-reference").experiment_id
    result = evaluate_model(
        store_path=sources["store_path"],
        sample_ids=list(sources["test_sample_id"]),
        model_uri=sources["model_uri"],
        dataset_run_id=sources["split_run_id"],
        dataset_digest=sources["dataset_digest"],
        checkpoint_digest=sources["checkpoint_digest"],
        experiment_id=experiment_id,
        metrics=metrics,
    )
    return {
        "experiment_id": experiment_id,
        "evaluation_run_id": result["evaluation_run_id"],
        "metrics": result["metrics"],
        "identity": result["identity"],
    }
