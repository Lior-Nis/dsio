"""Project-owned Prefect DAG for Parkinson's Freezing of Gait."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from prefect import flow

from dsio.tracking import resolve_experiment
from dsio.train.trainer import TrainerConfig
from reference_projects.kaggle.parkinsons_fog.tasks import (
    SPLIT_PARAMETERS,
    evaluate_model,
    export,
    infer_and_submit,
    ingest,
    scan_scale_sources,
    split_data,
    train,
)


@flow(name="kaggle-parkinsons-fog-consumer", persist_result=False)
def parkinsons_fog_flow(
    data_dir: str,
    workspace: str,
    *,
    seed: int = 19,
    labelled_root: str | None = None,
    daily_root: str | None = None,
    trainer_config: TrainerConfig | None = None,
    execution_calibration: Mapping[str, Any] | None = None,
    scan_batch_rows: int = 65_536,
    scan_memory_budget_bytes: int = 4 * 1024**3,
) -> dict[str, Any]:
    """Run the contract flow or an explicit full-directory scale experiment."""
    if (labelled_root is None) != (daily_root is None):
        raise ValueError("labelled_root and daily_root must be provided together")
    if labelled_root is not None and trainer_config is None:
        raise ValueError("scale execution requires an explicit TrainerConfig")
    experiment_id = resolve_experiment("dsio-kaggle-parkinsons-fog").experiment_id
    scale = None
    source_inventory = None
    if labelled_root is not None and daily_root is not None:
        scale = scan_scale_sources(
            data_dir,
            labelled_root,
            daily_root,
            experiment_id,
            batch_rows=scan_batch_rows,
            memory_budget_bytes=scan_memory_budget_bytes,
        )
        source_inventory = scale["inventory"]
    data = ingest(data_dir, workspace, experiment_id, source_inventory)
    split = split_data(data, experiment_id, seed)
    training = train(
        data,
        split,
        experiment_id,
        seed,
        trainer_config,
        execution_calibration,
    )
    exported = export(data, split, training, experiment_id)
    evaluation = evaluate_model(data, split, exported, experiment_id)
    inference = infer_and_submit(data, exported, workspace, experiment_id)
    return {
        "experiment_id": experiment_id,
        "data_run_id": data["data_run_id"],
        "split_run_id": split["split_run_id"],
        "train_run_id": training["train_run_id"],
        "export_run_id": exported["export_run_id"],
        "evaluation_run_id": evaluation["evaluation_run_id"],
        "inference_run_id": inference["inference_run_id"],
        "dataset_digest": data["dataset_digest"],
        "split_digest": split["split_digest"],
        "split_algorithm": "group_shuffle",
        "split_parameters": SPLIT_PARAMETERS,
        "assignments": split["assignments"],
        "split_groups": split["split_groups"],
        "model_uri": exported["model_uri"],
        "checkpoint_digest": exported["checkpoint_digest"],
        "metrics": evaluation["metrics"],
        "ignored_points": data["ignored_points"],
        "ingest_telemetry": data["telemetry"],
        "training_telemetry": training["telemetry"],
        "execution_calibration": training["execution_calibration"],
        "scale": (
            None
            if scale is None
            else {
                "scan_run_id": scale["scan_run_id"],
                "manifest_digest": scale["manifest_digest"],
                "scan": scale["scan"],
                "scan_telemetry": scale["telemetry"],
                "identity": scale["identity"],
            }
        ),
        "prediction": inference["prediction"],
        "prediction_count": inference["prediction_count"],
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
