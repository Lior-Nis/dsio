"""Project-owned raw-sequence CMI Prefect DAG."""

from __future__ import annotations

from typing import Any

from prefect import flow

from dsio.tracking import resolve_experiment
from dsio.train.trainer import TrainerConfig
from reference_projects.kaggle.child_mind.sequence.tasks import (
    evaluate_model,
    export,
    ingest,
    split_data,
    train,
)
from reference_projects.kaggle.child_mind.sequence.tasks.data import SPLIT_PARAMETERS


@flow(name="kaggle-child-mind-sequence-consumer", persist_result=False)
def child_mind_sequence_flow(
    data_dir: str,
    workspace: str,
    *,
    seed: int = 19,
    window_length: int = 8192,
    max_epochs: int = 2,
    accelerator: str = "auto",
    calibrate: bool = True,
    existing_store_path: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate one full raw-sequence experiment with explicit missing sensors."""
    trainer = TrainerConfig(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        checkpoint=False,
        deterministic=True,
        log_every_n_steps=1,
        limit_val_batches=1.0,
        num_sanity_val_steps=0,
    )
    calibration = (
        {
            "target_effective_batch_size": 128,
            "warmup_effective_batches": 1,
            "measure_effective_batches": 10,
            "max_device_memory_fraction": 0.8,
            "max_host_memory_fraction": 0.5,
        }
        if calibrate
        else None
    )
    experiment_id = resolve_experiment("dsio-kaggle-child-mind-sequence").experiment_id
    data = ingest(data_dir, workspace, experiment_id, window_length, existing_store_path)
    split = split_data(data, experiment_id, seed)
    training = train(data, split, experiment_id, seed, trainer, calibration)
    exported = export(data, split, training, experiment_id)
    evaluation = evaluate_model(
        data,
        split,
        training,
        exported,
        experiment_id,
    )
    return {
        "experiment_id": experiment_id,
        "data_run_id": data["data_run_id"],
        "split_run_id": split["split_run_id"],
        "train_run_id": training["train_run_id"],
        "export_run_id": exported["export_run_id"],
        "evaluation_run_id": evaluation["evaluation_run_id"],
        "dataset_digest": data["dataset_digest"],
        "split_digest": split["split_digest"],
        "split_algorithm": "stratified_group_kfold",
        "split_parameters": SPLIT_PARAMETERS,
        "assignments": split["assignments"],
        "split_groups": split["split_groups"],
        "participant_count": data["participant_count"],
        "sensor_present_participants": data["sensor_present_participants"],
        "sensor_missing_participants": data["sensor_missing_participants"],
        "raw_sensor_rows": data["raw_sensor_rows"],
        "stored_rows": data["stored_rows"],
        "window_count": data["window_count"],
        "window_length": data["window_length"],
        "model_uri": exported["model_uri"],
        "checkpoint_digest": exported["checkpoint_digest"],
        "metrics": evaluation["metrics"],
        "training_telemetry": training["telemetry"],
        "execution_calibration": training["execution_calibration"],
        "identities": {
            "data": data["identity"],
            "split": split["identity"],
            "train": training["identity"],
            "export": exported["identity"],
            "evaluation": evaluation["identity"],
        },
    }


__all__ = ["child_mind_sequence_flow"]
