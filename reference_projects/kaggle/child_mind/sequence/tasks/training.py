"""Canonical Lightning training over governed raw CMI windows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
from prefect import task

from dsio.config.components import ComponentConfig, resolve_component
from dsio.data.adapters import SignalExamples
from dsio.data.examples import Examples
from dsio.data.loading import DsioDataModule
from dsio.data.splits.models import SplitFile
from dsio.data.store import SignalStore
from dsio.experimental.telemetry import log_phase_evidence, measure_phase
from dsio.experimental.training import calibrate_training_execution, log_calibration
from dsio.model.module import DsioModule
from dsio.tracking import attempt, load_split_evidence, record_provenance
from dsio.train.artifacts import save_artifact
from dsio.train.capabilities import (
    check_requested_capabilities,
    log_capabilities,
    resolve_training_capabilities,
)
from dsio.train.trainer import TrainerConfig, build_callbacks, build_trainer
from reference_projects.kaggle.child_mind.components import CLASSES
from reference_projects.kaggle.child_mind.data import TABULAR_FEATURES
from reference_projects.kaggle.child_mind.sequence.components import (
    CmiSequenceClassifier,
    CmiSequenceObjective,
    sequence_windows,
)
from reference_projects.kaggle.child_mind.sequence.data import (
    RAW_SENSOR_FEATURES,
    sequence_examples,
)

ROLES = {"train": "train", "validate": "validate"}
SHUFFLE = {"train": True, "validate": False, "test": False, "predict": False}
DROP_LAST = {"train": False, "validate": False, "test": False, "predict": False}
OPTIMIZER_PARAMETERS = {"lr": 0.001, "weight_decay": 0.0001}
PREPROCESSOR: ComponentConfig = {"reference": "torch.nn:Identity", "parameters": {}}


@task(persist_result=False)
def train(
    data: dict[str, Any],
    split: dict[str, Any],
    experiment_id: str,
    seed: int,
    trainer_config: TrainerConfig,
    execution_calibration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        check_requested_capabilities(trainer_config)
        store = SignalStore(data["store_path"])
        examples = sequence_examples(store, window_length=int(data["window_length"]))
        manifest = load_split_evidence(
            split["split_uri"], examples, consumer_run_id=run.info.run_id
        )
        training_ids = list(manifest.fold(0).assignments["train"])
        model_config: ComponentConfig = {
            "reference": (
                "reference_projects.kaggle.child_mind.sequence.components:"
                "CmiSequenceClassifier"
            ),
            "parameters": {
                "window_length": int(data["window_length"]),
                "sensor_features": len(RAW_SENSOR_FEATURES),
                **_tabular_statistics(store, examples, training_ids),
                "hidden": 32,
            },
        }
        objective_config: ComponentConfig = {
            "reference": (
                "reference_projects.kaggle.child_mind.sequence.components:"
                "CmiSequenceObjective"
            ),
            "parameters": {"class_weights": _class_weights(store, examples, training_ids)},
        }

        def module_factory() -> DsioModule:
            return DsioModule(
                model=resolve_component(model_config, expected=CmiSequenceClassifier),
                objective=resolve_component(objective_config, expected=CmiSequenceObjective),
                optimizer_factory=torch.optim.AdamW,
                optimizer_parameters=OPTIMIZER_PARAMETERS,
            )

        def data_module_factory(execution: Mapping[str, Any]) -> DsioDataModule:
            return _data_module(store, examples, manifest, seed, execution)

        requested = trainer_config
        calibration = None
        loader_execution: dict[str, Any] = {
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": 2,
            "accumulate_grad_batches": requested.accumulate_grad_batches,
        }
        if execution_calibration is not None:
            calibration = calibrate_training_execution(
                module_factory,
                data_module_factory,
                seed=seed,
                accelerator=requested.accelerator,
                expected_epochs=requested.max_epochs,
                configuration=execution_calibration,
            )
            loader_execution = dict(calibration["selected"])
            requested = requested.model_copy(
                update={
                    "accumulate_grad_batches": loader_execution["accumulate_grad_batches"]
                }
            )
            log_calibration(run.info.run_id, calibration)

        seed_everything(seed, workers=True, verbose=False)
        data_module = data_module_factory(loader_execution)
        module = module_factory()
        logger = MLFlowLogger(
            experiment_name="dsio-kaggle-child-mind-sequence",
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run.info.run_id,
            log_model=False,
        )
        directory = Path(data["store_path"]).parent / "training"
        directory.mkdir(parents=True, exist_ok=True)
        trainer = build_trainer(
            requested,
            directory,
            logger,
            build_callbacks(requested, directory, has_validation=True),
        )
        execution = resolve_training_capabilities(trainer, requested=requested)
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "split_digest": split["split_digest"],
                "seed": seed,
                "fold": 0,
                "window_length": data["window_length"],
                "batch_size": loader_execution["batch_size"],
                "num_workers": loader_execution.get("num_workers", 0),
                "pin_memory": loader_execution.get("pin_memory", False),
                "prefetch_factor": loader_execution.get("prefetch_factor", 2),
                "optimizer_parameters": OPTIMIZER_PARAMETERS,
                "roles": ROLES,
                "shuffle": SHUFFLE,
                "drop_last": DROP_LAST,
                "trainer": requested.model_dump(mode="json"),
                "execution": execution,
                "execution_calibration": None
                if calibration is None
                else {
                    "policy": calibration["policy"],
                    "selected": calibration["selected"],
                    "environment": calibration["environment"],
                    "environment_digest": calibration["environment_digest"],
                },
            },
            components={
                "module": "dsio.model.module:DsioModule",
                "data_module": "dsio.data.loading.module:DsioDataModule",
                "dataset_factory": (
                    "reference_projects.kaggle.child_mind.sequence.components:"
                    "sequence_windows"
                ),
                "model": model_config,
                "objective": objective_config,
                "optimizer": "torch.optim:AdamW",
                "preprocessor": PREPROCESSOR,
            },
        )
        log_capabilities(logger, execution)
        root_device = trainer.strategy.root_device
        cuda_device = root_device.index if root_device.type == "cuda" else None
        with measure_phase("training", cuda_device=cuda_device) as telemetry:
            trainer.fit(module, datamodule=data_module)
        checkpoint = directory / "child-mind-sequence.ckpt"
        trainer.save_checkpoint(checkpoint)
        reference = save_artifact(
            checkpoint.read_bytes(), run_id=run.info.run_id, name="checkpoint"
        )
        client = MlflowClient()
        client.log_dict(
            run.info.run_id, reference.model_dump(mode="json"), "outputs/checkpoint.json"
        )
        elapsed = float(telemetry["elapsed_seconds"])
        processed_windows = sum(
            len(split["assignments"][role]) for role in ("train", "validate")
        )
        log_phase_evidence(
            run.info.run_id,
            "training",
            telemetry,
            {
                "epochs": int(trainer.current_epoch),
                "requested_epochs": requested.max_epochs,
                "windows": processed_windows,
                "raw_sensor_rows": int(data["raw_sensor_rows"]),
                "window_epochs_per_second": (
                    processed_windows * int(trainer.current_epoch) / elapsed
                ),
            },
        )
        return {
            "train_run_id": run.info.run_id,
            "checkpoint": reference.model_dump(mode="json"),
            "identity": identity,
            "telemetry": telemetry,
            "execution_calibration": calibration,
            "loader_execution": loader_execution,
        }


def _data_module(
    store: SignalStore,
    examples: Examples,
    manifest: SplitFile,
    seed: int,
    execution: Mapping[str, Any],
) -> DsioDataModule:
    return DsioDataModule(
        store,
        examples,
        manifest,
        fold=0,
        roles=ROLES,
        dataset_factory=sequence_windows,
        batch_size=int(execution["batch_size"]),
        num_workers=int(execution.get("num_workers", 0)),
        pin_memory=bool(execution.get("pin_memory", False)),
        prefetch_factor=int(execution.get("prefetch_factor", 2)),
        seed=seed,
        shuffle=SHUFFLE,
        drop_last=DROP_LAST,
    )


def _participant_entities(
    store: SignalStore, examples: SignalExamples, sample_ids: list[str]
) -> list[Any]:
    positions = {
        examples.index.sample_id(position): position for position in range(len(examples))
    }
    codes = {
        int(examples.index.entity_codes[positions[sample_id]]) for sample_id in sample_ids
    }
    return [store.entities[code] for code in sorted(codes)]


def _tabular_statistics(
    store: SignalStore, examples: SignalExamples, sample_ids: list[str]
) -> dict[str, list[float]]:
    entities = _participant_entities(store, examples, sample_ids)
    packed = np.asarray([entity.attrs["tabular"] for entity in entities], dtype=np.float64)
    values = packed[:, :TABULAR_FEATURES]
    mask = packed[:, TABULAR_FEATURES:].astype(bool)
    count = mask.sum(axis=0)
    total = np.where(mask, values, 0.0).sum(axis=0)
    center = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    variance = np.divide(
        np.where(mask, (values - center) ** 2, 0.0).sum(axis=0),
        count,
        out=np.zeros_like(total),
        where=count > 0,
    )
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-6] = 1.0
    return {"tabular_center": center.tolist(), "tabular_scale": scale.tolist()}


def _class_weights(
    store: SignalStore, examples: SignalExamples, sample_ids: list[str]
) -> list[float]:
    counts = np.zeros(CLASSES, dtype=np.int64)
    for entity in _participant_entities(store, examples, sample_ids):
        counts[int(entity.attrs["label"])] += 1
    if bool((counts == 0).any()):
        raise ValueError(
            f"CMI sequence training fold must contain all {CLASSES} classes; "
            f"counts={counts.tolist()}"
        )
    return (counts.sum() / (CLASSES * counts)).tolist()


__all__ = ["train"]
