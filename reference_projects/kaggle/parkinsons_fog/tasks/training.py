"""Canonical Lightning training for dense Freezing of Gait prediction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
from prefect import task

from dsio.config.components import ComponentConfig, resolve_component
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
from reference_projects.kaggle.parkinsons_fog.components import (
    FEATURE_COUNT,
    FogDetector,
    FogObjective,
    fog_windows,
    pad_windows,
)
from reference_projects.kaggle.parkinsons_fog.tasks.data import labelled_examples

TRAINER = TrainerConfig(
    max_epochs=2,
    accelerator="cpu",
    devices=1,
    deterministic=True,
    checkpoint=False,
    log_every_n_steps=1,
    limit_val_batches=1.0,
    num_sanity_val_steps=0,
)
ROLES = {"train": "train", "validate": "validate"}
SHUFFLE = {"train": True, "validate": False, "test": False, "predict": False}
DROP_LAST = {"train": False, "validate": False, "test": False, "predict": False}
BATCH_SIZE = 16
NUM_WORKERS = 2
OPTIMIZER_PARAMETERS = {"lr": 0.001}
MODEL_PARAMETERS = {"features": FEATURE_COUNT, "hidden": 16}
PREPROCESSOR: ComponentConfig = {"reference": "torch.nn:Identity", "parameters": {}}


@task(persist_result=False)
def train(
    data: dict[str, Any],
    split: dict[str, Any],
    experiment_id: str,
    seed: int,
    trainer_config: TrainerConfig | None = None,
    execution_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        requested = trainer_config or TRAINER
        check_requested_capabilities(requested)
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = load_split_evidence(
            split["split_uri"], examples, consumer_run_id=run.info.run_id
        )
        model_config: ComponentConfig = {
            "reference": "reference_projects.kaggle.parkinsons_fog.components:FogDetector",
            "parameters": MODEL_PARAMETERS,
        }

        def module_factory() -> DsioModule:
            return _module(model_config)

        def data_module_factory(candidate: Mapping[str, Any]) -> DsioDataModule:
            return _data_module(store, examples, manifest, seed, candidate)

        calibration = None
        loader_execution: dict[str, Any] = {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
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
            experiment_name="dsio-kaggle-parkinsons-fog",
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run.info.run_id,
            log_model=False,
        )
        directory = Path(data["store_path"]).parent
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
                "execution_calibration": (
                    None
                    if calibration is None
                    else {
                        "policy": calibration["policy"],
                        "candidates": [
                            trial["candidate"] for trial in calibration["trials"]
                        ],
                        "selected": calibration["selected"],
                        "environment": calibration["environment"],
                        "environment_digest": calibration["environment_digest"],
                    }
                ),
            },
            components={
                "module": "dsio.model.module:DsioModule",
                "data_module": "dsio.data.loading.module:DsioDataModule",
                "dataset_factory": (
                    "reference_projects.kaggle.parkinsons_fog.components:fog_windows"
                ),
                "collator": "reference_projects.kaggle.parkinsons_fog.components:pad_windows",
                "model": model_config,
                "objective": ("reference_projects.kaggle.parkinsons_fog.components:FogObjective"),
                "optimizer": "torch.optim:Adam",
                "preprocessor": PREPROCESSOR,
            },
        )
        log_capabilities(logger, execution)
        root_device = trainer.strategy.root_device
        cuda_device = root_device.index if root_device.type == "cuda" else None
        if requested.accelerator == "cuda" and cuda_device is None:
            raise RuntimeError(
                f"CUDA training resolved an invalid trainer root device: {root_device}"
            )
        telemetry_context = (
            measure_phase("training", cuda_device=cuda_device)
            if trainer_config is not None
            else _unmeasured_phase()
        )
        with telemetry_context as telemetry:
            trainer.fit(module, datamodule=data_module)
        completed_epochs = int(trainer.current_epoch)
        checkpoint = directory / "parkinsons-fog.ckpt"
        trainer.save_checkpoint(checkpoint)
        reference = save_artifact(
            checkpoint.read_bytes(), run_id=run.info.run_id, name="checkpoint"
        )
        MlflowClient().log_dict(
            run.info.run_id, reference.model_dump(mode="json"), "outputs/checkpoint.json"
        )
        if trainer_config is not None:
            elapsed = float(telemetry["elapsed_seconds"])
            labelled_windows = sum(
                len(split["assignments"][role]) for role in ("train", "validate")
            )
            log_phase_evidence(
                run.info.run_id,
                "training",
                telemetry,
                {
                    "epochs": completed_epochs,
                    "requested_epochs": requested.max_epochs,
                    "labelled_windows": labelled_windows,
                    "store_rows": store.n_rows,
                    "window_epochs_per_second": labelled_windows
                    * completed_epochs
                    / elapsed,
                },
            )
        return {
            "train_run_id": run.info.run_id,
            "checkpoint": reference.model_dump(mode="json"),
            "identity": identity,
            "telemetry": telemetry or None,
            "execution_calibration": calibration,
        }


def _unmeasured_phase() -> Any:
    from contextlib import nullcontext

    return nullcontext({})


def _module(model_config: ComponentConfig) -> DsioModule:
    return DsioModule(
        model=resolve_component(model_config, expected=FogDetector),
        objective=FogObjective(),
        optimizer_factory=torch.optim.Adam,
        optimizer_parameters=OPTIMIZER_PARAMETERS,
    )


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
        dataset_factory=fog_windows,
        batch_size=execution["batch_size"],
        num_workers=execution.get("num_workers", 0),
        pin_memory=execution.get("pin_memory", False),
        prefetch_factor=execution.get("prefetch_factor", 2),
        seed=seed,
        shuffle=SHUFFLE,
        drop_last=DROP_LAST,
        collate_fn=pad_windows,
    )
