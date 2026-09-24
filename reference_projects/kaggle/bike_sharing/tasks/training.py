"""Train-only preprocessing and canonical Lightning training for Bike Sharing."""

from __future__ import annotations

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
from dsio.data.loading import DsioDataModule
from dsio.data.store import SignalStore
from dsio.model.module import DsioModule
from dsio.tracking import attempt, load_split_evidence, record_provenance
from dsio.train.artifacts import save_artifact
from dsio.train.capabilities import (
    check_requested_capabilities,
    log_capabilities,
    resolve_training_capabilities,
)
from dsio.train.trainer import TrainerConfig, build_callbacks, build_trainer
from reference_projects.kaggle.bike_sharing.components import (
    DemandObjective,
    DemandRegressor,
    hourly_samples,
)
from reference_projects.kaggle.bike_sharing.tasks.data import labelled_examples

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
SHUFFLE = {"train": False, "validate": False, "test": False, "predict": False}
DROP_LAST = {"train": False, "validate": False, "test": False, "predict": False}
FOLD = 0
BATCH_SIZE = 4
NUM_WORKERS = 0
OPTIMIZER_PARAMETERS = {"lr": 0.005}
PREPROCESSOR: ComponentConfig = {"reference": "torch.nn:Identity", "parameters": {}}


def _scaler(store: SignalStore, sample_ids: list[str]) -> tuple[list[float], list[float]]:
    values = np.concatenate(
        [store.read_sample(value)["data"] for value in sample_ids], axis=0
    ).astype(np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return mean.tolist(), scale.tolist()


@task(persist_result=False)
def train(
    data: dict[str, Any], split: dict[str, Any], experiment_id: str, seed: int
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        check_requested_capabilities(TRAINER)
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = load_split_evidence(
            split["split_uri"], examples, consumer_run_id=run.info.run_id
        )
        fit_ids = list(manifest.fold(0).assignments["train"])
        mean, scale = _scaler(store, fit_ids)
        model_config: ComponentConfig = {
            "reference": "reference_projects.kaggle.bike_sharing.components:DemandRegressor",
            "parameters": {"mean": mean, "scale": scale},
        }
        seed_everything(seed, workers=True, verbose=False)
        data_module = DsioDataModule(
            store,
            examples,
            manifest,
            fold=FOLD,
            roles=ROLES,
            dataset_factory=hourly_samples,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            seed=seed,
            shuffle=SHUFFLE,
            drop_last=DROP_LAST,
        )
        module = DsioModule(
            model=resolve_component(model_config, expected=DemandRegressor),
            objective=DemandObjective(),
            optimizer_factory=torch.optim.SGD,
            optimizer_parameters=OPTIMIZER_PARAMETERS,
        )
        logger = MLFlowLogger(
            experiment_name="dsio-kaggle-bike-sharing",
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run.info.run_id,
            log_model=False,
        )
        directory = Path(data["store_path"]).parent
        trainer = build_trainer(
            TRAINER,
            directory,
            logger,
            build_callbacks(TRAINER, directory, has_validation=True),
        )
        execution = resolve_training_capabilities(trainer, requested=TRAINER)
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "split_digest": split["split_digest"],
                "seed": seed,
                "scaler": {"fit_ids": fit_ids, "mean": mean, "scale": scale},
                "fold": FOLD,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "optimizer_parameters": OPTIMIZER_PARAMETERS,
                "roles": ROLES,
                "shuffle": SHUFFLE,
                "drop_last": DROP_LAST,
                "trainer": TRAINER.model_dump(mode="json"),
                "execution": execution,
            },
            components={
                "module": "dsio.model.module:DsioModule",
                "data_module": "dsio.data.loading.module:DsioDataModule",
                "dataset_factory": (
                    "reference_projects.kaggle.bike_sharing.components:hourly_samples"
                ),
                "model": model_config,
                "objective": "reference_projects.kaggle.bike_sharing.components:DemandObjective",
                "optimizer": "torch.optim:SGD",
                "preprocessor": PREPROCESSOR,
            },
        )
        log_capabilities(logger, execution)
        trainer.fit(module, datamodule=data_module)
        checkpoint = directory / "bike.ckpt"
        trainer.save_checkpoint(checkpoint)
        reference = save_artifact(
            checkpoint.read_bytes(), run_id=run.info.run_id, name="checkpoint"
        )
        MlflowClient().log_dict(
            run.info.run_id, reference.model_dump(mode="json"), "outputs/checkpoint.json"
        )
        return {
            "train_run_id": run.info.run_id,
            "checkpoint": reference.model_dump(mode="json"),
            "scaler_fit_ids": fit_ids,
            "identity": identity,
        }
