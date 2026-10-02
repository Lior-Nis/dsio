"""Train-only preprocessing and canonical Lightning training for Bike Sharing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
from prefect import task

from dsio.config.components import ComponentConfig, resolve_component
from dsio.data.loading import DsioDataModule
from dsio.data.store import SignalStore
from dsio.experimental.data import StoredItems, fit_standardization, record_fitted
from dsio.experimental.model import Stages, SupervisedObjective
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
    DATASET,
    FEATURES,
    OBJECTIVE,
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
        standardization = fit_standardization(store, manifest, fold=FOLD)
        standardization_artifact = record_fitted(
            run.info.run_id, "standardization", standardization
        )
        model_config: ComponentConfig = {
            "reference": "dsio.experimental.model.compositions:Stages",
            "parameters": {
                "stages": [
                    {
                        "reference": "dsio.experimental.model.standardization:Standardize",
                        "parameters": {
                            "mean": standardization["mean"],
                            "scale": standardization["scale"],
                        },
                    },
                    {
                        "reference": "dsio.experimental.model.compositions:MLP",
                        "parameters": {
                            "input_shape": [1, FEATURES],
                            "output": 1,
                            "output_activation": "softplus",
                        },
                    },
                ]
            },
        }
        seed_everything(seed, workers=True, verbose=False)
        data_module = DsioDataModule(
            store,
            examples,
            manifest,
            fold=FOLD,
            roles=ROLES,
            dataset_factory=resolve_component(DATASET, expected=StoredItems),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            seed=seed,
            shuffle=SHUFFLE,
            drop_last=DROP_LAST,
        )
        module = DsioModule(
            model=resolve_component(model_config, expected=Stages),
            objective=resolve_component(OBJECTIVE, expected=SupervisedObjective),
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
                # Everything the fit used and produced, so provenance matches the artifact.
                "standardization": {"artifact": standardization_artifact, **standardization},
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
                "dataset_factory": DATASET,
                "model": model_config,
                "objective": OBJECTIVE,
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
            "standardization_sample_ids": standardization["sample_ids"],
            "identity": identity,
        }
