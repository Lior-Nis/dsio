"""Lightning training and model export tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
from prefect import task

from dsio.data.adapters import entity_examples
from dsio.data.loading import DsioDataModule
from dsio.data.store import SignalStore
from dsio.inference import (
    TensorOutput,
    build_predictor,
    log_predictor,
    validate_tensor_prediction,
)
from dsio.model.module import DsioModule
from dsio.tracking import attempt, load_split_evidence, record_provenance
from dsio.train.artifacts import ArtifactRef, save_artifact
from reference_projects.supervised.components import (
    RegressionObjective,
    TinyRegressor,
    evaluation_arrays,
    regression_samples,
)

_COMPONENTS = {
    "module": "dsio.model.module:DsioModule",
    "data_module": "dsio.data.loading.module:DsioDataModule",
    "dataset_factory": "reference_projects.supervised.components:regression_samples",
    "model": "reference_projects.supervised.components:TinyRegressor",
    "objective": "reference_projects.supervised.components:RegressionObjective",
    "optimizer": "torch.optim:SGD",
}

_TRAINING = {
    "batch_size": 4,
    "fold": 0,
    "max_epochs": 3,
    "optimizer_parameters": {"lr": 0.05},
    "roles": {"train": "train", "validate": "test"},
}


@task(persist_result=False)
def train_model(
    data: dict[str, Any],
    split: dict[str, Any],
    parent_run_id: str,
    seed: int,
) -> dict[str, Any]:
    with attempt(parent_run_id) as child:
        store = SignalStore(data["store_path"])
        examples = entity_examples(store)
        identity = record_provenance(
            child.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "split_digest": split["split_digest"],
                "seed": seed,
                **_TRAINING,
            },
            components=_COMPONENTS,
        )
        manifest = load_split_evidence(
            split["split_uri"],
            examples,
            consumer_run_id=child.info.run_id,
        )
        seed_everything(seed, workers=True, verbose=False)
        data_module = DsioDataModule(
            store,
            examples,
            manifest,
            fold=_TRAINING["fold"],
            roles=_TRAINING["roles"],
            dataset_factory=regression_samples,
            batch_size=_TRAINING["batch_size"],
            num_workers=0,
            seed=seed,
        )
        module = DsioModule(
            model=TinyRegressor(),
            objective=RegressionObjective(),
            optimizer_factory=torch.optim.SGD,
            optimizer_parameters=_TRAINING["optimizer_parameters"],
        )
        logger = MLFlowLogger(
            experiment_name="dsio-supervised-reference",
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=child.info.run_id,
            log_model=False,
        )
        trainer = Trainer(
            max_epochs=_TRAINING["max_epochs"],
            accelerator="cpu",
            devices=1,
            deterministic=True,
            logger=logger,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        )
        trainer.fit(module, datamodule=data_module)
        checkpoint = Path(data["store_path"]).parent / "model.ckpt"
        trainer.save_checkpoint(checkpoint)
        reference = save_artifact(
            checkpoint.read_bytes(),
            run_id=child.info.run_id,
            name="checkpoint",
        )
        MlflowClient().log_dict(
            child.info.run_id,
            reference.model_dump(mode="json"),
            "outputs/checkpoint.json",
        )
        return {
            "train_run_id": child.info.run_id,
            "checkpoint": reference.model_dump(mode="json"),
            "identity": identity,
        }


@task(persist_result=False)
def export_model(
    data: dict[str, Any],
    split: dict[str, Any],
    training: dict[str, Any],
    parent_run_id: str,
) -> dict[str, Any]:
    with attempt(parent_run_id) as child:
        inputs, _ = evaluation_arrays(data["store_path"], split["assignments"]["test"])
        reference = ArtifactRef.model_validate(training["checkpoint"])
        identity = record_provenance(
            child.info.run_id,
            {
                "checkpoint_digest": reference.digest,
                "dataset_digest": data["dataset_digest"],
                "export_form": "pyfunc",
            },
            components={
                "builder": "dsio.inference.predictor:build_predictor",
                "normalizer": "dsio.inference.predictor:TensorOutput",
                "validator": "dsio.inference.predictor:validate_tensor_prediction",
            },
        )
        input_example = {
            "sample_id": inputs["sample_id"].tolist(),
            "x": torch.from_numpy(inputs["x"]),
        }
        predictor = build_predictor(
            reference,
            model=TinyRegressor(),
            preprocessor=None,
            normalizer=TensorOutput(),
            validator=validate_tensor_prediction,
            input_example=input_example,
        )
        info = log_predictor(
            predictor,
            run_id=child.info.run_id,
            input_example=input_example,
            forms=("pyfunc",),
            name="supervised-reference",
        )["pyfunc"]
        return {
            "export_run_id": child.info.run_id,
            "model_uri": info.model_uri,
            "checkpoint_digest": reference.digest,
            "identity": identity,
        }
