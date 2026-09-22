"""SSL-specific training and export tasks on the shared public spine."""

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
from dsio.inference import build_predictor, log_predictor
from dsio.model.components import Jitter
from dsio.model.module import DsioModule
from dsio.tracking import attempt, load_split_evidence, record_provenance
from dsio.train.artifacts import ArtifactRef, save_artifact
from dsio.train.augmentation import TwoView
from reference_projects.self_supervised.components import (
    ContrastiveObjective,
    EmbeddingNorm,
    TinyEmbedding,
    validate_embedding_norm,
)
from reference_projects.supervised.components import evaluation_arrays, regression_samples

_AUGMENTATION = {
    "kind": "two_view",
    "component": {
        "reference": "dsio.model.components:Jitter",
        "parameters": {"sigma": 0.1},
    },
    "views": ["online", "target"],
}
_COMPONENTS = {
    "augmentation": "dsio.train.augmentation:TwoView",
    "augmentor": "dsio.model.components:Jitter",
    "data_module": "dsio.data.loading.module:DsioDataModule",
    "dataset_factory": "reference_projects.supervised.components:regression_samples",
    "model": "reference_projects.self_supervised.components:TinyEmbedding",
    "module": "dsio.model.module:DsioModule",
    "objective": "reference_projects.self_supervised.components:ContrastiveObjective",
    "optimizer": "torch.optim:SGD",
}
_TRAINING = {
    "accelerator": "auto",
    "augmentation": _AUGMENTATION,
    "batch_size": 4,
    "deterministic": True,
    "devices": 1,
    "fold": 0,
    "limit_val_batches": 0,
    "max_epochs": 3,
    "num_workers": 0,
    "objective_parameters": {"temperature": 0.2},
    "optimizer_parameters": {"lr": 0.05},
    "roles": {"train": "train"},
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
        configuration = {
            **_TRAINING,
            "augmentation_seed": seed,
            "dataset_digest": data["dataset_digest"],
            "seed": seed,
            "split_digest": split["split_digest"],
        }
        identity = record_provenance(
            child.info.run_id,
            configuration,
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
            num_workers=_TRAINING["num_workers"],
            seed=seed,
        )
        module = DsioModule(
            model=TinyEmbedding(),
            objective=ContrastiveObjective(**_TRAINING["objective_parameters"]),
            optimizer_factory=torch.optim.SGD,
            optimizer_parameters=_TRAINING["optimizer_parameters"],
            training_augmentation=TwoView(
                Jitter(sigma=0.1),
                views=("online", "target"),
            ),
            augmentation_seed=seed,
            augmentation_identity=_AUGMENTATION,
        )
        logger = MLFlowLogger(
            experiment_name="dsio-self-supervised-reference",
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=child.info.run_id,
            log_model=False,
        )
        trainer = Trainer(
            max_epochs=_TRAINING["max_epochs"],
            accelerator=_TRAINING["accelerator"],
            devices=_TRAINING["devices"],
            deterministic=_TRAINING["deterministic"],
            logger=logger,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            limit_val_batches=_TRAINING["limit_val_batches"],
            log_every_n_steps=1,
        )
        trainer.fit(module, datamodule=data_module)
        checkpoint = Path(data["store_path"]).parent / "ssl-model.ckpt"
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
                "model": "reference_projects.self_supervised.components:TinyEmbedding",
                "normalizer": "reference_projects.self_supervised.components:EmbeddingNorm",
                "validator": (
                    "reference_projects.self_supervised.components:validate_embedding_norm"
                ),
            },
        )
        input_example = {
            "sample_id": inputs["sample_id"].tolist(),
            "x": torch.from_numpy(inputs["x"]),
        }
        predictor = build_predictor(
            reference,
            model=TinyEmbedding(),
            preprocessor=None,
            normalizer=EmbeddingNorm(),
            validator=validate_embedding_norm,
            input_example=input_example,
        )
        info = log_predictor(
            predictor,
            run_id=child.info.run_id,
            input_example=input_example,
            forms=("pyfunc",),
            name="self-supervised-reference",
        )["pyfunc"]
        return {
            "export_run_id": child.info.run_id,
            "model_uri": info.model_uri,
            "checkpoint_digest": reference.digest,
            "identity": identity,
        }
