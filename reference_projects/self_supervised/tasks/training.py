"""SSL training through the shared Lightning spine."""

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
from dsio.model.components import Jitter
from dsio.model.module import DsioModule
from dsio.tracking import attempt, load_split_evidence, record_provenance
from dsio.train.artifacts import save_artifact
from dsio.train.augmentation import TwoView
from dsio.train.capabilities import log_capabilities, resolve_training_capabilities
from dsio.train.trainer import TrainerConfig
from reference_projects.self_supervised.components import (
    ContrastiveObjective,
    TinyEmbedding,
    unlabelled_samples,
)

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
    "dataset_factory": "reference_projects.self_supervised.components:unlabelled_samples",
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
            dataset_factory=unlabelled_samples,
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
        trainer_config = TrainerConfig(
            max_epochs=_TRAINING["max_epochs"],
            accelerator=_TRAINING["accelerator"],
            devices=_TRAINING["devices"],
            deterministic=_TRAINING["deterministic"],
            checkpoint=False,
            log_every_n_steps=1,
        )
        trainer = Trainer(
            max_epochs=trainer_config.max_epochs,
            accelerator=trainer_config.accelerator,
            devices=trainer_config.devices,
            precision=trainer_config.precision,
            deterministic=trainer_config.deterministic,
            logger=logger,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            limit_val_batches=_TRAINING["limit_val_batches"],
            log_every_n_steps=1,
        )
        capabilities = resolve_training_capabilities(trainer, requested=trainer_config)
        configuration = {
            **_TRAINING,
            "augmentation_seed": seed,
            "dataset_digest": data["dataset_digest"],
            "execution": capabilities,
            "seed": seed,
            "split_digest": split["split_digest"],
        }
        identity = record_provenance(
            child.info.run_id,
            configuration,
            components=_COMPONENTS,
        )
        log_capabilities(logger, capabilities)
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
