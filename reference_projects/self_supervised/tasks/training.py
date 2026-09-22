"""SSL training through the shared Lightning spine."""

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
from dsio.data.adapters import entity_examples
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
from reference_projects.self_supervised.components import (
    ContrastiveObjective,
    TinyEmbedding,
    unlabelled_samples,
)

_AUGMENTOR: ComponentConfig = {
    "reference": "dsio.model.components:Jitter",
    "parameters": {"sigma": 0.1},
}
_AUGMENTATION_WRAPPER: ComponentConfig = {
    "reference": "dsio.train.augmentation:TwoView",
    "parameters": {"views": ["online", "target"]},
}
_AUGMENTATION: dict[str, Any] = {
    "augmentor": _AUGMENTOR,
    "wrapper": _AUGMENTATION_WRAPPER,
}
_SHUFFLE = {"train": True, "validate": False, "test": False, "predict": False}
_DROP_LAST = {"train": True, "validate": False, "test": False, "predict": False}
_TRAINER = TrainerConfig(
    max_epochs=3,
    accelerator="auto",
    devices=1,
    deterministic=True,
    checkpoint=False,
    log_every_n_steps=1,
    limit_val_batches=0,
)
_COMPONENTS = {
    "augmentation": _AUGMENTATION_WRAPPER["reference"],
    "augmentor": _AUGMENTOR["reference"],
    "data_module": "dsio.data.loading.module:DsioDataModule",
    "dataset_factory": "reference_projects.self_supervised.components:unlabelled_samples",
    "model": "reference_projects.self_supervised.components:TinyEmbedding",
    "module": "dsio.model.module:DsioModule",
    "objective": "reference_projects.self_supervised.components:ContrastiveObjective",
    "optimizer": "torch.optim:SGD",
}
_TRAINING = {
    "augmentation": _AUGMENTATION,
    "batch_size": 4,
    "drop_last": _DROP_LAST,
    "fold": 0,
    "num_workers": 0,
    "objective_parameters": {"temperature": 0.2},
    "optimizer_parameters": {"lr": 0.05},
    "roles": {"train": "train"},
    "shuffle": _SHUFFLE,
    "trainer": _TRAINER.model_dump(mode="json"),
}


@task(persist_result=False)
def train_model(
    data: dict[str, Any],
    split: dict[str, Any],
    parent_run_id: str,
    seed: int,
) -> dict[str, Any]:
    with attempt(parent_run_id) as child:
        check_requested_capabilities(_TRAINER)
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
            shuffle=_SHUFFLE,
            drop_last=_DROP_LAST,
        )
        augmentor = resolve_component(_AUGMENTOR, expected=torch.nn.Module)
        augmentation = resolve_component(
            _AUGMENTATION_WRAPPER,
            augmentor,
            expected=torch.nn.Module,
        )
        module = DsioModule(
            model=TinyEmbedding(),
            objective=ContrastiveObjective(**_TRAINING["objective_parameters"]),
            optimizer_factory=torch.optim.SGD,
            optimizer_parameters=_TRAINING["optimizer_parameters"],
            training_augmentation=augmentation,
            augmentation_seed=seed,
            augmentation_identity=_AUGMENTATION,
        )
        logger = MLFlowLogger(
            experiment_name="dsio-self-supervised-reference",
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=child.info.run_id,
            log_model=False,
        )
        directory = Path(data["store_path"]).parent
        trainer = build_trainer(
            _TRAINER,
            directory,
            logger,
            build_callbacks(_TRAINER, directory, has_validation=False),
        )
        capabilities = resolve_training_capabilities(trainer, requested=_TRAINER)
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
        checkpoint = directory / "ssl-model.ckpt"
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
