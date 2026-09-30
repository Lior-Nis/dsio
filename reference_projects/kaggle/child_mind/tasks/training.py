"""Canonical DSIO Lightning training for both CMI configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

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
from reference_projects.kaggle.child_mind.components import (
    CLASSES,
    CmiFusionClassifier,
    CmiObjective,
    cmi_samples,
)
from reference_projects.kaggle.child_mind.data import (
    SENSOR_FEATURES,
    SENSOR_MASK_SLICE,
    SENSOR_VALUE_SLICE,
    TABULAR_FEATURES,
    TABULAR_MASK_SLICE,
    TABULAR_VALUE_SLICE,
)
from reference_projects.kaggle.child_mind.tasks.data import labelled_examples

Mode = Literal["tabular", "fused"]
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
NUM_WORKERS = 0
OPTIMIZER_PARAMETERS = {"lr": 0.003, "weight_decay": 0.0001}
PREPROCESSOR: ComponentConfig = {"reference": "torch.nn:Identity", "parameters": {}}


@task(persist_result=False)
def train(
    data: dict[str, Any],
    split: dict[str, Any],
    experiment_id: str,
    seed: int,
    mode: Mode,
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        if mode not in ("tabular", "fused"):
            raise ValueError(f"unsupported CMI mode {mode!r}")
        check_requested_capabilities(TRAINER)
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = load_split_evidence(
            split["split_uri"], examples, consumer_run_id=run.info.run_id
        )
        training_ids = list(manifest.fold(0).assignments["train"])
        statistics = _statistics(store, training_ids)
        weights = _class_weights(store, training_ids)
        model_config: ComponentConfig = {
            "reference": ("reference_projects.kaggle.child_mind.components:CmiFusionClassifier"),
            "parameters": {
                **statistics,
                "use_sensor": mode == "fused",
                "hidden": 32,
            },
        }
        objective_config: ComponentConfig = {
            "reference": "reference_projects.kaggle.child_mind.components:CmiObjective",
            "parameters": {"class_weights": weights},
        }
        seed_everything(seed, workers=True, verbose=False)
        data_module = DsioDataModule(
            store,
            examples,
            manifest,
            fold=0,
            roles=ROLES,
            dataset_factory=cmi_samples,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            seed=seed,
            shuffle=SHUFFLE,
            drop_last=DROP_LAST,
        )
        module = DsioModule(
            model=resolve_component(model_config, expected=CmiFusionClassifier),
            objective=resolve_component(objective_config, expected=CmiObjective),
            optimizer_factory=torch.optim.AdamW,
            optimizer_parameters=OPTIMIZER_PARAMETERS,
        )
        logger = MLFlowLogger(
            experiment_name="dsio-kaggle-child-mind",
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run.info.run_id,
            log_model=False,
        )
        directory = Path(data["store_path"]).parent / mode
        directory.mkdir(parents=True, exist_ok=True)
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
                "fold": 0,
                "mode": mode,
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
                "dataset_factory": ("reference_projects.kaggle.child_mind.components:cmi_samples"),
                "model": model_config,
                "objective": objective_config,
                "optimizer": "torch.optim:AdamW",
                "preprocessor": PREPROCESSOR,
            },
        )
        log_capabilities(logger, execution)
        trainer.fit(module, datamodule=data_module)
        checkpoint = directory / "child-mind.ckpt"
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
            "mode": mode,
            "identity": identity,
        }


def _statistics(store: SignalStore, sample_ids: list[str]) -> dict[str, list[float]]:
    tabular = _observed_statistics(
        store, sample_ids, TABULAR_VALUE_SLICE, TABULAR_MASK_SLICE, TABULAR_FEATURES
    )
    sensor = _observed_statistics(
        store, sample_ids, SENSOR_VALUE_SLICE, SENSOR_MASK_SLICE, SENSOR_FEATURES
    )
    return {
        "tabular_center": tabular[0],
        "tabular_scale": tabular[1],
        "sensor_center": sensor[0],
        "sensor_scale": sensor[1],
    }


def _observed_statistics(
    store: SignalStore,
    sample_ids: list[str],
    value_slice: slice,
    mask_slice: slice,
    features: int,
) -> tuple[list[float], list[float]]:
    total = np.zeros(features, dtype=np.float64)
    squares = np.zeros(features, dtype=np.float64)
    count = np.zeros(features, dtype=np.int64)
    for sample_id in sample_ids:
        packed = np.asarray(store.read_sample(sample_id)["data"][0], dtype=np.float64)
        values, mask = packed[value_slice], packed[mask_slice].astype(bool)
        total[mask] += values[mask]
        squares[mask] += np.square(values[mask])
        count[mask] += 1
    center = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    variance = np.divide(squares, count, out=np.zeros_like(total), where=count > 0) - center**2
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-6] = 1.0
    return center.tolist(), scale.tolist()


def _class_weights(store: SignalStore, sample_ids: list[str]) -> list[float]:
    counts = np.zeros(CLASSES, dtype=np.int64)
    for sample_id in sample_ids:
        counts[int(store.read_sample(sample_id)["attrs"]["target"])] += 1
    if bool((counts == 0).any()):
        raise ValueError(
            f"CMI training fold must contain all {CLASSES} classes; counts={counts.tolist()}"
        )
    weights = counts.sum() / (CLASSES * counts)
    return weights.tolist()
