"""Label-free pretraining and verified frozen-encoder classifier training."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import mlflow
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
from prefect import task

from dsio.data.loading import DsioDataModule
from dsio.data.splits.models import SplitFile
from dsio.data.store import SignalStore
from dsio.model.module import DsioModule
from dsio.tracking import attempt, load_split_evidence, record_provenance, require_evidence
from dsio.train.artifacts import ArtifactRef, load_artifact, save_artifact
from dsio.train.capabilities import (
    check_requested_capabilities,
    log_capabilities,
    resolve_training_capabilities,
)
from dsio.train.trainer import TrainerConfig, build_callbacks, build_trainer
from reference_projects.kaggle.digit_recognizer.components import (
    ClassificationObjective,
    DigitAutoencoder,
    FrozenDigitClassifier,
    ReconstructionObjective,
    labelled_digit_samples,
    unlabelled_digit_samples,
)
from reference_projects.kaggle.digit_recognizer.tasks.data import labelled_examples

TRAINER = TrainerConfig(
    max_epochs=1,
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
FOLD = 0
BATCH_SIZE = 4
NUM_WORKERS = 0
PRETRAIN_OPTIMIZER_PARAMETERS = {"lr": 0.001}
CLASSIFIER_OPTIMIZER_PARAMETERS = {"lr": 0.002}


def _data_module(
    store: SignalStore,
    manifest: SplitFile,
    seed: int,
    dataset_factory: Any,
) -> DsioDataModule:
    return DsioDataModule(
        store,
        labelled_examples(store),
        manifest,
        fold=FOLD,
        roles=ROLES,
        dataset_factory=dataset_factory,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=seed,
        shuffle=SHUFFLE,
        drop_last=DROP_LAST,
    )


def _trainer(run_id: str, directory: Path) -> tuple[Any, dict[str, str]]:
    logger = MLFlowLogger(
        experiment_name="dsio-kaggle-digits",
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=run_id,
        log_model=False,
    )
    trainer = build_trainer(
        TRAINER,
        directory,
        logger,
        build_callbacks(TRAINER, directory, has_validation=True),
    )
    return trainer, resolve_training_capabilities(trainer, requested=TRAINER)


@task(persist_result=False)
def pretrain_encoder(
    data: dict[str, Any], split: dict[str, Any], experiment_id: str, seed: int
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        check_requested_capabilities(TRAINER)
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = load_split_evidence(
            split["split_uri"], examples, consumer_run_id=run.info.run_id
        )
        seed_everything(seed, workers=True, verbose=False)
        data_module = _data_module(store, manifest, seed, unlabelled_digit_samples)
        module = DsioModule(
            model=DigitAutoencoder(),
            objective=ReconstructionObjective(),
            optimizer_factory=torch.optim.Adam,
            optimizer_parameters=PRETRAIN_OPTIMIZER_PARAMETERS,
        )
        directory = Path(data["store_path"]).parent
        trainer, execution = _trainer(run.info.run_id, directory)
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "split_digest": split["split_digest"],
                "sample_ids": split["assignments"],
                "label_fields_consumed": [],
                "seed": seed,
                "fold": FOLD,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "optimizer_parameters": PRETRAIN_OPTIMIZER_PARAMETERS,
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
                    "reference_projects.kaggle.digit_recognizer.components:unlabelled_digit_samples"
                ),
                "model": "reference_projects.kaggle.digit_recognizer.components:DigitAutoencoder",
                "objective": (
                    "reference_projects.kaggle.digit_recognizer.components:ReconstructionObjective"
                ),
                "optimizer": "torch.optim:Adam",
            },
        )
        log_capabilities(trainer.logger, execution)
        trainer.fit(module, datamodule=data_module)
        buffer = io.BytesIO()
        torch.save(module.model.encoder.state_dict(), buffer)
        encoder = save_artifact(buffer.getvalue(), run_id=run.info.run_id, name="encoder")
        MlflowClient().log_dict(
            run.info.run_id, encoder.model_dump(mode="json"), "outputs/encoder.json"
        )
        return {
            "pretrain_run_id": run.info.run_id,
            "encoder": encoder.model_dump(mode="json"),
            "identity": identity,
            "dataset_digest": data["dataset_digest"],
            "split_digest": split["split_digest"],
            "label_fields": [],
        }


def _verified_encoder(
    payload: dict[str, Any],
    *,
    identity: str,
    dataset_digest: str,
    split_digest: str,
) -> tuple[ArtifactRef, dict[str, Any]]:
    reference = ArtifactRef.model_validate(payload)
    require_evidence(
        reference.run_id,
        identity=identity,
        required_artifacts={reference.path},
        expected_configuration={
            "dataset_digest": dataset_digest,
            "split_digest": split_digest,
            "label_fields_consumed": [],
        },
    )
    payload_bytes = load_artifact(reference)
    try:
        state = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"encoder evidence cannot be decoded: {error}") from error
    if not isinstance(state, dict) or not state:
        raise ValueError("encoder evidence contains no state")
    return reference, state


def validate_encoder_handoff(
    pretraining: dict[str, Any], data: dict[str, Any], split: dict[str, Any]
) -> None:
    """Reject metadata that does not describe this downstream training input."""
    if pretraining.get("label_fields") != []:
        raise ValueError("encoder evidence is label-tainted and cannot be reused")
    if pretraining.get("dataset_digest") != data["dataset_digest"]:
        raise ValueError("encoder evidence belongs to a different dataset")
    if pretraining.get("split_digest") != split["split_digest"]:
        raise ValueError("encoder evidence belongs to a different split")


@task(persist_result=False)
def train_classifier(
    data: dict[str, Any],
    split: dict[str, Any],
    pretraining: dict[str, Any],
    experiment_id: str,
    seed: int,
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        check_requested_capabilities(TRAINER)
        validate_encoder_handoff(pretraining, data, split)
        encoder, state = _verified_encoder(
            pretraining["encoder"],
            identity=pretraining["identity"],
            dataset_digest=data["dataset_digest"],
            split_digest=split["split_digest"],
        )
        store = SignalStore(data["store_path"])
        examples = labelled_examples(store)
        manifest = load_split_evidence(
            split["split_uri"], examples, consumer_run_id=run.info.run_id
        )
        seed_everything(seed, workers=True, verbose=False)
        model = FrozenDigitClassifier()
        model.encoder.load_state_dict(state)
        model.freeze_encoder()
        data_module = _data_module(store, manifest, seed, labelled_digit_samples)
        module = DsioModule(
            model=model,
            objective=ClassificationObjective(),
            optimizer_factory=torch.optim.Adam,
            optimizer_parameters=CLASSIFIER_OPTIMIZER_PARAMETERS,
        )
        directory = Path(data["store_path"]).parent
        trainer, execution = _trainer(run.info.run_id, directory)
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "split_digest": split["split_digest"],
                "encoder_digest": encoder.digest,
                "encoder_source_identity": pretraining["identity"],
                "encoder_frozen": True,
                "seed": seed,
                "fold": FOLD,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "optimizer_parameters": CLASSIFIER_OPTIMIZER_PARAMETERS,
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
                    "reference_projects.kaggle.digit_recognizer.components:labelled_digit_samples"
                ),
                "model": (
                    "reference_projects.kaggle.digit_recognizer.components:FrozenDigitClassifier"
                ),
                "objective": (
                    "reference_projects.kaggle.digit_recognizer.components:ClassificationObjective"
                ),
                "optimizer": "torch.optim:Adam",
            },
        )
        log_capabilities(trainer.logger, execution)
        trainer.fit(module, datamodule=data_module)
        checkpoint = directory / "digit-classifier.ckpt"
        trainer.save_checkpoint(checkpoint)
        reference = save_artifact(
            checkpoint.read_bytes(), run_id=run.info.run_id, name="checkpoint"
        )
        MlflowClient().log_dict(
            run.info.run_id, reference.model_dump(mode="json"), "outputs/checkpoint.json"
        )
        MlflowClient().log_dict(
            run.info.run_id, encoder.model_dump(mode="json"), "inputs/encoder.json"
        )
        return {
            "train_run_id": run.info.run_id,
            "checkpoint": reference.model_dump(mode="json"),
            "encoder_digest": encoder.digest,
            "encoder_verified": True,
            "identity": identity,
        }
