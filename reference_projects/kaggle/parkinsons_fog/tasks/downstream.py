"""Predictor export, masked evaluation, inference, and submission evidence."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mlflow import MlflowClient
from prefect import task

from dsio.config.components import resolve_component
from dsio.data.store import SignalStore
from dsio.eval import evaluate
from dsio.inference import build_predictor, log_predictor, predict, require_checkpoint_lineage
from dsio.tracking import attempt, record_provenance
from dsio.train.artifacts import ArtifactRef, save_artifact
from reference_projects.kaggle.parkinsons_fog.components import (
    FEATURE_COUNT,
    TARGET_COUNT,
    FogOutput,
    normalize_signal,
    validate_fog_prediction,
)
from reference_projects.kaggle.parkinsons_fog.data import TARGETS, WINDOW_SIZE


def _arrays(store_path: str, sample_ids: list[str]) -> dict[str, np.ndarray[Any, Any]]:
    store = SignalStore(store_path)
    values = np.zeros((len(sample_ids), WINDOW_SIZE, FEATURE_COUNT), dtype=np.float32)
    for index, sample_id in enumerate(sample_ids):
        sample = np.asarray(store.read_sample(sample_id)["data"], dtype=np.float32)
        _require_window_size(sample_id, sample)
        values[index, : len(sample)] = normalize_signal(sample[:, :FEATURE_COUNT])
    return {"sample_id": np.asarray(sample_ids, dtype=np.str_), "x": values}


def _targets_and_mask(
    store_path: str, sample_ids: list[str]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    store = SignalStore(store_path)
    targets = np.zeros((len(sample_ids), WINDOW_SIZE, TARGET_COUNT), dtype=np.int64)
    mask = np.zeros((len(sample_ids), WINDOW_SIZE), dtype=bool)
    for index, sample_id in enumerate(sample_ids):
        sample = np.asarray(store.read_sample(sample_id)["data"], dtype=np.float32)
        _require_window_size(sample_id, sample)
        labels = sample[:, FEATURE_COUNT : FEATURE_COUNT + TARGET_COUNT]
        valid = sample[:, -1]
        if not bool(np.isfinite(labels).all()) or not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError(f"FoG sample {sample_id!r} labels must be finite binary values")
        if not bool(np.isfinite(valid).all()) or not bool(((valid == 0) | (valid == 1)).all()):
            raise ValueError(f"FoG sample {sample_id!r} mask must contain only zero or one")
        targets[index, : len(sample)] = labels.astype(np.int64)
        mask[index, : len(sample)] = valid.astype(bool)
    return targets, mask


def _require_window_size(sample_id: str, sample: np.ndarray[Any, Any]) -> None:
    if len(sample) > WINDOW_SIZE:
        raise ValueError(
            f"FoG sample {sample_id!r} has {len(sample)} points; maximum is {WINDOW_SIZE}"
        )


@task(persist_result=False)
def export(
    data: dict[str, Any], split: dict[str, Any], training: dict[str, Any], experiment_id: str
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        reference = ArtifactRef.model_validate(training["checkpoint"])
        components = require_checkpoint_lineage(
            reference,
            training_run_id=training["train_run_id"],
            training_identity=training["identity"],
            dataset_digest=data["dataset_digest"],
            split_digest=split["split_digest"],
            components=("model", "preprocessor"),
        )
        model = resolve_component(components["model"], expected=torch.nn.Module)
        preprocessor = resolve_component(components["preprocessor"], expected=torch.nn.Module)
        sample_id = next(iter(split["assignments"]["validate"]))
        inputs = _arrays(data["store_path"], [sample_id])
        identity = record_provenance(
            run.info.run_id,
            {
                "checkpoint_digest": reference.digest,
                "dataset_digest": data["dataset_digest"],
                "split_digest": split["split_digest"],
                "export_form": "pyfunc",
            },
            components={
                "builder": "dsio.inference.predictor:build_predictor",
                "model": components["model"],
                "preprocessor": components["preprocessor"],
                "normalizer": ("reference_projects.kaggle.parkinsons_fog.components:FogOutput"),
                "validator": (
                    "reference_projects.kaggle.parkinsons_fog.components:validate_fog_prediction"
                ),
            },
        )
        example = {
            "sample_id": inputs["sample_id"].tolist(),
            "x": torch.from_numpy(inputs["x"]),
        }
        predictor = build_predictor(
            reference,
            model=model,
            preprocessor=preprocessor,
            normalizer=FogOutput(),
            validator=validate_fog_prediction,
            input_example=example,
        )
        info = log_predictor(
            predictor,
            run_id=run.info.run_id,
            input_example=example,
            forms=("pyfunc",),
            name="parkinsons-fog",
        )["pyfunc"]
        return {
            "export_run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "checkpoint_digest": reference.digest,
            "identity": identity,
        }


@task(persist_result=False)
def evaluate_model(
    data: dict[str, Any], split: dict[str, Any], exported: dict[str, Any], experiment_id: str
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        sample_ids = list(split["assignments"]["validate"])
        inputs = _arrays(data["store_path"], sample_ids)
        targets, mask = _targets_and_mask(data["store_path"], sample_ids)
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "metrics": ["average_precision", "positive_rate"],
                "sample_ids": sample_ids,
                "scoreable_points": int(mask.sum()),
            },
            components={"evaluation": "dsio.eval.execution:evaluate"},
        )
        metrics = evaluate(
            run_id=run.info.run_id,
            model_uri=exported["model_uri"],
            dataset_run_id=split["split_run_id"],
            inputs=inputs,
            targets=targets,
            metrics=("average_precision", "positive_rate"),
            score_field="probability",
            mask=mask,
            target_names=TARGETS,
        )
        return {"evaluation_run_id": run.info.run_id, "metrics": metrics, "identity": identity}


@task(persist_result=False)
def infer_and_submit(
    data: dict[str, Any], exported: dict[str, Any], workspace: str, experiment_id: str
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        sample_ids = list(data["test_sample_ids"])
        outputs = predict(exported["model_uri"], _arrays(data["store_path"], sample_ids))
        matrix = np.asarray(outputs["probability"], dtype=np.float64)
        store = SignalStore(data["store_path"])
        by_id: dict[str, list[float]] = {}
        for index, sample_id in enumerate(sample_ids):
            sample = store.read_sample(sample_id)
            identifiers = sample["attrs"]["submission_ids"]
            by_id.update(
                {
                    str(identifier): values.tolist()
                    for identifier, values in zip(
                        identifiers, matrix[index, : len(identifiers)], strict=True
                    )
                }
            )
        ordered = [by_id[str(identifier)] for identifier in data["submission_ids"]]
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "sample_ids": sample_ids,
                "submission_ids": data["submission_ids"],
            },
            components={"inference": "dsio.inference.loading:predict"},
        )
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["Id", *TARGETS])
        writer.writerows(
            [identifier, *values]
            for identifier, values in zip(data["submission_ids"], ordered, strict=True)
        )
        payload = buffer.getvalue().encode()
        path = Path(workspace) / run.info.run_id / "submission.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        submission = save_artifact(payload, run_id=run.info.run_id, name="submission")
        MlflowClient().log_dict(
            run.info.run_id, submission.model_dump(mode="json"), "outputs/submission.json"
        )
        return {
            "inference_run_id": run.info.run_id,
            "prediction": ordered,
            "prediction_count": len(ordered),
            "submission_path": str(path),
            "submission_bytes": payload,
            "submission_digest": submission.digest,
            "identity": identity,
        }
