"""Predictor export, evaluation, inference, and submission evidence."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mlflow import MlflowClient
from prefect import task

from dsio.data.store import SignalStore
from dsio.eval import evaluate
from dsio.inference import build_predictor, log_predictor, predict
from dsio.tracking import attempt, record_provenance
from dsio.train.artifacts import ArtifactRef, save_artifact
from reference_projects.kaggle.digit_recognizer.components import (
    DigitPrediction,
    FrozenDigitClassifier,
    ScalePixels,
    validate_digit_prediction,
)


def _arrays(store_path: str, sample_ids: list[str]) -> dict[str, np.ndarray[Any, Any]]:
    store = SignalStore(store_path)
    return {
        "sample_id": np.asarray(sample_ids, dtype=np.str_),
        "x": np.stack([store.read_sample(value)["data"] for value in sample_ids]).astype(np.uint8),
    }


@task(persist_result=False)
def export(
    data: dict[str, Any], split: dict[str, Any], training: dict[str, Any], experiment_id: str
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        reference = ArtifactRef.model_validate(training["checkpoint"])
        inputs = _arrays(data["store_path"], list(split["assignments"]["validate"]))
        identity = record_provenance(
            run.info.run_id,
            {
                "checkpoint_digest": reference.digest,
                "dataset_digest": data["dataset_digest"],
                "split_digest": split["split_digest"],
                "encoder_digest": training["encoder_digest"],
                "export_form": "pyfunc",
            },
            components={
                "builder": "dsio.inference.predictor:build_predictor",
                "model": (
                    "reference_projects.kaggle.digit_recognizer.components:FrozenDigitClassifier"
                ),
                "preprocessor": (
                    "reference_projects.kaggle.digit_recognizer.components:ScalePixels"
                ),
                "normalizer": (
                    "reference_projects.kaggle.digit_recognizer.components:DigitPrediction"
                ),
                "validator": (
                    "reference_projects.kaggle.digit_recognizer.components:"
                    "validate_digit_prediction"
                ),
            },
        )
        example = {
            "sample_id": inputs["sample_id"].tolist(),
            "x": torch.from_numpy(inputs["x"]),
        }
        model = FrozenDigitClassifier()
        model.freeze_encoder()
        predictor = build_predictor(
            reference,
            model=model,
            preprocessor=ScalePixels(),
            normalizer=DigitPrediction(),
            validator=validate_digit_prediction,
            input_example=example,
        )
        info = log_predictor(
            predictor,
            run_id=run.info.run_id,
            input_example=example,
            forms=("pyfunc",),
            name="digit-recognizer",
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
        ids = list(split["assignments"]["validate"])
        store = SignalStore(data["store_path"])
        targets = np.asarray(
            [int(store.read_sample(value)["attrs"]["target"]) for value in ids], dtype=np.int64
        )
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "metrics": ["accuracy"],
                "sample_ids": ids,
            },
            components={"evaluation": "dsio.eval.execution:evaluate"},
        )
        metrics = evaluate(
            run_id=run.info.run_id,
            model_uri=exported["model_uri"],
            dataset_run_id=split["split_run_id"],
            inputs=_arrays(data["store_path"], ids),
            targets=targets,
            metrics=("accuracy",),
        )
        return {"evaluation_run_id": run.info.run_id, "metrics": metrics, "identity": identity}


@task(persist_result=False)
def infer_and_submit(
    data: dict[str, Any], exported: dict[str, Any], workspace: str, experiment_id: str
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        ids = list(data["test_ids"])
        outputs = predict(exported["model_uri"], _arrays(data["store_path"], ids))
        values = np.asarray(outputs["prediction"], dtype=np.int64).tolist()
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "sample_ids": ids,
            },
            components={"inference": "dsio.inference.loading:predict"},
        )
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["ImageId", "Label"])
        writer.writerows(zip(ids, values, strict=True))
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
            "prediction": values,
            "submission_path": str(path),
            "submission_bytes": payload,
            "submission_digest": submission.digest,
            "identity": identity,
        }
