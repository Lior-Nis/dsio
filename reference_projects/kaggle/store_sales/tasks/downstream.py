"""Store Sales export, RMSLE evaluation, inference, and submission evidence."""

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
from reference_projects.kaggle.store_sales.components import ForecastOutput, validate_forecast
from reference_projects.kaggle.store_sales.data import CONTEXT_DAYS, HORIZON_DAYS
from reference_projects.kaggle.store_sales.tasks.training import TRAINING_FOLD


def _arrays(store_path: str, sample_ids: list[str]) -> dict[str, np.ndarray[Any, Any]]:
    store = SignalStore(store_path)
    return {
        "sample_id": np.asarray(sample_ids, dtype=np.str_),
        "x": np.stack([store.read_sample(value)["data"] for value in sample_ids]).astype(
            np.float32
        ),
    }


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
        ids = list(split["fold_assignments"][TRAINING_FOLD]["validate"])
        inputs = _arrays(data["store_path"], ids)
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
                "normalizer": ("reference_projects.kaggle.store_sales.components:ForecastOutput"),
                "validator": ("reference_projects.kaggle.store_sales.components:validate_forecast"),
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
            normalizer=ForecastOutput(),
            validator=validate_forecast,
            input_example=example,
        )
        info = log_predictor(
            predictor,
            run_id=run.info.run_id,
            input_example=example,
            forms=("pyfunc",),
            name="store-sales",
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
        ids = list(split["fold_assignments"][TRAINING_FOLD]["validate"])
        store = SignalStore(data["store_path"])
        targets = np.asarray(
            [store.read_sample(value)["attrs"]["target"] for value in ids], dtype=np.float32
        )
        inputs = _arrays(data["store_path"], ids)
        context_log_sales = inputs["x"][:, :CONTEXT_DAYS, 0]
        seasonal_log_prediction = np.stack(
            [context_log_sales[:, -7 + offset % 7] for offset in range(HORIZON_DAYS)],
            axis=1,
        )
        log_targets = np.log1p(targets)
        seasonal_naive_rmsle = float(np.sqrt(np.mean((log_targets - seasonal_log_prediction) ** 2)))
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "metrics": [
                    "rmsle=rmse(log1p(target),log_prediction)",
                    "seasonal_naive_rmsle=rmse(log1p(target),weekly_repeat)",
                ],
                "sample_ids": ids,
            },
            components={"evaluation": "dsio.eval.execution:evaluate"},
        )
        values = evaluate(
            run_id=run.info.run_id,
            model_uri=exported["model_uri"],
            dataset_run_id=split["split_run_id"],
            inputs=inputs,
            targets=log_targets,
            metrics=("rmse",),
            prediction_field="log_prediction",
        )
        rmsle = values["rmse"]
        client = MlflowClient()
        client.log_metric(run.info.run_id, "rmsle", rmsle)
        client.log_metric(run.info.run_id, "seasonal_naive_rmsle", seasonal_naive_rmsle)
        return {
            "evaluation_run_id": run.info.run_id,
            "metrics": {
                "rmsle": rmsle,
                "seasonal_naive_rmsle": seasonal_naive_rmsle,
            },
            "identity": identity,
        }


@task(persist_result=False)
def infer_and_submit(
    data: dict[str, Any], exported: dict[str, Any], workspace: str, experiment_id: str
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        sample_ids = list(data["test_sample_ids"])
        outputs = predict(exported["model_uri"], _arrays(data["store_path"], sample_ids))
        forecasts = np.asarray(outputs["prediction"], dtype=float)
        store = SignalStore(data["store_path"])
        by_id: dict[int, float] = {}
        for sample_id, values in zip(sample_ids, forecasts, strict=True):
            submission_ids = store.read_sample(sample_id)["attrs"]["submission_ids"]
            by_id.update(
                {
                    int(identifier): float(value)
                    for identifier, value in zip(submission_ids, values, strict=True)
                }
            )
        ordered = [by_id[int(identifier)] for identifier in data["test_ids"]]
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "sample_ids": sample_ids,
                "submission_ids": data["test_ids"],
            },
            components={"inference": "dsio.inference.loading:predict"},
        )
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["id", "sales"])
        writer.writerows(zip(data["test_ids"], ordered, strict=True))
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
            "forecast_shape": list(forecasts.shape),
            "submission_path": str(path),
            "submission_bytes": payload,
            "submission_digest": submission.digest,
            "identity": identity,
        }
