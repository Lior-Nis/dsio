"""Evaluate immutable MLflow evidence inside a caller-owned task."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from mlflow import MlflowClient
from mlflow.entities import (
    DatasetInput,
    InputTag,
    LoggedModelInput,
    LoggedModelStatus,
    Metric,
    Run,
)

from dsio.eval.metrics import MetricError, compute
from dsio.inference.loading import InferenceError, predict

_LOGGED_MODEL_URI = re.compile(r"models:/(m-[0-9a-f]{32})")
_PARENT_RUN_TAG = "mlflow.parentRunId"


class EvaluationError(ValueError):
    """Evaluation evidence, arrays, metrics, or persistence violated its contract."""


def evaluate(
    *,
    run_id: str,
    model_uri: str,
    dataset_run_id: str,
    inputs: Mapping[str, Any],
    targets: np.ndarray[Any, Any],
    metrics: Sequence[str],
    prediction_field: str = "prediction",
    score_field: str | None = None,
) -> dict[str, float]:
    """Compute and record evaluation evidence on one caller-owned child Run."""
    client = MlflowClient()
    _child_run(client, run_id)
    model_id, model_source_run_id = _model_evidence(client, model_uri)
    dataset_input = _dataset_evidence(client, dataset_run_id)
    names = _metric_names(metrics)
    _field_name(prediction_field, "prediction")
    if score_field is not None:
        _field_name(score_field, "score")
    if not isinstance(targets, np.ndarray):
        raise EvaluationError(
            f"evaluation targets must be a NumPy array, got {type(targets).__name__}"
        )
    if targets.ndim == 0 or targets.shape[0] == 0:
        raise EvaluationError("evaluation targets must contain at least one row")

    try:
        outputs = predict(model_uri, inputs)
    except InferenceError as error:
        raise EvaluationError(f"prediction boundary is incompatible: {error}") from error
    predicted = _output(outputs, prediction_field, "prediction")
    if predicted.shape != targets.shape:
        raise EvaluationError(
            f"evaluation target shape {targets.shape} does not match prediction field "
            f"{prediction_field!r} shape {predicted.shape}"
        )
    score = None if score_field is None else _output(outputs, score_field, "score")
    if score is not None and (score.ndim == 0 or score.shape[0] != targets.shape[0]):
        raise EvaluationError(
            f"evaluation score field {score_field!r} has {score.shape} for "
            f"{targets.shape[0]} target rows"
        )
    try:
        values = compute(names, targets, predicted, score)
    except (MetricError, KeyError, TypeError, ValueError) as error:
        raise EvaluationError(
            f"metric inputs or configuration are incompatible: {error}"
        ) from error
    if any(not math.isfinite(value) for value in values.values()):
        raise EvaluationError("evaluation metrics must all be finite")

    _child_run(client, run_id)
    _model_evidence(client, model_uri)
    _dataset_evidence(client, dataset_run_id)
    linked_dataset = _evaluation_dataset_input(dataset_input, dataset_run_id)
    try:
        with TemporaryDirectory(prefix="dsio-evaluation-") as directory:
            artifact = Path(directory) / "predictions.npz"
            np.savez_compressed(
                artifact,
                target=targets,
                # NumPy accepts arbitrary named arrays; its stub mistakes the third one
                # for the reserved allow_pickle keyword.
                **{f"prediction__{name}": value for name, value in outputs.items()},  # type: ignore[arg-type]
            )
            client.log_artifact(run_id, str(artifact), artifact_path="evaluation")
        client.log_inputs(
            run_id,
            datasets=[linked_dataset],
            models=[LoggedModelInput(model_id)],
        )
        client.set_tag(run_id, "dsio.evaluation.model_source_run_id", model_source_run_id)
        client.set_tag(run_id, "dsio.evaluation.dataset_source_run_id", dataset_run_id)
        client.log_param(run_id, "evaluation.metrics", json.dumps(names))
        client.log_param(run_id, "evaluation.prediction_field", prediction_field)
        if score_field is not None:
            client.log_param(run_id, "evaluation.score_field", score_field)
        _child_run(client, run_id)
        _model_evidence(client, model_uri)
        _dataset_evidence(client, dataset_run_id)
        timestamp = int(time.time() * 1000)
        client.log_batch(
            run_id,
            metrics=[
                Metric(
                    name,
                    value,
                    timestamp,
                    0,
                    dataset_name=dataset_input.dataset.name,
                    dataset_digest=dataset_input.dataset.digest,
                )
                for name, value in values.items()
            ],
        )
    except EvaluationError:
        raise
    except Exception as error:
        raise EvaluationError(
            f"could not persist native MLflow evaluation evidence: {error}"
        ) from error
    return values


def _child_run(client: MlflowClient, run_id: str) -> Run:
    try:
        run = client.get_run(run_id)
    except Exception as error:
        raise EvaluationError(
            f"evaluation child Run {run_id!r} cannot be resolved: {error}"
        ) from error
    if run.info.lifecycle_stage != "active" or run.info.status != "RUNNING":
        raise EvaluationError(f"evaluation child Run {run_id!r} must be active and RUNNING")
    if not run.data.tags.get(_PARENT_RUN_TAG):
        raise EvaluationError(
            f"evaluation Run {run_id!r} must be a tracked child of a project flow"
        )
    return run


def _model_evidence(client: MlflowClient, model_uri: str) -> tuple[str, str]:
    matched = _LOGGED_MODEL_URI.fullmatch(model_uri) if isinstance(model_uri, str) else None
    if matched is None:
        raise EvaluationError("evaluation model must use an immutable MLflow Logged Model URI")
    model_id = matched.group(1)
    try:
        model = client.get_logged_model(model_id)
    except Exception as error:
        raise EvaluationError(
            f"evaluation model {model_id!r} cannot be resolved: {error}"
        ) from error
    if (
        model.status != LoggedModelStatus.READY
        or model.model_uri != model_uri
        or model.model_type != "dsio.predictor"
        or model.tags.get("dsio.export_form") != "pyfunc"
        or not model.source_run_id
    ):
        raise EvaluationError(f"evaluation model {model_id!r} is not READY DSIO PyFunc evidence")
    source = _finished_run(client, model.source_run_id, "model source")
    return model_id, source.info.run_id


def _dataset_evidence(client: MlflowClient, run_id: str) -> DatasetInput:
    source = _finished_run(client, run_id, "dataset source")
    inputs = [] if source.inputs is None else source.inputs.dataset_inputs
    if len(inputs) != 1:
        raise EvaluationError(
            f"evaluation dataset source Run {run_id!r} must contain exactly one native "
            f"dataset input, found {len(inputs)}"
        )
    dataset_input = inputs[0]
    dataset = dataset_input.dataset
    if not dataset.name or not dataset.digest or not dataset.source_type or not dataset.source:
        raise EvaluationError("evaluation dataset input has incomplete native MLflow identity")
    return dataset_input


def _finished_run(client: MlflowClient, run_id: str, role: str) -> Run:
    try:
        run = client.get_run(run_id)
    except Exception as error:
        raise EvaluationError(
            f"evaluation {role} Run {run_id!r} cannot be resolved: {error}"
        ) from error
    if run.info.lifecycle_stage != "active" or run.info.status != "FINISHED":
        raise EvaluationError(f"evaluation {role} Run {run_id!r} must be active and FINISHED")
    return run


def _evaluation_dataset_input(source: DatasetInput, source_run_id: str) -> DatasetInput:
    tags = {tag.key: tag.value for tag in source.tags}
    tags.update(
        {
            "mlflow.data.context": "evaluation",
            "dsio.evaluation.source_run_id": source_run_id,
        }
    )
    return DatasetInput(
        source.dataset,
        [InputTag(key, value) for key, value in sorted(tags.items())],
    )


def _metric_names(metrics: Sequence[str]) -> tuple[str, ...]:
    if isinstance(metrics, str) or not isinstance(metrics, Sequence) or not metrics:
        raise EvaluationError("evaluation requires at least one metric name")
    names = tuple(metrics)
    if any(not isinstance(name, str) or not name for name in names):
        raise EvaluationError("evaluation metric names must be non-empty strings")
    if len(set(names)) != len(names):
        raise EvaluationError("evaluation metric names must be unique")
    return names


def _field_name(name: str, role: str) -> None:
    if not isinstance(name, str) or not name:
        raise EvaluationError(f"evaluation {role} field must be a non-empty string")


def _output(
    outputs: Mapping[str, np.ndarray[Any, Any]], field: str, role: str
) -> np.ndarray[Any, Any]:
    value = outputs.get(field)
    if not isinstance(value, np.ndarray):
        raise EvaluationError(
            f"evaluation {role} field {field!r} is missing or is not a NumPy array"
        )
    return value
