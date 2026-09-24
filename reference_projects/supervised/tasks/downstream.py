"""Evaluation and inference tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
from mlflow import MlflowClient
from mlflow.entities import DatasetInput, InputTag, LoggedModelInput
from prefect import task

from dsio.data.store import SignalStore
from dsio.eval import evaluate
from dsio.inference import predict
from dsio.tracking import (
    attempt,
    canonical_dataset_digest,
    record_provenance,
    require_evidence,
)
from reference_projects.supervised.components import evaluation_arrays


def _sources(
    *,
    store_path: str,
    dataset_run_id: str,
    dataset_digest: str,
    model_uri: str,
    checkpoint_digest: str,
) -> tuple[DatasetInput, str, str, str]:
    store = SignalStore(store_path)
    store.verify()
    if store.identity != dataset_digest:
        raise ValueError("store identity does not match the declared dataset digest")

    dataset_run = require_evidence(dataset_run_id)
    dataset_inputs = [] if dataset_run.inputs is None else dataset_run.inputs.dataset_inputs
    if len(dataset_inputs) != 1 or canonical_dataset_digest(dataset_inputs[0]) != dataset_digest:
        raise ValueError("dataset evidence does not match the consumed store")

    prefix = "models:/"
    if not model_uri.startswith(prefix):
        raise ValueError("model evidence must use an immutable MLflow Logged Model URI")
    model_id = model_uri.removeprefix(prefix)
    model = MlflowClient().get_logged_model(model_id)
    if (
        model.model_uri != model_uri
        or model.tags.get("dsio.checkpoint_digest") != checkpoint_digest
        or not model.source_run_id
    ):
        raise ValueError("model evidence does not match the declared checkpoint")
    model_run = require_evidence(model.source_run_id)
    return (
        dataset_inputs[0],
        model_id,
        dataset_run.data.params["dsio.execution_identity"],
        model_run.data.params["dsio.execution_identity"],
    )


@task(persist_result=False)
def evaluate_model(
    *,
    store_path: str,
    sample_ids: list[str],
    model_uri: str,
    dataset_run_id: str,
    dataset_digest: str,
    checkpoint_digest: str,
    parent_run_id: str,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    with attempt(parent_run_id) as child:
        _, _, dataset_identity, model_identity = _sources(
            store_path=store_path,
            dataset_run_id=dataset_run_id,
            dataset_digest=dataset_digest,
            model_uri=model_uri,
            checkpoint_digest=checkpoint_digest,
        )
        inputs, targets = evaluation_arrays(store_path, sample_ids)
        identity = record_provenance(
            child.info.run_id,
            {
                "dataset_digest": dataset_digest,
                "checkpoint_digest": checkpoint_digest,
                "dataset_evidence_identity": dataset_identity,
                "metrics": list(metrics),
                "model_evidence_identity": model_identity,
                "sample_ids": sample_ids,
            },
            components={"evaluation": "dsio.eval.execution:evaluate"},
        )
        values = evaluate(
            run_id=child.info.run_id,
            model_uri=model_uri,
            dataset_run_id=dataset_run_id,
            inputs=inputs,
            targets=targets,
            metrics=metrics,
        )
        _sources(
            store_path=store_path,
            dataset_run_id=dataset_run_id,
            dataset_digest=dataset_digest,
            model_uri=model_uri,
            checkpoint_digest=checkpoint_digest,
        )
        return {
            "evaluation_run_id": child.info.run_id,
            "metrics": values,
            "identity": identity,
        }


@task(persist_result=False)
def infer(
    *,
    store_path: str,
    sample_ids: list[str],
    model_uri: str,
    dataset_run_id: str,
    dataset_digest: str,
    checkpoint_digest: str,
    parent_run_id: str,
) -> dict[str, Any]:
    with attempt(parent_run_id) as child:
        source, model_id, dataset_identity, model_identity = _sources(
            store_path=store_path,
            dataset_run_id=dataset_run_id,
            dataset_digest=dataset_digest,
            model_uri=model_uri,
            checkpoint_digest=checkpoint_digest,
        )
        inputs, _ = evaluation_arrays(store_path, sample_ids)
        identity = record_provenance(
            child.info.run_id,
            {
                "dataset_digest": dataset_digest,
                "checkpoint_digest": checkpoint_digest,
                "dataset_evidence_identity": dataset_identity,
                "input_sample_ids": sample_ids,
                "model_evidence_identity": model_identity,
            },
            components={"inference": "dsio.inference.loading:predict"},
        )
        outputs = predict(model_uri, inputs)
        client = MlflowClient()
        tags = {tag.key: tag.value for tag in source.tags}
        tags["mlflow.data.context"] = "inference"
        tags["dsio.inference.source_run_id"] = dataset_run_id
        client.log_inputs(
            child.info.run_id,
            datasets=[
                DatasetInput(
                    source.dataset,
                    [InputTag(key, value) for key, value in sorted(tags.items())],
                )
            ],
            models=[LoggedModelInput(model_id)],
        )
        client.log_dict(
            child.info.run_id,
            {
                "sample_id": outputs["sample_id"].tolist(),
                "prediction": outputs["prediction"].tolist(),
                "model_uri": model_uri,
            },
            "outputs/inference.json",
        )
        _sources(
            store_path=store_path,
            dataset_run_id=dataset_run_id,
            dataset_digest=dataset_digest,
            model_uri=model_uri,
            checkpoint_digest=checkpoint_digest,
        )
        return {
            "inference_run_id": child.info.run_id,
            "sample_id": outputs["sample_id"].tolist(),
            "prediction": np.asarray(outputs["prediction"]),
            "identity": identity,
        }
