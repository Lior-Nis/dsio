"""Evaluation and inference tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
from mlflow import MlflowClient
from mlflow.entities import DatasetInput, InputTag, LoggedModelInput
from prefect import task

from dsio.eval import evaluate
from dsio.inference import predict
from dsio.tracking import attempt, record_provenance
from reference_projects.supervised.components import evaluation_arrays


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
    inputs, targets = evaluation_arrays(store_path, sample_ids)
    with attempt(parent_run_id) as child:
        identity = record_provenance(
            child.info.run_id,
            {
                "dataset_digest": dataset_digest,
                "checkpoint_digest": checkpoint_digest,
                "metrics": list(metrics),
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
    inputs, _ = evaluation_arrays(store_path, sample_ids)
    with attempt(parent_run_id) as child:
        identity = record_provenance(
            child.info.run_id,
            {
                "dataset_digest": dataset_digest,
                "checkpoint_digest": checkpoint_digest,
                "input_sample_ids": sample_ids,
            },
            components={"inference": "dsio.inference.loading:predict"},
        )
        outputs = predict(model_uri, inputs)
        client = MlflowClient()
        dataset_inputs = client.get_run(dataset_run_id).inputs.dataset_inputs
        if len(dataset_inputs) != 1:
            raise ValueError("inference requires one native dataset input")
        source = dataset_inputs[0]
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
            models=[LoggedModelInput(model_uri.removeprefix("models:/"))],
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
        return {
            "inference_run_id": child.info.run_id,
            "sample_id": outputs["sample_id"].tolist(),
            "prediction": np.asarray(outputs["prediction"]),
            "identity": identity,
        }
