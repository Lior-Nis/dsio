"""Bike Sharing predictor export, evaluation, inference, and submission evidence."""

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
from dsio.experimental.data import collate_arrays
from dsio.experimental.inference import RegressionOutput
from dsio.inference import build_predictor, log_predictor, predict, require_checkpoint_lineage
from dsio.tracking import attempt, record_provenance, require_evidence
from dsio.train.artifacts import ArtifactRef, save_artifact
from reference_projects.kaggle.bike_sharing.components import (
    COLLATOR,
    DATASET,
    INPUTS,
    OUTPUT,
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
        standardization = training["standardization"]
        require_evidence(
            training["train_run_id"],
            identity=training["identity"],
            required_artifacts={reference.path, standardization["artifact"]},
            expected_configuration={"standardization": standardization},
        )
        model = resolve_component(
            components["model"],
            mean=standardization["mean"],
            scale=standardization["scale"],
            expected=torch.nn.Module,
        )
        preprocessor = resolve_component(components["preprocessor"], expected=torch.nn.Module)
        inputs = collate_arrays(
            INPUTS,
            SignalStore(data["store_path"]),
            list(split["assignments"]["validate"]),
            collate_fn=resolve_component(COLLATOR),
        )
        output = resolve_component(OUTPUT, expected=RegressionOutput)
        identity = record_provenance(
            run.info.run_id,
            {
                "checkpoint_digest": reference.digest,
                "dataset_digest": data["dataset_digest"],
                "split_digest": split["split_digest"],
                "standardization": standardization,
                "standardization_sample_ids": training["standardization_sample_ids"],
                "export_form": "pyfunc",
            },
            components={
                "builder": "dsio.inference.predictor:build_predictor",
                "collator": COLLATOR,
                "dataset_factory": INPUTS,
                "model": components["model"],
                "preprocessor": components["preprocessor"],
                "output": OUTPUT,
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
            normalizer=output,
            validator=output.validator,
            input_example=example,
        )
        info = log_predictor(
            predictor,
            run_id=run.info.run_id,
            input_example=example,
            forms=("pyfunc",),
            name="bike-sharing",
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
        arrays = collate_arrays(
            DATASET,
            SignalStore(data["store_path"]),
            ids,
            collate_fn=resolve_component(COLLATOR),
        )
        inputs = {"sample_id": arrays["sample_id"], "x": arrays["x"]}
        targets = arrays["y"]
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "metrics": ["rmse"],
                "sample_ids": ids,
            },
            components={
                "collator": COLLATOR,
                "dataset_factory": DATASET,
                "evaluation": "dsio.eval.execution:evaluate",
            },
        )
        metrics = evaluate(
            run_id=run.info.run_id,
            model_uri=exported["model_uri"],
            dataset_run_id=split["split_run_id"],
            inputs=inputs,
            targets=targets,
            metrics=("rmse",),
        )
        return {"evaluation_run_id": run.info.run_id, "metrics": metrics, "identity": identity}


@task(persist_result=False)
def infer_and_submit(
    data: dict[str, Any], exported: dict[str, Any], workspace: str, experiment_id: str
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        ids = list(data["test_ids"])
        outputs = predict(
            exported["model_uri"],
            collate_arrays(
                INPUTS,
                SignalStore(data["store_path"]),
                ids,
                collate_fn=resolve_component(COLLATOR),
            ),
        )
        values = np.asarray(outputs["prediction"], dtype=float).reshape(-1).tolist()
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "sample_ids": ids,
            },
            components={
                "collator": COLLATOR,
                "dataset_factory": INPUTS,
                "inference": "dsio.inference.loading:predict",
            },
        )
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["datetime", "count"])
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
