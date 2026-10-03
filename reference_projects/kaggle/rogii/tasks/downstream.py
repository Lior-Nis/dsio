"""ROGII predictor export, masked evaluation, inference, and submission evidence."""

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
from dsio.experimental.data import PadCollator, collate_arrays
from dsio.experimental.inference import RegressionOutput
from dsio.inference import build_predictor, log_predictor, predict, require_checkpoint_lineage
from dsio.tracking import attempt, record_provenance
from dsio.train.artifacts import ArtifactRef, save_artifact
from reference_projects.kaggle.rogii.components import (
    COLLATOR,
    DATASET,
    OUTPUT,
    TARGET_SCALE,
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
        arrays = collate_arrays(
            DATASET,
            SignalStore(data["store_path"]),
            [sample_id],
            collate_fn=resolve_component(COLLATOR, expected=PadCollator),
        )
        inputs = {"sample_id": arrays["sample_id"], "x": arrays["x"]}
        output = resolve_component(OUTPUT, expected=RegressionOutput)
        identity = record_provenance(
            run.info.run_id,
            {
                "checkpoint_digest": reference.digest,
                "dataset_digest": data["dataset_digest"],
                "split_digest": split["split_digest"],
                "export_form": "pyfunc",
                "dynamic_axes": {"x": [1], "prediction": [1]},
            },
            components={
                "builder": "dsio.inference.predictor:build_predictor",
                "model": components["model"],
                "preprocessor": components["preprocessor"],
                "output": OUTPUT,
                "dataset_factory": DATASET,
                "collator": COLLATOR,
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
            name="rogii",
            dynamic_axes={"x": [1], "prediction": [1]},
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
        store = SignalStore(data["store_path"])
        arrays = collate_arrays(
            DATASET,
            store,
            sample_ids,
            collate_fn=resolve_component(COLLATOR, expected=PadCollator),
        )
        inputs = {"sample_id": arrays["sample_id"], "x": arrays["x"]}
        targets = arrays["y"] * TARGET_SCALE
        mask = arrays["mask"]
        baseline_residuals = []
        for index, sample_id in enumerate(sample_ids):
            sample = store.read_sample(sample_id)
            target = targets[index, mask[index]].astype(np.float64)
            baseline = float(sample["attrs"]["last_known_tvt"])
            baseline_residuals.extend((baseline - target).tolist())
        last_value_rmse = float(np.sqrt(np.mean(np.square(baseline_residuals))))
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "metrics": ["rmse", "last_value_rmse"],
                "baseline": "last-known-TVT",
                "sample_ids": sample_ids,
            },
            components={
                "evaluation": "dsio.eval.execution:evaluate",
                "dataset_factory": DATASET,
                "collator": COLLATOR,
            },
        )
        metrics = evaluate(
            run_id=run.info.run_id,
            model_uri=exported["model_uri"],
            dataset_run_id=split["split_run_id"],
            inputs=inputs,
            targets=targets,
            metrics=("rmse",),
            mask=mask,
        )
        client = MlflowClient()
        client.log_metric(run.info.run_id, "last_value_rmse", last_value_rmse)
        metrics["last_value_rmse"] = last_value_rmse
        return {
            "evaluation_run_id": run.info.run_id,
            "metrics": metrics,
            "identity": identity,
        }


@task(persist_result=False)
def infer_and_submit(
    data: dict[str, Any], exported: dict[str, Any], workspace: str, experiment_id: str
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        sample_ids = list(data["test_sample_ids"])
        arrays = collate_arrays(
            DATASET,
            SignalStore(data["store_path"]),
            sample_ids,
            collate_fn=resolve_component(COLLATOR, expected=PadCollator),
        )
        outputs = predict(
            exported["model_uri"],
            {"sample_id": arrays["sample_id"], "x": arrays["x"]},
        )
        matrix = np.asarray(outputs["prediction"], dtype=np.float64)
        values = [
            float(value)
            for index, sample_id in enumerate(sample_ids)
            for value in matrix[index, : data["tail_lengths"][sample_id]]
        ]
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "sample_ids": sample_ids,
                "submission_ids": data["submission_ids"],
            },
            components={
                "inference": "dsio.inference.loading:predict",
                "dataset_factory": DATASET,
                "collator": COLLATOR,
            },
        )
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["id", "tvt"])
        writer.writerows(zip(data["submission_ids"], values, strict=True))
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
            "prediction_count": len(values),
            "submission_path": str(path),
            "submission_bytes": payload,
            "submission_digest": submission.digest,
            "identity": identity,
        }
