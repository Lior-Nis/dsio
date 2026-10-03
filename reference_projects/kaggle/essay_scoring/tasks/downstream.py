"""Essay scorer export, QWK evaluation, inference, and submission evidence."""

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
from dsio.eval.metrics import quadratic_weighted_kappa
from dsio.experimental.data import PadCollator, collate_arrays
from dsio.inference import build_predictor, log_predictor, predict, require_checkpoint_lineage
from dsio.tracking import attempt, record_provenance
from dsio.train.artifacts import ArtifactRef, save_artifact
from reference_projects.kaggle.essay_scoring.components import (
    COLLATOR,
    INPUTS,
    OrdinalPrediction,
    validate_ordinal_prediction,
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
        store = SignalStore(data["store_path"])
        arrays = collate_arrays(
            INPUTS,
            store,
            list(split["assignments"]["validate"]),
            collate_fn=resolve_component(COLLATOR, expected=PadCollator),
        )
        inputs = {"sample_id": arrays["sample_id"], "x": arrays["x"]}
        identity = record_provenance(
            run.info.run_id,
            {
                "checkpoint_digest": reference.digest,
                "dataset_digest": data["dataset_digest"],
                "split_digest": split["split_digest"],
                "export_form": "pyfunc",
                "dynamic_axes": {"x": [1]},
            },
            components={
                "builder": "dsio.inference.predictor:build_predictor",
                "model": components["model"],
                "preprocessor": components["preprocessor"],
                "normalizer": (
                    "reference_projects.kaggle.essay_scoring.components:OrdinalPrediction"
                ),
                "validator": (
                    "reference_projects.kaggle.essay_scoring.components:validate_ordinal_prediction"
                ),
                "dataset_factory": INPUTS,
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
            normalizer=OrdinalPrediction(),
            validator=validate_ordinal_prediction,
            input_example=example,
        )
        info = log_predictor(
            predictor,
            run_id=run.info.run_id,
            input_example=example,
            forms=("pyfunc",),
            name="essay-scoring",
            dynamic_axes={"x": [1]},
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
        targets = np.asarray(
            [int(store.read_sample(value)["attrs"]["target"]) for value in sample_ids],
            dtype=np.int64,
        )
        arrays = collate_arrays(
            INPUTS,
            store,
            sample_ids,
            collate_fn=resolve_component(COLLATOR, expected=PadCollator),
        )
        inputs = {"sample_id": arrays["sample_id"], "x": arrays["x"]}
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "metrics": ["accuracy", "quadratic_weighted_kappa"],
                "sample_ids": sample_ids,
            },
            components={
                "evaluation": "dsio.eval.execution:evaluate",
                "qwk": ("dsio.eval.metrics:quadratic_weighted_kappa"),
                "dataset_factory": INPUTS,
                "collator": COLLATOR,
            },
        )
        metrics = evaluate(
            run_id=run.info.run_id,
            model_uri=exported["model_uri"],
            dataset_run_id=split["split_run_id"],
            inputs=inputs,
            targets=targets,
            metrics=("accuracy",),
        )
        outputs = predict(exported["model_uri"], inputs)
        qwk = quadratic_weighted_kappa(targets, np.asarray(outputs["prediction"]))
        MlflowClient().log_metric(run.info.run_id, "quadratic_weighted_kappa", qwk)
        metrics["quadratic_weighted_kappa"] = qwk
        return {"evaluation_run_id": run.info.run_id, "metrics": metrics, "identity": identity}


@task(persist_result=False)
def infer_and_submit(
    data: dict[str, Any], exported: dict[str, Any], workspace: str, experiment_id: str
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        test_ids = list(data["test_ids"])
        test_sample_ids = list(data["test_sample_ids"])
        arrays = collate_arrays(
            INPUTS,
            SignalStore(data["store_path"]),
            test_sample_ids,
            collate_fn=resolve_component(COLLATOR, expected=PadCollator),
        )
        outputs = predict(
            exported["model_uri"],
            {"sample_id": arrays["sample_id"], "x": arrays["x"]},
        )
        values = np.asarray(outputs["prediction"], dtype=np.int64).reshape(-1).tolist()
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "sample_ids": test_sample_ids,
                "submission_ids": test_ids,
            },
            components={
                "inference": "dsio.inference.loading:predict",
                "dataset_factory": INPUTS,
                "collator": COLLATOR,
            },
        )
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["essay_id", "score"])
        writer.writerows(zip(test_ids, values, strict=True))
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
