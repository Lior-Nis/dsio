"""Export raw-sequence predictors and score participant-aggregated predictions."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.entities import LoggedModelInput
from prefect import task

from dsio.config.components import resolve_component
from dsio.data.loading import DsioDataModule
from dsio.data.store import SignalStore
from dsio.eval.metrics import accuracy, quadratic_weighted_kappa
from dsio.inference import build_predictor, log_predictor, require_checkpoint_lineage
from dsio.tracking import attempt, load_split_evidence, record_provenance
from dsio.train.artifacts import ArtifactRef
from reference_projects.kaggle.child_mind.sequence.components import (
    CmiSequencePrediction,
    ablate_sequence_sensor,
    sequence_windows,
    validate_sequence_prediction,
)
from reference_projects.kaggle.child_mind.sequence.data import (
    RAW_SENSOR_FEATURES,
    sequence_examples,
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
        examples = sequence_examples(store, window_length=int(data["window_length"]))
        sample_id = next(iter(split["assignments"]["validate"]))
        item = sequence_windows(store, examples, [sample_id])[0]
        example = {"sample_id": [sample_id], "x": item["x"].unsqueeze(0)}
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
                "normalizer": (
                    "reference_projects.kaggle.child_mind.sequence.components:CmiSequencePrediction"
                ),
                "validator": (
                    "reference_projects.kaggle.child_mind.sequence.components:"
                    "validate_sequence_prediction"
                ),
            },
        )
        predictor = build_predictor(
            reference,
            model=model,
            preprocessor=preprocessor,
            normalizer=CmiSequencePrediction(),
            validator=validate_sequence_prediction,
            input_example=example,
        )
        info = log_predictor(
            predictor,
            run_id=run.info.run_id,
            input_example=example,
            forms=("pyfunc",),
            name="child-mind-sequence",
        )["pyfunc"]
        return {
            "export_run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "checkpoint_digest": reference.digest,
            "identity": identity,
        }


@task(persist_result=False)
def evaluate_model(
    data: dict[str, Any],
    split: dict[str, Any],
    training: dict[str, Any],
    exported: dict[str, Any],
    experiment_id: str,
) -> dict[str, Any]:
    with attempt(experiment_id) as run:
        store = SignalStore(data["store_path"])
        examples = sequence_examples(store, window_length=int(data["window_length"]))
        manifest = load_split_evidence(
            split["split_uri"], examples, consumer_run_id=run.info.run_id
        )
        execution = training["loader_execution"]
        data_module = DsioDataModule(
            store,
            examples,
            manifest,
            fold=0,
            roles={"predict": "validate"},
            dataset_factory=sequence_windows,
            batch_size=int(execution["batch_size"]),
            num_workers=int(execution.get("num_workers", 0)),
            pin_memory=bool(execution.get("pin_memory", False)),
            prefetch_factor=int(execution.get("prefetch_factor", 2)),
            seed=0,
            shuffle={"predict": False},
        )
        data_module.setup("predict")
        predictor = mlflow.pyfunc.load_model(str(exported["model_uri"]))

        probability_sum: dict[str, np.ndarray[Any, Any]] = {}
        ablated_sum: dict[str, np.ndarray[Any, Any]] = {}
        window_count: dict[str, int] = {}
        target: dict[str, int] = {}
        present: dict[str, bool] = {}
        for batch in data_module.predict_dataloader():
            identities = np.asarray(batch["sample_id"], dtype=np.str_)
            participants = list(batch["participant_id"])
            x = batch["x"].numpy()
            output = predictor.predict({"sample_id": identities, "x": x})
            if not np.array_equal(output["sample_id"], identities):
                raise ValueError("CMI sequence export changed sample identity order")
            probabilities = np.asarray(output["probability"])
            ablated_output = predictor.predict(
                {
                    "sample_id": identities,
                    "x": ablate_sequence_sensor(
                        batch["x"],
                        window_length=int(data["window_length"]),
                        sensor_features=len(RAW_SENSOR_FEATURES),
                    ).numpy(),
                }
            )
            if not np.array_equal(ablated_output["sample_id"], identities):
                raise ValueError("CMI sequence ablation changed sample identity order")
            ablated = np.asarray(ablated_output["probability"])
            for index, participant in enumerate(participants):
                participant = str(participant)
                probability_sum.setdefault(participant, np.zeros(4, dtype=np.float64))
                probability_sum[participant] += probabilities[index]
                ablated_sum.setdefault(participant, np.zeros(4, dtype=np.float64))
                ablated_sum[participant] += ablated[index]
                window_count[participant] = window_count.get(participant, 0) + 1
                target[participant] = int(batch["y"][index])
                present[participant] = bool(batch["sensor_present"][index])

        participants = list(probability_sum)
        truth = np.asarray([target[value] for value in participants], dtype=np.int64)
        predicted = np.asarray(
            [
                int(np.argmax(probability_sum[value] / window_count[value]))
                for value in participants
            ],
            dtype=np.int64,
        )
        ablated_prediction = np.asarray(
            [int(np.argmax(ablated_sum[value] / window_count[value])) for value in participants],
            dtype=np.int64,
        )
        sensor_present = np.asarray([present[value] for value in participants], dtype=bool)
        if not bool(sensor_present.any()) or bool(sensor_present.all()):
            raise ValueError("CMI sequence validation requires present and missing sensors")
        metrics = {
            "accuracy": accuracy(truth, predicted, None),
            "quadratic_weighted_kappa": quadratic_weighted_kappa(truth, predicted, None),
            "qwk.sensor_present": quadratic_weighted_kappa(
                truth[sensor_present], predicted[sensor_present], None
            ),
            "qwk.sensor_missing": quadratic_weighted_kappa(
                truth[~sensor_present], predicted[~sensor_present], None
            ),
            "qwk.sensor_ablated": quadratic_weighted_kappa(
                truth[sensor_present], ablated_prediction[sensor_present], None
            ),
        }
        metrics["qwk.sensor_ablation_delta"] = (
            metrics["qwk.sensor_present"] - metrics["qwk.sensor_ablated"]
        )
        identity = record_provenance(
            run.info.run_id,
            {
                "dataset_digest": data["dataset_digest"],
                "checkpoint_digest": exported["checkpoint_digest"],
                "export_identity": exported["identity"],
                "aggregation": "mean-window-probability-per-participant",
                "metrics": list(metrics),
                "participant_ids": participants,
            },
            components={
                "prediction_schema": (
                    "reference_projects.kaggle.child_mind.sequence.components:"
                    "validate_sequence_prediction"
                ),
                "qwk": "dsio.eval.metrics:quadratic_weighted_kappa",
                "ablation": (
                    "reference_projects.kaggle.child_mind.sequence.components:"
                    "ablate_sequence_sensor"
                ),
            },
        )
        client = MlflowClient()
        model_uri = str(exported["model_uri"])
        model_id = model_uri.removeprefix("models:/")
        if model_id == model_uri or not model_id.startswith("m-"):
            raise ValueError("CMI sequence evaluation requires an immutable Logged Model URI")
        client.log_inputs(run.info.run_id, models=[LoggedModelInput(model_id)])
        client.set_tag(
            run.info.run_id,
            "dsio.evaluation.model_source_run_id",
            str(exported["export_run_id"]),
        )
        for name, value in metrics.items():
            client.log_metric(run.info.run_id, name, value)
        client.log_dict(
            run.info.run_id,
            {
                "participant_count": len(participants),
                "participant_ids": participants,
                "target": truth.tolist(),
                "prediction": predicted.tolist(),
                "sensor_ablated_prediction": ablated_prediction.tolist(),
                "sensor_present": sensor_present.tolist(),
                "window_count": [window_count[value] for value in participants],
            },
            "outputs/participant-predictions.json",
        )
        return {
            "evaluation_run_id": run.info.run_id,
            "metrics": metrics,
            "participant_count": len(participants),
            "identity": identity,
        }


__all__ = ["evaluate_model", "export"]
