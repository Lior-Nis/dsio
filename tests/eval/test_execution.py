"""Evaluation composes inside project-owned Prefect tasks and native MLflow Runs."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from mlflow import MlflowClient
from mlflow.entities import Dataset, DatasetInput, InputTag
from torch import Tensor, nn

from dsio.eval import EvaluationError, evaluate
from dsio.inference import Predictor, PredictorError, TensorOutput, log_predictor


class AddOne(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return value + 1


def validate_prediction(output: Mapping[str, Any]) -> None:
    value = output.get("prediction")
    if not isinstance(value, Tensor) or not bool(torch.isfinite(value).all()):
        raise PredictorError("prediction must be a finite tensor")


def _predictor() -> Predictor:
    return Predictor(
        model=nn.Identity(),
        preprocessor=AddOne(),
        normalizer=TensorOutput(),
        validator=validate_prediction,
        checkpoint_uri="runs:/training/checkpoint",
        checkpoint_digest="a" * 64,
    )


def _sources() -> tuple[str, str, str]:
    client = MlflowClient()
    experiment_id = client.create_experiment("evaluation-sources")
    model_run_id = client.create_run(experiment_id).info.run_id
    model_uri = log_predictor(
        _predictor(),
        run_id=model_run_id,
        input_example={
            "sample_id": ["a", "b"],
            "x": torch.tensor([[1.0], [2.0]]),
        },
        forms=("pyfunc",),
    )["pyfunc"].model_uri
    client.set_terminated(model_run_id, "FINISHED")

    dataset_run_id = client.create_run(experiment_id).info.run_id
    dataset = Dataset(
        name="held-out",
        digest="dataset-v1",
        source_type="local",
        source=json.dumps({"uri": "memory://held-out"}),
    )
    client.log_inputs(
        dataset_run_id,
        datasets=[DatasetInput(dataset, [InputTag("mlflow.data.context", "test")])],
    )
    client.set_terminated(dataset_run_id, "FINISHED")
    return model_uri, model_run_id, dataset_run_id


def _inputs() -> dict[str, np.ndarray[Any, Any]]:
    return {
        "sample_id": np.asarray(["a", "b"]),
        "x": np.asarray([[1.0], [2.0]], dtype=np.float32),
    }


def _evaluation_child(client: MlflowClient, experiment_id: str) -> tuple[str, str]:
    parent_run_id = client.create_run(experiment_id).info.run_id
    child_run_id = client.create_run(
        experiment_id,
        tags={"mlflow.parentRunId": parent_run_id},
    ).info.run_id
    return parent_run_id, child_run_id


def _isolate_prefect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith("PREFECT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))
    monkeypatch.setenv("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    monkeypatch.setenv("DO_NOT_TRACK", "1")


def test_project_owned_task_logs_native_evaluation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_prefect(tmp_path, monkeypatch)
    from prefect import flow, task
    from prefect.testing.utilities import prefect_test_harness

    from dsio.tracking import attempt, experiment

    model_uri, model_run_id, dataset_run_id = _sources()

    @task
    def evaluation_task(parent_run_id: str) -> tuple[str, dict[str, float]]:
        with attempt(parent_run_id) as child:
            metrics = evaluate(
                run_id=child.info.run_id,
                model_uri=model_uri,
                dataset_run_id=dataset_run_id,
                inputs=_inputs(),
                targets=np.asarray([[2.0], [3.0]], dtype=np.float32),
                metrics=("mae", "rmse"),
            )
            return child.info.run_id, metrics

    @flow
    def project_flow() -> tuple[str, str, dict[str, float]]:
        with experiment("evaluation-flow") as parent:
            child_run_id, metrics = evaluation_task(parent.info.run_id)
            return parent.info.run_id, child_run_id, metrics

    with prefect_test_harness():
        parent_run_id, child_run_id, metrics = project_flow()

    assert metrics == {"mae": 0.0, "rmse": 0.0}
    client = MlflowClient()
    child = client.get_run(child_run_id)
    assert child.info.status == "FINISHED"
    assert child.data.metrics == metrics
    assert child.data.tags["mlflow.parentRunId"] == parent_run_id
    assert child.data.tags["dsio.evaluation.model_source_run_id"] == model_run_id
    assert child.data.tags["dsio.evaluation.dataset_source_run_id"] == dataset_run_id
    assert [item.model_id for item in child.inputs.model_inputs] == [
        model_uri.removeprefix("models:/")
    ]
    assert len(child.inputs.dataset_inputs) == 1
    assert child.inputs.dataset_inputs[0].dataset.digest == "dataset-v1"
    assert [item.path for item in client.list_artifacts(child_run_id, "evaluation")] == [
        "evaluation/predictions.npz"
    ]


def test_downstream_rerun_reuses_sources_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_prefect(tmp_path, monkeypatch)
    from lightning import Trainer
    from prefect import flow, task
    from prefect.testing.utilities import prefect_test_harness

    from dsio.tracking import attempt, experiment

    model_uri, _, dataset_run_id = _sources()
    monkeypatch.setattr(
        Trainer,
        "fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("training invoked")),
    )

    @task
    def evaluation_task(parent_run_id: str, names: tuple[str, ...]) -> str:
        with attempt(parent_run_id) as child:
            evaluate(
                run_id=child.info.run_id,
                model_uri=model_uri,
                dataset_run_id=dataset_run_id,
                inputs=_inputs(),
                targets=np.asarray([[2.0], [3.0]], dtype=np.float32),
                metrics=names,
            )
            return child.info.run_id

    @flow
    def downstream(names: tuple[str, ...]) -> tuple[str, str]:
        with experiment("evaluation-rerun") as parent:
            return parent.info.run_id, evaluation_task(parent.info.run_id, names)

    with prefect_test_harness():
        first = downstream(("mae",))
        second = downstream(("rmse",))

    assert first[0] != second[0]
    assert first[1] != second[1]
    client = MlflowClient()
    expected_model_id = model_uri.removeprefix("models:/")
    for _, child_run_id in (first, second):
        assert [item.model_id for item in client.get_run(child_run_id).inputs.model_inputs] == [
            expected_model_id
        ]


def test_invalid_target_fails_before_logging_successful_metrics() -> None:
    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("evaluation-invalid")
    _, child_run_id = _evaluation_child(client, experiment_id)

    with pytest.raises(EvaluationError, match="target.*shape"):
        evaluate(
            run_id=child_run_id,
            model_uri=model_uri,
            dataset_run_id=dataset_run_id,
            inputs=_inputs(),
            targets=np.asarray([2.0, 3.0], dtype=np.float32),
            metrics=("mae",),
        )

    assert client.get_run(child_run_id).data.metrics == {}


@pytest.mark.parametrize("source", ["RUNNING", "FAILED"])
def test_invalid_dataset_source_fails_before_metrics(source: str) -> None:
    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    if source == "RUNNING":
        replacement = client.create_run(
            client.get_run(dataset_run_id).info.experiment_id
        ).info.run_id
        dataset_run_id = replacement
    else:
        client.set_terminated(dataset_run_id, "FAILED")
    parent_run_id, child_run_id = _evaluation_child(
        client, client.get_run(dataset_run_id).info.experiment_id
    )

    with pytest.raises(EvaluationError, match="dataset source Run.*FINISHED"):
        evaluate(
            run_id=child_run_id,
            model_uri=model_uri,
            dataset_run_id=dataset_run_id,
            inputs=_inputs(),
            targets=np.asarray([[2.0], [3.0]], dtype=np.float32),
            metrics=("mae",),
        )

    assert client.get_run(child_run_id).data.metrics == {}
    assert client.get_run(parent_run_id).info.status == "RUNNING"


@pytest.mark.parametrize(
    "targets",
    [
        np.asarray([[np.nan], [3.0]], dtype=np.float32),
        np.asarray([[2.0], [3.0]], dtype=object),
        np.asarray([["2"], ["3"]]),
    ],
)
def test_unsafe_target_values_fail_before_writes(targets: np.ndarray[Any, Any]) -> None:
    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("evaluation-unsafe-target")
    _, child_run_id = _evaluation_child(client, experiment_id)

    with pytest.raises(EvaluationError, match="target"):
        evaluate(
            run_id=child_run_id,
            model_uri=model_uri,
            dataset_run_id=dataset_run_id,
            inputs=_inputs(),
            targets=targets,
            metrics=("mae",),
        )

    child = client.get_run(child_run_id)
    assert child.data.metrics == {}
    assert child.inputs.dataset_inputs == []
    assert child.inputs.model_inputs == []


@pytest.mark.parametrize("role", ["prediction_field", "score_field"])
def test_identity_cannot_be_selected_as_a_metric_field(role: str) -> None:
    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("evaluation-identity-field")
    _, child_run_id = _evaluation_child(client, experiment_id)

    arguments = {role: "sample_id"}
    with pytest.raises(EvaluationError, match="non-identity"):
        evaluate(
            run_id=child_run_id,
            model_uri=model_uri,
            dataset_run_id=dataset_run_id,
            inputs=_inputs(),
            targets=np.asarray([[2.0], [3.0]], dtype=np.float32),
            metrics=("mae",),
            **arguments,
        )


def test_incompatible_score_rank_fails_before_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dsio.eval.execution as execution

    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("evaluation-score-rank")
    _, child_run_id = _evaluation_child(client, experiment_id)
    monkeypatch.setattr(
        execution,
        "predict",
        lambda *_: {
            "sample_id": np.asarray(["a", "b"]),
            "prediction": np.asarray([0, 1]),
            "score": np.ones((2, 2, 2), dtype=np.float32),
        },
    )

    with pytest.raises(EvaluationError, match="score.*shape"):
        evaluate(
            run_id=child_run_id,
            model_uri=model_uri,
            dataset_run_id=dataset_run_id,
            inputs=_inputs(),
            targets=np.asarray([0, 1]),
            metrics=("average_precision",),
            score_field="score",
        )

    assert client.get_run(child_run_id).data.metrics == {}


def test_evaluation_requires_a_clean_dedicated_child() -> None:
    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("evaluation-collision")
    _, child_run_id = _evaluation_child(client, experiment_id)
    client.log_param(child_run_id, "evaluation.metrics", '["old"]')

    with pytest.raises(EvaluationError, match="already contains evaluation evidence"):
        evaluate(
            run_id=child_run_id,
            model_uri=model_uri,
            dataset_run_id=dataset_run_id,
            inputs=_inputs(),
            targets=np.asarray([[2.0], [3.0]], dtype=np.float32),
            metrics=("mae",),
        )

    assert client.list_artifacts(child_run_id, "evaluation") == []
