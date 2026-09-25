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


def _evaluation_attempt(client: MlflowClient, experiment_id: str) -> str:
    return client.create_run(
        experiment_id,
        tags={
            "dsio.prefect.flow_run_id": "flow-run-id",
            "dsio.prefect.task_key": "evaluation-task",
            "dsio.prefect.task_run_id": "task-run-id",
            "dsio.prefect.dynamic_key": "0",
            "dsio.prefect.attempt": "1",
        },
    ).info.run_id


def _isolate_prefect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith("PREFECT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))
    monkeypatch.setenv("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    monkeypatch.setenv("DO_NOT_TRACK", "1")


@pytest.mark.parametrize(
    "tags",
    [
        {},
        {
            "mlflow.parentRunId": "legacy-parent",
            "dsio.prefect.flow_run_id": "flow-run-id",
            "dsio.prefect.task_key": "evaluation-task",
            "dsio.prefect.task_run_id": "task-run-id",
            "dsio.prefect.dynamic_key": "0",
            "dsio.prefect.attempt": "1",
        },
    ],
)
def test_evaluation_rejects_untracked_or_nested_runs(tags: dict[str, str]) -> None:
    client = MlflowClient()
    experiment_id = client.create_experiment("invalid-evaluation-attempt")
    run_id = client.create_run(experiment_id, tags=tags).info.run_id

    with pytest.raises(EvaluationError, match="top-level tracked Prefect attempt"):
        evaluate(
            run_id=run_id,
            model_uri="models:/m-00000000000000000000000000000000",
            dataset_run_id="dataset-run",
            inputs={},
            targets=np.asarray([1.0]),
            metrics=("mae",),
        )


def test_project_owned_task_logs_native_evaluation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_prefect(tmp_path, monkeypatch)
    from prefect import flow, task
    from prefect.testing.utilities import prefect_test_harness

    from dsio.tracking import attempt, resolve_experiment

    model_uri, model_run_id, dataset_run_id = _sources()

    @task
    def evaluation_task(experiment_id: str) -> tuple[str, dict[str, float]]:
        with attempt(experiment_id) as child:
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
    def project_flow() -> tuple[str, dict[str, float]]:
        resolved = resolve_experiment("evaluation-flow")
        return evaluation_task(resolved.experiment_id)

    with prefect_test_harness():
        child_run_id, metrics = project_flow()

    assert metrics == {"mae": 0.0, "rmse": 0.0}
    client = MlflowClient()
    child = client.get_run(child_run_id)
    assert child.info.status == "FINISHED"
    assert child.data.metrics == metrics
    assert "mlflow.parentRunId" not in child.data.tags
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

    from dsio.tracking import attempt, resolve_experiment

    model_uri, _, dataset_run_id = _sources()
    monkeypatch.setattr(
        Trainer,
        "fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("training invoked")),
    )

    @task
    def evaluation_task(experiment_id: str, names: tuple[str, ...]) -> str:
        with attempt(experiment_id) as child:
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
    def downstream(names: tuple[str, ...]) -> str:
        resolved = resolve_experiment("evaluation-rerun")
        return evaluation_task(resolved.experiment_id, names)

    with prefect_test_harness():
        first = downstream(("mae",))
        second = downstream(("rmse",))

    assert first != second
    client = MlflowClient()
    expected_model_id = model_uri.removeprefix("models:/")
    for child_run_id in (first, second):
        assert [item.model_id for item in client.get_run(child_run_id).inputs.model_inputs] == [
            expected_model_id
        ]


def test_invalid_target_fails_before_logging_successful_metrics() -> None:
    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("evaluation-invalid")
    child_run_id = _evaluation_attempt(client, experiment_id)

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
    child_run_id = _evaluation_attempt(client, client.get_run(dataset_run_id).info.experiment_id)

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
    child_run_id = _evaluation_attempt(client, experiment_id)

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
    child_run_id = _evaluation_attempt(client, experiment_id)

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
    child_run_id = _evaluation_attempt(client, experiment_id)
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


def test_unicode_and_bytes_labels_are_not_mixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dsio.eval.execution as execution

    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("evaluation-text-dtype")
    child_run_id = _evaluation_attempt(client, experiment_id)
    monkeypatch.setattr(
        execution,
        "predict",
        lambda *_: {
            "sample_id": np.asarray(["a", "b"]),
            "prediction": np.asarray([b"yes", b"no"]),
        },
    )

    with pytest.raises(EvaluationError, match="dtype.*incompatible"):
        evaluate(
            run_id=child_run_id,
            model_uri=model_uri,
            dataset_run_id=dataset_run_id,
            inputs=_inputs(),
            targets=np.asarray(["yes", "no"]),
            metrics=("accuracy",),
        )

    assert client.get_run(child_run_id).data.metrics == {}


def test_evaluation_requires_a_clean_dedicated_attempt() -> None:
    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("evaluation-collision")
    child_run_id = _evaluation_attempt(client, experiment_id)
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


def test_masked_named_targets_are_scored_per_target_and_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dsio.eval.execution as execution

    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("masked-named-evaluation")
    run_id = _evaluation_attempt(client, experiment_id)
    targets = np.asarray(
        [
            [[1, 0], [0, 1], [1, 1]],
            [[0, 0], [1, 0], [0, 1]],
        ],
        dtype=np.int64,
    )
    mask = np.asarray([[True, False, True], [True, True, False]])
    score = np.asarray(
        [
            [[0.9, 0.1], [0.1, 0.8], [0.8, 0.9]],
            [[0.2, 0.2], [0.7, 0.3], [0.1, 0.8]],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        execution,
        "predict",
        lambda *_: {
            "sample_id": np.asarray(["a", "b"]),
            "prediction": (score >= 0.5).astype(np.int64),
            "score": score,
        },
    )

    values = evaluate(
        run_id=run_id,
        model_uri=model_uri,
        dataset_run_id=dataset_run_id,
        inputs=_inputs(),
        targets=targets,
        metrics=("average_precision", "positive_rate"),
        score_field="score",
        mask=mask,
        target_names=("freeze", "turn"),
    )

    assert values == {
        "average_precision.freeze": 1.0,
        "average_precision.turn": 1.0,
        "average_precision.mean": 1.0,
        "positive_rate.freeze": 0.75,
        "positive_rate.turn": 0.25,
        "positive_rate.mean": 0.5,
    }
    run = client.get_run(run_id)
    assert run.data.params["evaluation.masked"] == "true"
    assert json.loads(run.data.params["evaluation.target_names"]) == ["freeze", "turn"]
    artifact = client.download_artifacts(run_id, "evaluation/predictions.npz", str(tmp_path))
    with np.load(artifact) as evidence:
        assert np.array_equal(evidence["target"], targets)
        assert np.array_equal(evidence["mask"], mask)
        assert evidence["target_names"].tolist() == ["freeze", "turn"]


@pytest.mark.parametrize(
    ("mask", "target_names", "message"),
    [
        (np.ones((2, 3), dtype=np.int64), ("freeze", "turn"), "mask.*boolean"),
        (np.ones((2, 2), dtype=bool), ("freeze", "turn"), "mask.*shape"),
        (np.ones((2, 3), dtype=bool), ("freeze",), "target_names.*last axis"),
        (np.zeros((2, 3), dtype=bool), ("freeze", "turn"), "mask.*at least one"),
        (np.ones((2, 3), dtype=bool), ("freeze", "freeze"), "target_names.*unique"),
        (np.ones((2, 3), dtype=bool), ("mean", "turn"), "cannot use 'mean'"),
        (np.ones((2, 3), dtype=bool), ("freeze.now", "turn"), "start with a letter"),
        (np.ones((2, 3), dtype=bool), ("a" * 245, "turn"), "longer than 250"),
    ],
)
def test_invalid_masked_target_configuration_fails_before_writes(
    mask: np.ndarray[Any, Any],
    target_names: tuple[str, ...],
    message: str,
) -> None:
    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("invalid-masked-evaluation")
    run_id = _evaluation_attempt(client, experiment_id)

    with pytest.raises(EvaluationError, match=message):
        evaluate(
            run_id=run_id,
            model_uri=model_uri,
            dataset_run_id=dataset_run_id,
            inputs=_inputs(),
            targets=np.ones((2, 3, 2), dtype=np.int64),
            metrics=("accuracy",),
            mask=mask,
            target_names=target_names,
        )

    run = client.get_run(run_id)
    assert run.data.metrics == {}
    assert run.inputs.dataset_inputs == []
    assert run.inputs.model_inputs == []
    assert client.list_artifacts(run_id, "evaluation") == []


def test_target_names_respect_mlflow_parameter_and_batch_limits() -> None:
    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("bounded-target-names")
    long_names = tuple(f"Target{index}_{'a' * 210}" for index in range(30))
    too_many_names = tuple(f"Target{index}" for index in range(100))
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1",
        "precision",
        "recall",
        "average_precision",
        "roc_auc",
        "log_loss",
        "positive_rate",
    )

    for names, metrics, message in (
        (long_names, ("accuracy",), "parameter limit"),
        (too_many_names, metric_names, "more than 1000"),
    ):
        run_id = _evaluation_attempt(client, experiment_id)
        with pytest.raises(EvaluationError, match=message):
            evaluate(
                run_id=run_id,
                model_uri=model_uri,
                dataset_run_id=dataset_run_id,
                inputs=_inputs(),
                targets=np.ones((2, 1, len(names)), dtype=np.int64),
                metrics=metrics,
                mask=np.ones((2, 1), dtype=bool),
                target_names=names,
            )
        assert client.list_artifacts(run_id, "evaluation") == []


def test_unnamed_mask_and_unmasked_named_targets_are_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dsio.eval.execution as execution

    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("evaluation-mask-shapes")
    values = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
    monkeypatch.setattr(
        execution,
        "predict",
        lambda *_: {"sample_id": np.asarray(["a", "b"]), "prediction": values},
    )

    masked_run_id = _evaluation_attempt(client, experiment_id)
    assert evaluate(
        run_id=masked_run_id,
        model_uri=model_uri,
        dataset_run_id=dataset_run_id,
        inputs=_inputs(),
        targets=values,
        metrics=("accuracy",),
        mask=np.asarray([[True, False], [True, True]]),
    ) == {"accuracy": 1.0}

    named_run_id = _evaluation_attempt(client, experiment_id)
    assert evaluate(
        run_id=named_run_id,
        model_uri=model_uri,
        dataset_run_id=dataset_run_id,
        inputs=_inputs(),
        targets=values,
        metrics=("accuracy",),
        target_names=("left", "right"),
    ) == {"accuracy.left": 1.0, "accuracy.right": 1.0, "accuracy.mean": 1.0}


def test_named_evaluation_rejects_empty_values_and_mismatched_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dsio.eval.execution as execution

    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("invalid-named-arrays")

    empty = np.empty((2, 0, 2), dtype=np.int64)
    monkeypatch.setattr(
        execution,
        "predict",
        lambda *_: {"sample_id": np.asarray(["a", "b"]), "prediction": empty},
    )
    empty_run_id = _evaluation_attempt(client, experiment_id)
    with pytest.raises(EvaluationError, match="scoreable value"):
        evaluate(
            run_id=empty_run_id,
            model_uri=model_uri,
            dataset_run_id=dataset_run_id,
            inputs=_inputs(),
            targets=empty,
            metrics=("accuracy",),
            target_names=("left", "right"),
        )

    targets = np.ones((2, 3, 2), dtype=np.int64)
    monkeypatch.setattr(
        execution,
        "predict",
        lambda *_: {
            "sample_id": np.asarray(["a", "b"]),
            "prediction": targets,
            "score": np.ones((2, 3, 1), dtype=np.float32),
        },
    )
    score_run_id = _evaluation_attempt(client, experiment_id)
    with pytest.raises(EvaluationError, match="score.*must match named target"):
        evaluate(
            run_id=score_run_id,
            model_uri=model_uri,
            dataset_run_id=dataset_run_id,
            inputs=_inputs(),
            targets=targets,
            metrics=("average_precision",),
            score_field="score",
            mask=np.ones((2, 3), dtype=bool),
            target_names=("left", "right"),
        )


def test_named_metric_failure_identifies_the_target(monkeypatch: pytest.MonkeyPatch) -> None:
    import dsio.eval.execution as execution

    model_uri, _, dataset_run_id = _sources()
    client = MlflowClient()
    experiment_id = client.create_experiment("named-metric-failure")
    run_id = _evaluation_attempt(client, experiment_id)
    targets = np.asarray([[1, 0], [0, 0]], dtype=np.int64)
    scores = np.asarray([[0.9, 0.2], [0.1, 0.1]], dtype=np.float32)
    monkeypatch.setattr(
        execution,
        "predict",
        lambda *_: {
            "sample_id": np.asarray(["a", "b"]),
            "prediction": (scores >= 0.5).astype(np.int64),
            "score": scores,
        },
    )

    with pytest.raises(EvaluationError, match="average_precision.*target 'absent'"):
        evaluate(
            run_id=run_id,
            model_uri=model_uri,
            dataset_run_id=dataset_run_id,
            inputs=_inputs(),
            targets=targets,
            metrics=("average_precision",),
            score_field="score",
            target_names=("present", "absent"),
        )

    assert client.list_artifacts(run_id, "evaluation") == []
