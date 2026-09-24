"""The native MLflow Experiment boundary for consumer-owned Prefect flows."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


def test_experiment_resolves_a_native_experiment_without_creating_a_run() -> None:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.entities import Experiment

    from dsio.tracking import resolve_experiment

    resolved = resolve_experiment("project-flow")

    assert isinstance(resolved, Experiment)
    assert resolved.name == "project-flow"
    assert MlflowClient().search_runs([resolved.experiment_id]) == []
    assert mlflow.active_run() is None


def test_experiment_reuses_the_native_experiment_without_creating_flow_runs() -> None:
    from mlflow import MlflowClient

    from dsio.tracking import resolve_experiment

    first = resolve_experiment("repeated-flow")
    second = resolve_experiment("repeated-flow")

    assert second.experiment_id == first.experiment_id
    assert MlflowClient().search_runs([first.experiment_id]) == []


def test_experiment_does_not_disturb_an_active_run() -> None:
    import mlflow
    from mlflow import MlflowClient

    from dsio.tracking import resolve_experiment

    with mlflow.start_run() as active:
        resolved = resolve_experiment("independent-experiment")

        assert mlflow.active_run() is active
        assert MlflowClient().get_run(active.info.run_id).info.status == "RUNNING"
        assert MlflowClient().search_runs([resolved.experiment_id]) == []


def test_experiment_resolution_failure_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dsio.tracking import TrackingError, resolve_experiment

    tracking = importlib.import_module("dsio.tracking.experiment")

    class FailingClient:
        def get_experiment_by_name(self, _name: str) -> object:
            raise RuntimeError("tracking store is unavailable")

    monkeypatch.setattr(tracking, "MlflowClient", FailingClient)

    with pytest.raises(
        TrackingError,
        match=r"resolve MLflow experiment 'project-flow'.*tracking store is unavailable",
    ):
        resolve_experiment("project-flow")


def test_concurrent_experiment_creation_returns_the_winning_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS

    from dsio.tracking import resolve_experiment

    tracking = importlib.import_module("dsio.tracking.experiment")
    winner = SimpleNamespace(experiment_id="winner", name="raced-experiment")
    reads = iter([None, winner])

    class RacingClient:
        def get_experiment_by_name(self, _name: str) -> object | None:
            return next(reads)

        def create_experiment(self, _name: str) -> str:
            raise MlflowException("already exists", RESOURCE_ALREADY_EXISTS)

    monkeypatch.setattr(tracking, "MlflowClient", RacingClient)

    assert resolve_experiment("raced-experiment") is winner


def test_experiment_resolution_preserves_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prefect.exceptions import CancelledRun

    from dsio.tracking import resolve_experiment

    tracking = importlib.import_module("dsio.tracking.experiment")
    cancelled = CancelledRun("cancelled during experiment resolution")

    class CancelledClient:
        def get_experiment_by_name(self, _name: str) -> object:
            raise cancelled

    monkeypatch.setattr(tracking, "MlflowClient", CancelledClient)

    with pytest.raises(CancelledRun) as caught:
        resolve_experiment("cancelled-experiment")

    assert caught.value is cancelled
