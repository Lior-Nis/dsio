"""Failure semantics for visible tracked attempt Runs."""

from __future__ import annotations

import concurrent.futures
import importlib
from types import SimpleNamespace

import pytest


def _install_task_context(monkeypatch: pytest.MonkeyPatch) -> None:
    tracking = importlib.import_module("dsio.tracking.attempt")
    context = SimpleNamespace(
        task=SimpleNamespace(name="tracked"),
        task_run=SimpleNamespace(
            flow_run_id="flow-run-id",
            task_key="tracked-key",
            id="task-run-id",
            dynamic_key="0",
            run_count=1,
        ),
    )
    monkeypatch.setattr(tracking.TaskRunContext, "get", staticmethod(lambda: context))


def _experiment_id(name: str) -> str:
    from mlflow import MlflowClient

    return MlflowClient().create_experiment(name)


def test_cancellation_kills_the_attempt_and_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import attempt

    _install_task_context(monkeypatch)
    experiment_id = _experiment_id("cancelled-attempt")
    cancelled = concurrent.futures.CancelledError("cancelled by Prefect")

    with pytest.raises(concurrent.futures.CancelledError) as caught:
        with attempt(experiment_id) as run:
            raise cancelled

    assert caught.value is cancelled
    assert MlflowClient().get_run(run.info.run_id).info.status == "KILLED"


def test_cancellation_after_creation_but_before_yield_kills_the_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import attempt

    _install_task_context(monkeypatch)
    experiment_id = _experiment_id("cancelled-before-yield")
    tracking = importlib.import_module("dsio.tracking.attempt")
    real_create_attempt = tracking._create_attempt
    cancelled = concurrent.futures.CancelledError("cancelled before body entry")

    def cancel_after_create(*args: object, **kwargs: object) -> object:
        real_create_attempt(*args, **kwargs)
        raise cancelled

    monkeypatch.setattr(tracking, "_create_attempt", cancel_after_create)

    with pytest.raises(concurrent.futures.CancelledError) as caught:
        with attempt(experiment_id):
            raise AssertionError("the body must not execute")

    assert caught.value is cancelled
    runs = MlflowClient().search_runs([experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "KILLED"


def test_unknown_experiment_is_rejected_before_a_run_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dsio.tracking import TrackingError, attempt

    _install_task_context(monkeypatch)

    with pytest.raises(TrackingError, match="Could not resolve MLflow Experiment"):
        with attempt("missing-experiment"):
            raise AssertionError("the body must not execute")


def test_deleted_experiment_is_rejected_before_a_run_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, attempt

    _install_task_context(monkeypatch)
    client = MlflowClient()
    experiment_id = _experiment_id("deleted-experiment")
    client.delete_experiment(experiment_id)

    with pytest.raises(TrackingError, match="lifecycle stage"):
        with attempt(experiment_id):
            raise AssertionError("the body must not execute")


def test_success_finalization_cancellation_kills_the_attempt_and_stays_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import attempt

    _install_task_context(monkeypatch)
    experiment_id = _experiment_id("cancelled-finalization")
    tracking = importlib.import_module("dsio.tracking.attempt")
    real_client = MlflowClient()
    cancelled = concurrent.futures.CancelledError("cancelled while finishing")

    class CancelFinishOnceClient:
        def __getattr__(self, name: str) -> object:
            return getattr(real_client, name)

        def set_terminated(self, run_id: str, status: str) -> None:
            if not hasattr(self, "interrupted"):
                self.interrupted = True
                raise cancelled
            real_client.set_terminated(run_id, status)

    monkeypatch.setattr(tracking, "MlflowClient", CancelFinishOnceClient)

    with pytest.raises(concurrent.futures.CancelledError) as caught:
        with attempt(experiment_id) as run:
            pass

    assert caught.value is cancelled
    assert real_client.get_run(run.info.run_id).info.status == "KILLED"


def test_success_fails_closed_when_finished_status_cannot_be_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, attempt

    _install_task_context(monkeypatch)
    experiment_id = _experiment_id("failed-success-finalization")
    tracking = importlib.import_module("dsio.tracking.attempt")
    real_client = MlflowClient()

    class FailFinishOnceClient:
        def __getattr__(self, name: str) -> object:
            return getattr(real_client, name)

        def set_terminated(self, run_id: str, status: str) -> None:
            if not hasattr(self, "interrupted"):
                self.interrupted = True
                raise RuntimeError("tracking response lost")
            real_client.set_terminated(run_id, status)

    monkeypatch.setattr(tracking, "MlflowClient", FailFinishOnceClient)

    with pytest.raises(TrackingError, match="tracking response lost"):
        with attempt(experiment_id) as run:
            pass

    assert real_client.get_run(run.info.run_id).info.status == "FAILED"


def test_cancellation_during_failed_success_recovery_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import attempt

    _install_task_context(monkeypatch)
    experiment_id = _experiment_id("cancelled-success-recovery")
    tracking = importlib.import_module("dsio.tracking.attempt")
    real_client = MlflowClient()
    cancelled = concurrent.futures.CancelledError("cancelled during recovery")

    class CancelRecoveryClient:
        calls = 0

        def __getattr__(self, name: str) -> object:
            return getattr(real_client, name)

        def set_terminated(self, run_id: str, status: str) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("finished write failed")
            if self.calls == 2:
                raise cancelled
            real_client.set_terminated(run_id, status)

    monkeypatch.setattr(tracking, "MlflowClient", CancelRecoveryClient)

    with pytest.raises(concurrent.futures.CancelledError) as caught:
        with attempt(experiment_id) as run:
            pass

    assert caught.value is cancelled
    assert real_client.get_run(run.info.run_id).info.status == "KILLED"


def test_body_failure_survives_an_interrupted_terminal_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import attempt

    _install_task_context(monkeypatch)
    experiment_id = _experiment_id("failed-finalization")
    tracking = importlib.import_module("dsio.tracking.attempt")
    real_client = MlflowClient()

    class FailFinishOnceClient:
        def __getattr__(self, name: str) -> object:
            return getattr(real_client, name)

        def set_terminated(self, run_id: str, status: str) -> None:
            if not hasattr(self, "interrupted"):
                self.interrupted = True
                raise RuntimeError("tracking response lost")
            real_client.set_terminated(run_id, status)

    monkeypatch.setattr(tracking, "MlflowClient", FailFinishOnceClient)
    task_error = ValueError("task failed")

    with pytest.raises(ValueError) as caught:
        with attempt(experiment_id) as run:
            raise task_error

    assert caught.value is task_error
    assert "tracking response lost" in " ".join(task_error.__notes__)
    assert real_client.get_run(run.info.run_id).info.status == "FAILED"


def test_committed_attempt_is_reconciled_when_creation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, attempt

    _install_task_context(monkeypatch)
    experiment_id = _experiment_id("uncertain-attempt-creation")
    tracking = importlib.import_module("dsio.tracking.attempt")
    real_client = MlflowClient()

    class CommitThenFailClient:
        def __getattr__(self, name: str) -> object:
            return getattr(real_client, name)

        def create_run(
            self,
            target_experiment_id: str,
            *,
            run_name: str | None,
            tags: dict[str, str],
        ) -> object:
            real_client.create_run(target_experiment_id, run_name=run_name, tags=tags)
            raise RuntimeError("response lost after commit")

    monkeypatch.setattr(tracking, "MlflowClient", CommitThenFailClient)

    with pytest.raises(TrackingError, match="response lost after commit"):
        with attempt(experiment_id):
            raise AssertionError("the body must not execute")

    runs = real_client.search_runs([experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
