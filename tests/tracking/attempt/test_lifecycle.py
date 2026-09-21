"""Failure semantics for tracked child Runs."""

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
            task_key="tracked-key",
            id="task-run-id",
            dynamic_key="0",
            run_count=1,
        ),
    )
    monkeypatch.setattr(tracking.TaskRunContext, "get", staticmethod(lambda: context))


def _running_parent(name: str) -> object:
    from mlflow import MlflowClient

    client = MlflowClient()
    experiment_id = client.create_experiment(name)
    return client.create_run(experiment_id)


def test_cancellation_kills_the_child_and_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import attempt

    _install_task_context(monkeypatch)
    parent = _running_parent("cancelled-child")
    cancelled = concurrent.futures.CancelledError("cancelled by Prefect")

    with pytest.raises(concurrent.futures.CancelledError) as caught:
        with attempt(parent.info.run_id) as child:
            raise cancelled

    assert caught.value is cancelled
    assert MlflowClient().get_run(child.info.run_id).info.status == "KILLED"


def test_cancellation_after_creation_but_before_yield_kills_the_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import attempt

    _install_task_context(monkeypatch)
    parent = _running_parent("cancelled-before-yield")
    tracking = importlib.import_module("dsio.tracking.attempt")
    real_create_child = tracking._create_child
    cancelled = concurrent.futures.CancelledError("cancelled before body entry")

    def cancel_after_create(*args: object, **kwargs: object) -> object:
        real_create_child(*args, **kwargs)
        raise cancelled

    monkeypatch.setattr(tracking, "_create_child", cancel_after_create)

    with pytest.raises(concurrent.futures.CancelledError) as caught:
        with attempt(parent.info.run_id):
            raise AssertionError("the body must not execute")

    assert caught.value is cancelled
    children = MlflowClient().search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
    )
    assert len(children) == 1
    assert children[0].info.status == "KILLED"


def test_closed_parent_is_rejected_before_a_child_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, attempt

    _install_task_context(monkeypatch)
    client = MlflowClient()
    parent = _running_parent("closed-parent")
    client.set_terminated(parent.info.run_id, "FINISHED")

    with pytest.raises(TrackingError, match="must be RUNNING"):
        with attempt(parent.info.run_id):
            raise AssertionError("the body must not execute")

    runs = client.search_runs([parent.info.experiment_id])
    assert [run.info.run_id for run in runs] == [parent.info.run_id]


def test_deleted_running_parent_is_rejected_before_a_child_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient
    from mlflow.entities import ViewType

    from dsio.tracking import TrackingError, attempt

    _install_task_context(monkeypatch)
    client = MlflowClient()
    parent = _running_parent("deleted-parent")
    client.delete_run(parent.info.run_id)
    deleted = client.get_run(parent.info.run_id)
    assert deleted.info.status == "RUNNING"
    assert deleted.info.lifecycle_stage == "deleted"

    with pytest.raises(TrackingError, match="lifecycle stage"):
        with attempt(parent.info.run_id):
            raise AssertionError("the body must not execute")

    runs = client.search_runs(
        [parent.info.experiment_id],
        run_view_type=ViewType.ALL,
    )
    assert [run.info.run_id for run in runs] == [parent.info.run_id]


def test_success_finalization_cancellation_kills_the_child_and_stays_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import attempt

    _install_task_context(monkeypatch)
    parent = _running_parent("cancelled-finalization")
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
        with attempt(parent.info.run_id) as child:
            pass

    assert caught.value is cancelled
    assert real_client.get_run(child.info.run_id).info.status == "KILLED"


def test_success_fails_closed_when_finished_status_cannot_be_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, attempt

    _install_task_context(monkeypatch)
    parent = _running_parent("failed-success-finalization")
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
        with attempt(parent.info.run_id) as child:
            pass

    assert real_client.get_run(child.info.run_id).info.status == "FAILED"


def test_cancellation_during_failed_success_recovery_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import attempt

    _install_task_context(monkeypatch)
    parent = _running_parent("cancelled-success-recovery")
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
        with attempt(parent.info.run_id) as child:
            pass

    assert caught.value is cancelled
    assert real_client.get_run(child.info.run_id).info.status == "KILLED"


def test_body_failure_survives_an_interrupted_terminal_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import attempt

    _install_task_context(monkeypatch)
    parent = _running_parent("failed-finalization")
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
        with attempt(parent.info.run_id) as child:
            raise task_error

    assert caught.value is task_error
    assert "tracking response lost" in " ".join(task_error.__notes__)
    assert real_client.get_run(child.info.run_id).info.status == "FAILED"


def test_committed_child_is_reconciled_when_creation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, attempt

    _install_task_context(monkeypatch)
    parent = _running_parent("uncertain-child-creation")
    tracking = importlib.import_module("dsio.tracking.attempt")
    real_client = MlflowClient()

    class CommitThenFailClient:
        def __getattr__(self, name: str) -> object:
            return getattr(real_client, name)

        def create_run(
            self,
            experiment_id: str,
            *,
            run_name: str | None,
            tags: dict[str, str],
        ) -> object:
            real_client.create_run(experiment_id, run_name=run_name, tags=tags)
            raise RuntimeError("response lost after commit")

    monkeypatch.setattr(tracking, "MlflowClient", CommitThenFailClient)

    with pytest.raises(TrackingError, match="response lost after commit"):
        with attempt(parent.info.run_id):
            raise AssertionError("the body must not execute")

    children = real_client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
    )
    assert len(children) == 1
    assert children[0].info.status == "FAILED"
