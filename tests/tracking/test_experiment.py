"""The explicit parent-Run boundary around a consumer-owned Prefect flow."""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import signal
from types import SimpleNamespace

import pytest


def test_experiment_creates_and_finishes_one_native_active_run() -> None:
    import mlflow
    from mlflow import MlflowClient

    from dsio.tracking import experiment

    client = MlflowClient()

    with experiment("project-flow") as parent:
        assert isinstance(parent, mlflow.ActiveRun)
        assert mlflow.active_run() is parent
        assert client.get_run(parent.info.run_id).info.status == "RUNNING"

    stored_experiment = client.get_experiment_by_name("project-flow")
    assert stored_experiment is not None
    runs = client.search_runs([stored_experiment.experiment_id])
    assert [run.info.run_id for run in runs] == [parent.info.run_id]
    assert runs[0].info.status == "FINISHED"
    assert mlflow.active_run() is None


def test_each_entry_creates_a_fresh_parent_without_changing_prior_evidence() -> None:
    import mlflow
    from mlflow import MlflowClient

    from dsio.tracking import experiment

    client = MlflowClient()
    with experiment("repeated-flow") as first:
        mlflow.log_param("dataset", "algae-v1")

    first_before_retry = client.get_run(first.info.run_id)

    with experiment("repeated-flow") as retry:
        mlflow.log_param("dataset", "algae-v2")

    assert retry.info.run_id != first.info.run_id
    first_after_retry = client.get_run(first.info.run_id)
    assert first_after_retry.info.status == "FINISHED"
    assert first_after_retry.info.end_time == first_before_retry.info.end_time
    assert first_after_retry.data.params == {"dataset": "algae-v1"}


def test_inherited_mlflow_run_id_cannot_resume_a_prior_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import experiment

    client = MlflowClient()
    with experiment("environment-resumption") as first:
        pass

    monkeypatch.setenv("MLFLOW_RUN_ID", first.info.run_id)
    with experiment("environment-resumption") as second:
        pass

    assert second.info.run_id != first.info.run_id
    assert client.get_run(first.info.run_id).info.status == "FINISHED"


def test_cancellation_is_killed_and_propagates_unchanged() -> None:
    from mlflow import MlflowClient

    from dsio.tracking import experiment

    client = MlflowClient()
    cancelled = asyncio.CancelledError("cancelled by the flow runner")

    with pytest.raises(asyncio.CancelledError) as caught:
        with experiment("cancelled-flow") as parent:
            raise cancelled

    assert caught.value is cancelled
    assert client.get_run(parent.info.run_id).info.status == "KILLED"


@pytest.mark.parametrize("error", [RuntimeError("flow failed"), AssertionError("invalid output")])
def test_failure_is_failed_and_propagates_unchanged(error: Exception) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import experiment

    client = MlflowClient()
    with pytest.raises(type(error)) as caught:
        with experiment("failed-flow") as parent:
            raise error

    assert caught.value is error
    assert client.get_run(parent.info.run_id).info.status == "FAILED"


def test_parent_is_failed_when_body_ends_it_before_raising() -> None:
    import mlflow
    from mlflow import MlflowClient

    from dsio.tracking import experiment

    client = MlflowClient()
    original = RuntimeError("flow failed after ending its parent")

    with pytest.raises(RuntimeError) as caught:
        with experiment("premature-end") as parent:
            mlflow.end_run("FINISHED")
            raise original

    assert caught.value is original
    assert client.get_run(parent.info.run_id).info.status == "FAILED"
    assert any("no longer the active Run" in note for note in original.__notes__)


def test_parent_finalization_never_terminates_an_unrelated_active_run() -> None:
    import mlflow
    from mlflow import MlflowClient

    from dsio.tracking import experiment

    client = MlflowClient()
    original = RuntimeError("flow failed with an unrelated run active")

    try:
        with pytest.raises(RuntimeError) as caught:
            with experiment("active-run-mismatch") as parent:
                mlflow.end_run("FINISHED")
                unrelated = mlflow.start_run()
                raise original

        assert caught.value is original
        assert client.get_run(parent.info.run_id).info.status == "FAILED"
        assert client.get_run(unrelated.info.run_id).info.status == "RUNNING"
        assert mlflow.active_run() is unrelated
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run("KILLED")


def test_prefect_and_runtime_cancellation_forms_are_killed() -> None:
    from mlflow import MlflowClient
    from prefect.exceptions import CancelledRun, TerminationSignal

    from dsio.tracking import experiment

    client = MlflowClient()
    cancellations: list[BaseException] = [
        concurrent.futures.CancelledError("future cancelled"),
        CancelledRun("flow cancelled"),
        TerminationSignal(signal.SIGTERM),
        KeyboardInterrupt(),
    ]

    for index, cancelled in enumerate(cancellations):
        with pytest.raises(type(cancelled)) as caught:
            with experiment(f"cancelled-flow-{index}") as parent:
                raise cancelled

        assert caught.value is cancelled
        assert client.get_run(parent.info.run_id).info.status == "KILLED"


def test_non_cancellation_base_exception_is_failed() -> None:
    from mlflow import MlflowClient

    from dsio.tracking import experiment

    client = MlflowClient()
    failure = SystemExit("worker exited")

    with pytest.raises(SystemExit) as caught:
        with experiment("system-exit") as parent:
            raise failure

    assert caught.value is failure
    assert client.get_run(parent.info.run_id).info.status == "FAILED"


def test_an_ambient_active_run_is_rejected_before_a_parent_is_created() -> None:
    import mlflow
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, experiment

    client = MlflowClient()
    with mlflow.start_run() as ambient:
        with pytest.raises(TrackingError, match=r"already active.*end it"):
            with experiment("must-not-nest"):
                raise AssertionError("the body must not execute")

        assert mlflow.active_run() is ambient
        assert client.get_experiment_by_name("must-not-nest") is None


def test_run_creation_failure_prevents_the_body_and_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow.exceptions import MlflowException

    from dsio.tracking import TrackingError, experiment

    tracking = importlib.import_module("dsio.tracking.experiment")
    body_entered = False

    class FailingClient:
        def get_experiment_by_name(self, _name: str) -> object:
            return SimpleNamespace(experiment_id="experiment-1")

        def create_run(
            self,
            _experiment_id: str,
            *,
            run_name: str | None,
            tags: dict[str, str],
        ) -> object:
            raise MlflowException("tracking store is read-only")

        def search_runs(self, *args: object, **kwargs: object) -> list[object]:
            return []

    monkeypatch.setattr(tracking, "MlflowClient", FailingClient)

    with pytest.raises(
        TrackingError,
        match=r"create.*MLflow parent Run.*project-flow.*tracking store is read-only",
    ):
        with experiment("project-flow"):
            body_entered = True

    assert body_entered is False


@pytest.mark.parametrize(
    ("creation_error", "expected_status"),
    [
        (RuntimeError("response lost after commit"), "FAILED"),
        (concurrent.futures.CancelledError("cancelled after commit"), "KILLED"),
    ],
)
def test_committed_parent_is_reconciled_when_creation_raises(
    creation_error: Exception,
    expected_status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, experiment

    tracking = importlib.import_module("dsio.tracking.experiment")
    real_client = MlflowClient()

    class CommitThenFailClient:
        def get_experiment_by_name(self, name: str) -> object | None:
            return real_client.get_experiment_by_name(name)

        def create_experiment(self, name: str) -> str:
            return real_client.create_experiment(name)

        def create_run(
            self,
            experiment_id: str,
            *,
            run_name: str | None,
            tags: dict[str, str],
        ) -> object:
            real_client.create_run(experiment_id, run_name=run_name, tags=tags)
            raise creation_error

        def search_runs(self, *args: object, **kwargs: object) -> list[object]:
            return real_client.search_runs(*args, **kwargs)

        def set_terminated(self, run_id: str, status: str) -> None:
            real_client.set_terminated(run_id, status)

    monkeypatch.setattr(tracking, "MlflowClient", CommitThenFailClient)

    expected_error = type(creation_error) if expected_status == "KILLED" else TrackingError
    with pytest.raises(expected_error) as caught:
        with experiment("uncertain-creation"):
            raise AssertionError("the body must not execute")

    if expected_status == "KILLED":
        assert caught.value is creation_error

    stored = real_client.get_experiment_by_name("uncertain-creation")
    assert stored is not None
    runs = real_client.search_runs([stored.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == expected_status


def test_cancellation_during_activation_kills_the_created_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlflow
    from mlflow import MlflowClient

    from dsio.tracking import experiment

    def cancel_activation(*, run_id: str) -> object:
        raise KeyboardInterrupt(run_id)

    monkeypatch.setattr(mlflow, "start_run", cancel_activation)

    with pytest.raises(KeyboardInterrupt):
        with experiment("cancelled-activation"):
            raise AssertionError("the body must not execute")

    client = MlflowClient()
    stored = client.get_experiment_by_name("cancelled-activation")
    assert stored is not None
    runs = client.search_runs([stored.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "KILLED"


def test_cancellation_after_native_activation_kills_the_created_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlflow
    from mlflow import MlflowClient

    from dsio.tracking import experiment

    real_start_run = mlflow.start_run

    def activate_then_cancel(*, run_id: str) -> object:
        real_start_run(run_id=run_id)
        raise KeyboardInterrupt(run_id)

    monkeypatch.setattr(mlflow, "start_run", activate_then_cancel)

    with pytest.raises(KeyboardInterrupt):
        with experiment("post-activation-cancellation"):
            raise AssertionError("the body must not execute")

    client = MlflowClient()
    stored = client.get_experiment_by_name("post-activation-cancellation")
    assert stored is not None
    runs = client.search_runs([stored.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "KILLED"
    assert mlflow.active_run() is None


@pytest.mark.parametrize("operation", ["resolve", "create"])
def test_entry_cancellation_propagates_unchanged(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prefect.exceptions import CancelledRun

    from dsio.tracking import experiment

    tracking = importlib.import_module("dsio.tracking.experiment")
    cancelled = CancelledRun(f"cancelled during {operation}")

    class CancelledClient:
        def get_experiment_by_name(self, _name: str) -> object:
            if operation == "resolve":
                raise cancelled
            return SimpleNamespace(experiment_id="experiment-1")

        def create_run(
            self,
            _experiment_id: str,
            *,
            run_name: str | None,
            tags: dict[str, str],
        ) -> object:
            raise cancelled

    monkeypatch.setattr(tracking, "MlflowClient", CancelledClient)

    with pytest.raises(CancelledRun) as caught:
        with experiment("entry-cancelled"):
            raise AssertionError("the body must not execute")

    assert caught.value is cancelled


def test_activation_and_cleanup_failures_remain_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlflow
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, experiment

    tracking = importlib.import_module("dsio.tracking.experiment")
    real_client = MlflowClient()

    class CleanupFailingClient:
        def get_experiment_by_name(self, name: str) -> object | None:
            return real_client.get_experiment_by_name(name)

        def create_experiment(self, name: str) -> str:
            return real_client.create_experiment(name)

        def create_run(
            self,
            experiment_id: str,
            *,
            run_name: str | None,
            tags: dict[str, str],
        ) -> object:
            return real_client.create_run(experiment_id, run_name=run_name, tags=tags)

        def set_terminated(self, _run_id: str, _status: str) -> None:
            raise OSError("cleanup transport failed")

    def fail_activation(*, run_id: str) -> object:
        raise RuntimeError(f"activation failed for {run_id}")

    monkeypatch.setattr(tracking, "MlflowClient", CleanupFailingClient)
    monkeypatch.setattr(mlflow, "start_run", fail_activation)

    with pytest.raises(
        TrackingError,
        match=r"activation failed.*marking the created Run FAILED also failed.*cleanup transport",
    ):
        with experiment("activation-cleanup-failure"):
            raise AssertionError("the body must not execute")


def test_concurrent_experiment_creation_race_reuses_the_native_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS

    from dsio.tracking import experiment

    tracking = importlib.import_module("dsio.tracking.experiment")
    real_client = MlflowClient()

    class RacingClient:
        first_lookup = True

        def get_experiment_by_name(self, name: str) -> object | None:
            if self.first_lookup:
                self.first_lookup = False
                return None
            return real_client.get_experiment_by_name(name)

        def create_experiment(self, name: str) -> str:
            real_client.create_experiment(name)
            raise MlflowException("created by another flow", RESOURCE_ALREADY_EXISTS)

        def create_run(
            self,
            experiment_id: str,
            *,
            run_name: str | None,
            tags: dict[str, str],
        ) -> object:
            return real_client.create_run(experiment_id, run_name=run_name, tags=tags)

    racing_client = RacingClient()
    monkeypatch.setattr(tracking, "MlflowClient", lambda: racing_client)

    with experiment("raced-experiment") as parent:
        pass

    stored = real_client.get_experiment_by_name("raced-experiment")
    assert stored is not None
    assert real_client.get_run(parent.info.run_id).info.experiment_id == stored.experiment_id


def test_successful_body_fails_closed_when_terminal_status_cannot_be_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlflow
    from mlflow.exceptions import MlflowException

    from dsio.tracking import TrackingError, experiment

    real_end_run = mlflow.end_run

    def fail_after_ending(status: str) -> None:
        real_end_run(status)
        raise MlflowException("status store unavailable")

    monkeypatch.setattr(mlflow, "end_run", fail_after_ending)

    with pytest.raises(
        TrackingError,
        match=r"finish.*MLflow parent Run.*successful-flow.*status store unavailable",
    ):
        with experiment("successful-flow"):
            pass


def test_finalization_failure_does_not_hide_the_original_flow_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlflow
    from mlflow.exceptions import MlflowException

    from dsio.tracking import experiment

    real_end_run = mlflow.end_run

    def fail_after_ending(status: str) -> None:
        real_end_run(status)
        raise MlflowException("status store unavailable")

    monkeypatch.setattr(mlflow, "end_run", fail_after_ending)
    original = RuntimeError("flow failed")

    with pytest.raises(RuntimeError) as caught:
        with experiment("failed-finalization"):
            raise original

    assert caught.value is original
    assert any("status store unavailable" in note for note in original.__notes__)


def test_finalization_interrupt_does_not_hide_the_original_flow_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlflow
    from mlflow import MlflowClient

    from dsio.tracking import experiment

    real_end_run = mlflow.end_run
    interrupted = False

    def interrupt_finalization(_status: str) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("interrupted while persisting status")
        real_end_run(_status)

    monkeypatch.setattr(mlflow, "end_run", interrupt_finalization)
    original = RuntimeError("flow failed")

    with pytest.raises(RuntimeError) as caught:
        with experiment("interrupted-failure-finalization") as parent:
            raise original

    assert caught.value is original
    assert MlflowClient().get_run(parent.info.run_id).info.status == "FAILED"
    assert any("interrupted while persisting status" in note for note in original.__notes__)


def test_cancellation_during_success_finalization_is_killed_and_propagated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlflow
    from mlflow import MlflowClient

    from dsio.tracking import experiment

    cancelled = KeyboardInterrupt("cancelled while persisting success")
    real_end_run = mlflow.end_run
    interrupted = False

    def cancel_finalization(_status: str) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise cancelled
        real_end_run(_status)

    monkeypatch.setattr(mlflow, "end_run", cancel_finalization)

    with pytest.raises(KeyboardInterrupt) as caught:
        with experiment("cancelled-success-finalization") as parent:
            pass

    assert caught.value is cancelled
    assert MlflowClient().get_run(parent.info.run_id).info.status == "KILLED"


def test_exception_derived_cancellation_during_finalization_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlflow
    from mlflow import MlflowClient
    from prefect.exceptions import CancelledRun

    from dsio.tracking import experiment

    cancelled = CancelledRun("cancelled while persisting success")
    real_end_run = mlflow.end_run
    interrupted = False

    def cancel_once(status: str) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise cancelled
        real_end_run(status)

    monkeypatch.setattr(mlflow, "end_run", cancel_once)

    with pytest.raises(CancelledRun) as caught:
        with experiment("exception-cancelled-finalization") as parent:
            pass

    assert caught.value is cancelled
    assert MlflowClient().get_run(parent.info.run_id).info.status == "KILLED"


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (
            BaseExceptionGroup(
                "cancelled",
                [asyncio.CancelledError(), KeyboardInterrupt()],
            ),
            "KILLED",
        ),
        (
            BaseExceptionGroup(
                "mixed",
                [asyncio.CancelledError(), RuntimeError("task failed")],
            ),
            "FAILED",
        ),
    ],
)
def test_exception_groups_use_their_leaf_failures_for_status(
    failure: BaseExceptionGroup,
    expected_status: str,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import experiment

    with pytest.raises(BaseExceptionGroup) as caught:
        with experiment("grouped-failure") as parent:
            raise failure

    assert caught.value is failure
    assert MlflowClient().get_run(parent.info.run_id).info.status == expected_status
