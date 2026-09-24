"""Prefect retry and concurrency behavior for visible tracked Runs."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest


def _isolate_prefect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith("PREFECT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))
    monkeypatch.setenv("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    monkeypatch.setenv("DO_NOT_TRACK", "1")


def test_attempt_requires_a_prefect_task_context() -> None:
    from dsio.tracking import TrackingError, attempt

    with pytest.raises(TrackingError, match="Prefect task context"):
        with attempt("parent-id"):
            raise AssertionError("the body must not execute")


def test_attempt_creates_one_visible_top_level_run_with_explicit_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_prefect(tmp_path, monkeypatch)

    import mlflow
    from mlflow import MlflowClient
    from mlflow.entities import Run
    from prefect import flow, task
    from prefect.context import get_run_context
    from prefect.testing.utilities import prefect_test_harness

    from dsio.tracking import attempt, resolve_experiment

    observed: dict[str, str | None] = {}

    @task
    def tracked(experiment_id: str) -> str:
        context = get_run_context()
        observed["flow_run_id"] = str(context.task_run.flow_run_id)
        observed["task_key"] = context.task_run.task_key
        observed["task_run_id"] = str(context.task_run.id)
        observed["dynamic_key"] = context.task_run.dynamic_key
        observed["attempt"] = str(context.task_run.run_count)
        active_before = mlflow.active_run()
        with attempt(experiment_id) as run:
            assert isinstance(run, Run)
            assert mlflow.active_run() is active_before
            MlflowClient().log_param(run.info.run_id, "dataset", "algae-v1")
            run_id = run.info.run_id
        assert mlflow.active_run() is active_before
        return run_id

    @flow
    def project_flow() -> tuple[str, str]:
        resolved = resolve_experiment("visible-attempt")
        return resolved.experiment_id, tracked(resolved.experiment_id)

    with prefect_test_harness():
        experiment_id, run_id = project_flow()

    run = MlflowClient().get_run(run_id)
    assert run.info.experiment_id == experiment_id
    assert run.info.status == "FINISHED"
    assert "mlflow.parentRunId" not in run.data.tags
    assert run.data.tags["dsio.prefect.flow_run_id"] == observed["flow_run_id"]
    assert run.data.tags["dsio.prefect.task_key"] == observed["task_key"]
    assert run.data.tags["dsio.prefect.task_run_id"] == observed["task_run_id"]
    assert run.data.tags["dsio.prefect.dynamic_key"] == observed["dynamic_key"]
    assert run.data.tags["dsio.prefect.attempt"] == observed["attempt"] == "1"
    assert run.data.params == {"dataset": "algae-v1"}


def test_prefect_retry_creates_a_fresh_run_and_preserves_failed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_prefect(tmp_path, monkeypatch)

    from mlflow import MlflowClient
    from prefect import flow, task
    from prefect.context import get_run_context
    from prefect.testing.utilities import prefect_test_harness

    from dsio.tracking import attempt, resolve_experiment

    @task(retries=1)
    def flaky(experiment_id: str) -> str:
        run_count = get_run_context().task_run.run_count
        with attempt(experiment_id) as run:
            MlflowClient().log_param(run.info.run_id, "observed_attempt", run_count)
            if run_count == 1:
                raise ValueError("retry me")
            return run.info.run_id

    @flow
    def project_flow() -> tuple[str, str]:
        resolved = resolve_experiment("retried-attempt")
        return resolved.experiment_id, flaky(resolved.experiment_id)

    with prefect_test_harness():
        experiment_id, successful_run_id = project_flow()

    client = MlflowClient()
    runs = client.search_runs(
        [experiment_id],
        order_by=["attributes.start_time ASC"],
    )
    assert [run.info.status for run in runs] == ["FAILED", "FINISHED"]
    assert [run.data.tags["dsio.prefect.attempt"] for run in runs] == ["1", "2"]
    assert [run.data.params["observed_attempt"] for run in runs] == ["1", "2"]
    assert len({run.data.tags["dsio.prefect.flow_run_id"] for run in runs}) == 1
    assert runs[0].info.run_id != runs[1].info.run_id == successful_run_id


def test_async_tasks_keep_explicit_child_destinations_when_they_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_prefect(tmp_path, monkeypatch)

    import mlflow
    from mlflow import MlflowClient
    from prefect import flow, task
    from prefect.context import get_run_context
    from prefect.testing.utilities import prefect_test_harness

    from dsio.tracking import attempt, resolve_experiment

    both_open = asyncio.Event()
    open_count = 0
    observed: dict[str, dict[str, str]] = {}

    @task
    async def tracked(label: str, experiment_id: str) -> str:
        nonlocal open_count
        task_run = get_run_context().task_run
        observed[label] = {
            "dsio.prefect.flow_run_id": str(task_run.flow_run_id),
            "dsio.prefect.task_key": task_run.task_key,
            "dsio.prefect.task_run_id": str(task_run.id),
            "dsio.prefect.dynamic_key": task_run.dynamic_key,
            "dsio.prefect.attempt": str(task_run.run_count),
        }
        active_before = mlflow.active_run()
        assert active_before is None
        with attempt(experiment_id) as run:
            assert mlflow.active_run() is active_before
            open_count += 1
            if open_count == 2:
                both_open.set()
            await asyncio.wait_for(both_open.wait(), timeout=5)
            MlflowClient().log_param(run.info.run_id, "label", label)
            return run.info.run_id

    @flow
    async def project_flow() -> tuple[str, list[str]]:
        resolved = resolve_experiment("concurrent-attempts")
        run_ids = await asyncio.gather(
            tracked("left", resolved.experiment_id),
            tracked("right", resolved.experiment_id),
        )
        return resolved.experiment_id, run_ids

    with prefect_test_harness():
        experiment_id, run_ids = asyncio.run(project_flow())

    client = MlflowClient()
    runs = [client.get_run(run_id) for run_id in run_ids]
    assert len(set(run_ids)) == 2
    assert {run.info.experiment_id for run in runs} == {experiment_id}
    assert {run.data.params["label"] for run in runs} == {"left", "right"}
    assert all("mlflow.parentRunId" not in run.data.tags for run in runs)
    assert {run.info.status for run in runs} == {"FINISHED"}
    for run in runs:
        label = run.data.params["label"]
        assert {
            tag: run.data.tags[tag]
            for tag in (
                "dsio.prefect.flow_run_id",
                "dsio.prefect.task_key",
                "dsio.prefect.task_run_id",
                "dsio.prefect.dynamic_key",
                "dsio.prefect.attempt",
            )
        } == observed[label]
