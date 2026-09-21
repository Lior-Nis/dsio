"""Prefect retry and concurrency behavior for tracked child Runs."""

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


def test_attempt_creates_one_native_child_with_explicit_logging(
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

    from dsio.tracking import attempt, experiment

    observed: dict[str, str | None] = {}

    @task
    def tracked(parent_run_id: str) -> str:
        context = get_run_context()
        observed["task_key"] = context.task_run.task_key
        observed["task_run_id"] = str(context.task_run.id)
        observed["dynamic_key"] = context.task_run.dynamic_key
        observed["attempt"] = str(context.task_run.run_count)
        active_before = mlflow.active_run()
        with attempt(parent_run_id) as child:
            assert isinstance(child, Run)
            assert mlflow.active_run() is active_before
            MlflowClient().log_param(child.info.run_id, "dataset", "algae-v1")
            child_run_id = child.info.run_id
        assert mlflow.active_run() is active_before
        return child_run_id

    @flow
    def project_flow() -> tuple[str, str]:
        with experiment("child-attempt") as parent:
            return parent.info.run_id, tracked(parent.info.run_id)

    with prefect_test_harness():
        parent_run_id, child_run_id = project_flow()

    child = MlflowClient().get_run(child_run_id)
    assert child.info.status == "FINISHED"
    assert child.data.tags["mlflow.parentRunId"] == parent_run_id
    assert child.data.tags["dsio.prefect.task_key"] == observed["task_key"]
    assert child.data.tags["dsio.prefect.task_run_id"] == observed["task_run_id"]
    assert child.data.tags["dsio.prefect.dynamic_key"] == observed["dynamic_key"]
    assert child.data.tags["dsio.prefect.attempt"] == observed["attempt"] == "1"
    assert child.data.params == {"dataset": "algae-v1"}


def test_prefect_retry_creates_a_fresh_child_and_preserves_failed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_prefect(tmp_path, monkeypatch)

    from mlflow import MlflowClient
    from prefect import flow, task
    from prefect.context import get_run_context
    from prefect.testing.utilities import prefect_test_harness

    from dsio.tracking import attempt, experiment

    @task(retries=1)
    def flaky(parent_run_id: str) -> str:
        run_count = get_run_context().task_run.run_count
        with attempt(parent_run_id) as child:
            MlflowClient().log_param(child.info.run_id, "observed_attempt", run_count)
            if run_count == 1:
                raise ValueError("retry me")
            return child.info.run_id

    @flow
    def project_flow() -> tuple[str, str]:
        with experiment("retried-child-attempt") as parent:
            return parent.info.run_id, flaky(parent.info.run_id)

    with prefect_test_harness():
        parent_run_id, successful_child_id = project_flow()

    client = MlflowClient()
    parent = client.get_run(parent_run_id)
    children = client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        order_by=["attributes.start_time ASC"],
    )
    assert [child.info.status for child in children] == ["FAILED", "FINISHED"]
    assert [child.data.tags["dsio.prefect.attempt"] for child in children] == ["1", "2"]
    assert [child.data.params["observed_attempt"] for child in children] == ["1", "2"]
    assert children[0].info.run_id != children[1].info.run_id == successful_child_id


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

    from dsio.tracking import attempt, experiment

    both_open = asyncio.Event()
    open_count = 0
    observed: dict[str, dict[str, str]] = {}

    @task
    async def tracked(label: str, parent_run_id: str) -> str:
        nonlocal open_count
        task_run = get_run_context().task_run
        observed[label] = {
            "dsio.prefect.task_key": task_run.task_key,
            "dsio.prefect.task_run_id": str(task_run.id),
            "dsio.prefect.dynamic_key": task_run.dynamic_key,
            "dsio.prefect.attempt": str(task_run.run_count),
        }
        active_before = mlflow.active_run()
        assert active_before is not None
        assert active_before.info.run_id == parent_run_id
        with attempt(parent_run_id) as child:
            assert mlflow.active_run() is active_before
            open_count += 1
            if open_count == 2:
                both_open.set()
            await asyncio.wait_for(both_open.wait(), timeout=5)
            MlflowClient().log_param(child.info.run_id, "label", label)
            return child.info.run_id

    @flow
    async def project_flow() -> tuple[str, list[str]]:
        with experiment("concurrent-child-attempts") as parent:
            child_ids = await asyncio.gather(
                tracked("left", parent.info.run_id),
                tracked("right", parent.info.run_id),
            )
            return parent.info.run_id, child_ids

    with prefect_test_harness():
        parent_run_id, child_ids = asyncio.run(project_flow())

    client = MlflowClient()
    children = [client.get_run(child_id) for child_id in child_ids]
    assert len(set(child_ids)) == 2
    assert {child.data.params["label"] for child in children} == {"left", "right"}
    assert {child.data.tags["mlflow.parentRunId"] for child in children} == {parent_run_id}
    assert {child.info.status for child in children} == {"FINISHED"}
    for child in children:
        label = child.data.params["label"]
        assert {
            tag: child.data.tags[tag]
            for tag in (
                "dsio.prefect.task_key",
                "dsio.prefect.task_run_id",
                "dsio.prefect.dynamic_key",
                "dsio.prefect.attempt",
            )
        } == observed[label]
