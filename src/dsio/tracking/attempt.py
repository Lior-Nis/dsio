"""Native MLflow child Runs for Prefect task attempts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

from mlflow import MlflowClient
from mlflow.entities import Run
from prefect.context import TaskRunContext

from dsio.tracking._lifecycle import (
    TrackingError,
    create_run,
    is_cancellation,
    reconcile_creation,
    status_for,
)

_PARENT_RUN_TAG = "mlflow.parentRunId"
_TASK_KEY_TAG = "dsio.prefect.task_key"
_TASK_RUN_ID_TAG = "dsio.prefect.task_run_id"
_DYNAMIC_KEY_TAG = "dsio.prefect.dynamic_key"
_ATTEMPT_TAG = "dsio.prefect.attempt"


@contextmanager
def attempt(parent_run_id: str) -> Iterator[Run]:
    """Create one native MLflow child Run for the current Prefect task attempt."""
    context = TaskRunContext.get()
    if context is None:
        raise TrackingError(
            "A Prefect task context is required to open a tracked MLflow attempt."
        )

    client = MlflowClient()
    parent = _require_parent(client, parent_run_id)
    creation_token = uuid4().hex
    child: Run | None = None

    try:
        child = _create_child(client, parent, context, creation_token)
        yield child
    except BaseException as error:
        status = status_for(error)
        if child is None:
            reconciliation = reconcile_creation(
                client,
                parent.info.experiment_id,
                creation_token,
                status,
            )
            if reconciliation is not None:
                error.add_note(
                    f"Interrupted while opening an MLflow child Run beneath parent "
                    f"{parent.info.run_id!r}; {reconciliation}."
                )
            raise
        try:
            client.set_terminated(child.info.run_id, status)
        except BaseException as finalization_error:  # noqa: BLE001 - preserve task failure
            recovery = _recover_status(client, child.info.run_id, status)
            message = (
                f"Could not finish MLflow child Run {child.info.run_id!r} as {status}: "
                f"{finalization_error}"
            )
            if recovery is not None:
                message += f"; direct status recovery also failed: {recovery}"
            error.add_note(message)
        raise
    else:
        try:
            client.set_terminated(child.info.run_id, "FINISHED")
        except BaseException as error:
            status = status_for(error)
            recovery = _recover_status(client, child.info.run_id, status)
            message = (
                f"Could not finish MLflow child Run {child.info.run_id!r} as FINISHED: "
                f"{error}"
            )
            if recovery is not None and (
                is_cancellation(recovery) or not isinstance(recovery, Exception)
            ):
                recovery_status = status_for(recovery)
                terminal_recovery = _recover_status(
                    client,
                    child.info.run_id,
                    recovery_status,
                )
                message += f"; direct {status} status recovery was interrupted: {recovery}"
                if terminal_recovery is not None:
                    message += (
                        f"; direct {recovery_status} status recovery also failed: "
                        f"{terminal_recovery}"
                    )
                recovery.add_note(message)
                raise recovery from error
            if recovery is not None:
                message += f"; direct {status} status recovery also failed: {recovery}"
            if is_cancellation(error) or not isinstance(error, Exception):
                error.add_note(message)
                raise
            raise TrackingError(message) from error


def _recover_status(
    client: MlflowClient,
    run_id: str,
    status: str,
) -> BaseException | None:
    try:
        client.set_terminated(run_id, status)
    except BaseException as error:  # noqa: BLE001 - caller preserves the original failure
        return error
    return None


def _require_parent(client: MlflowClient, parent_run_id: str) -> Run:
    if not parent_run_id:
        raise TrackingError("A non-empty MLflow parent Run ID is required.")
    try:
        parent = client.get_run(parent_run_id)
    except BaseException as error:
        if is_cancellation(error) or not isinstance(error, Exception):
            raise
        raise TrackingError(
            f"Could not resolve MLflow parent Run {parent_run_id!r}: {error}"
        ) from error
    if parent.info.lifecycle_stage != "active":
        raise TrackingError(
            f"MLflow parent Run {parent_run_id!r} must have lifecycle stage 'active', "
            f"not {parent.info.lifecycle_stage!r}."
        )
    if parent.info.status != "RUNNING":
        raise TrackingError(
            f"MLflow parent Run {parent_run_id!r} must be RUNNING, not "
            f"{parent.info.status!r}."
        )
    if _PARENT_RUN_TAG in parent.data.tags:
        raise TrackingError(
            f"MLflow Run {parent_run_id!r} is already a child; a top-level parent Run "
            "is required."
        )
    return parent


def _create_child(
    client: MlflowClient,
    parent: Run,
    context: TaskRunContext,
    creation_token: str,
) -> Run:
    task_run = context.task_run
    tags = {
        _PARENT_RUN_TAG: parent.info.run_id,
        _TASK_KEY_TAG: task_run.task_key,
        _TASK_RUN_ID_TAG: str(task_run.id),
        _DYNAMIC_KEY_TAG: task_run.dynamic_key,
        _ATTEMPT_TAG: str(task_run.run_count),
    }
    return create_run(
        client,
        parent.info.experiment_id,
        run_name=f"{context.task.name} attempt {task_run.run_count}",
        tags=tags,
        description=f"MLflow child Run beneath parent {parent.info.run_id!r}",
        creation_token=creation_token,
    )
