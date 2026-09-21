"""Shared failure semantics for MLflow Run creation and termination."""

from __future__ import annotations

import asyncio
import concurrent.futures
from uuid import uuid4

from mlflow import MlflowClient
from mlflow.entities import Run
from prefect.exceptions import CancelledRun, TerminationSignal

CREATION_TOKEN_TAG = "dsio_creation_token"

_CANCELLATION_ERRORS = (
    asyncio.CancelledError,
    concurrent.futures.CancelledError,
    CancelledRun,
    TerminationSignal,
    KeyboardInterrupt,
)


class TrackingError(RuntimeError):
    """A required MLflow lifecycle operation did not complete."""


def create_run(
    client: MlflowClient,
    experiment_id: str,
    *,
    run_name: str | None,
    tags: dict[str, str],
    description: str,
    creation_token: str | None = None,
) -> Run:
    """Create exactly one Run or reconcile a commit whose response was lost."""
    token = creation_token or uuid4().hex
    try:
        return client.create_run(
            experiment_id,
            run_name=run_name,
            tags={**tags, CREATION_TOKEN_TAG: token},
        )
    except BaseException as error:
        status = status_for(error)
        reconciliation = reconcile_creation(client, experiment_id, token, status)
        message = f"Could not create {description}: {error}"
        if reconciliation is not None:
            message += f"; {reconciliation}"
        if is_cancellation(error) or not isinstance(error, Exception):
            if reconciliation is not None:
                error.add_note(message)
            raise
        raise TrackingError(message) from error


def is_cancellation(error: BaseException) -> bool:
    """Return whether an error consists only of explicit cancellation signals."""
    if isinstance(error, BaseExceptionGroup):
        return all(is_cancellation(child) for child in error.exceptions)
    return isinstance(error, _CANCELLATION_ERRORS)


def status_for(error: BaseException) -> str:
    """Map a Python failure to its native MLflow terminal status."""
    return "KILLED" if is_cancellation(error) else "FAILED"


def reconcile_creation(
    client: MlflowClient,
    experiment_id: str,
    token: str,
    status: str,
) -> str | None:
    try:
        runs = client.search_runs(
            [experiment_id],
            filter_string=f"tags.{CREATION_TOKEN_TAG} = '{token}'",
            max_results=2,
        )
    except BaseException as error:  # noqa: BLE001 - preserve the creation failure
        return f"could not reconcile the uncertain create operation: {error}"
    if not runs:
        return None

    failures: list[str] = []
    for run in runs:
        try:
            client.set_terminated(run.info.run_id, status)
        except BaseException as error:  # noqa: BLE001 - report every uncertain Run
            failures.append(f"{run.info.run_id}: {error}")
    if failures:
        return "could not terminate reconciled Run(s): " + "; ".join(failures)
    run_ids = ", ".join(run.info.run_id for run in runs)
    return f"reconciled committed Run(s) {run_ids} as {status}"
