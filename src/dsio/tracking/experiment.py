"""One native MLflow parent Run for one project-owned flow execution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from mlflow.tracking.fluent import ActiveRun

from dsio.tracking._lifecycle import (
    TrackingError,
    create_run,
    is_cancellation,
    status_for,
)


@contextmanager
def experiment(experiment_name: str, *, run_name: str | None = None) -> Iterator[ActiveRun]:
    """Create and activate a fresh MLflow parent Run for one flow execution."""
    active = mlflow.active_run()
    if active is not None:
        raise TrackingError(
            f"MLflow Run {active.info.run_id!r} is already active; end it before opening "
            "a DSio experiment parent Run."
        )

    client = MlflowClient()
    experiment_id = _resolve_experiment(client, experiment_name)
    created = _create_parent(client, experiment_id, experiment_name, run_name)

    run_id = created.info.run_id
    parent: ActiveRun | None = None
    try:
        parent = mlflow.start_run(run_id=run_id)
        yield parent
    except BaseException as error:
        status = status_for(error)
        if parent is None:
            message = (
                f"Could not activate MLflow parent Run {run_id!r} for experiment "
                f"{experiment_name!r}: {error}"
            )
            try:
                _terminate_parent(client, run_id, experiment_name, status, require_active=False)
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve cancellation
                message += f"; marking the created Run {status} also failed: {cleanup_error}"
            if is_cancellation(error) or not isinstance(error, Exception):
                error.add_note(message)
                raise
            raise TrackingError(message) from error

        try:
            _terminate_parent(
                client,
                run_id,
                experiment_name,
                status_for(error),
                require_active=True,
            )
        except TrackingError as tracking_error:
            error.add_note(str(tracking_error))
        except BaseException as finalization_error:  # noqa: BLE001 - preserve body failure
            recovery = _recover_parent_status(client, run_id, status)
            message = (
                f"Could not finish MLflow parent Run {run_id!r} for experiment "
                f"{experiment_name!r} as {status}: {finalization_error}"
            )
            if recovery is not None:
                message += f"; direct status recovery also failed: {recovery}"
            error.add_note(message)
        raise
    else:
        try:
            _terminate_parent(client, run_id, experiment_name, "FINISHED", require_active=True)
        except TrackingError:
            raise
        except BaseException as error:
            status = status_for(error)
            recovery = _recover_parent_status(client, run_id, status)
            if recovery is not None:
                error.add_note(
                    f"Could not persist {status} for MLflow parent Run {run_id!r}: {recovery}"
                )
            raise


def _resolve_experiment(client: MlflowClient, experiment_name: str) -> str:
    try:
        stored = client.get_experiment_by_name(experiment_name)
        if stored is not None:
            return stored.experiment_id
        try:
            return client.create_experiment(experiment_name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
            stored = client.get_experiment_by_name(experiment_name)
            if stored is None:
                raise
            return stored.experiment_id
    except BaseException as error:
        if is_cancellation(error) or not isinstance(error, Exception):
            raise
        raise TrackingError(
            f"Could not resolve MLflow experiment {experiment_name!r}: {error}"
        ) from error


def _create_parent(
    client: MlflowClient,
    experiment_id: str,
    experiment_name: str,
    run_name: str | None,
) -> Run:
    return create_run(
        client,
        experiment_id,
        run_name=run_name,
        tags={},
        description=f"MLflow parent Run for experiment {experiment_name!r}",
    )


def _recover_parent_status(client: MlflowClient, run_id: str, status: str) -> BaseException | None:
    try:
        client.set_terminated(run_id, status)
    except BaseException as error:  # noqa: BLE001 - caller must preserve the original failure
        return error
    active = mlflow.active_run()
    if active is not None and active.info.run_id == run_id:
        try:
            mlflow.end_run(status)
        except BaseException as error:  # noqa: BLE001 - caller must preserve the original failure
            return error
    return None


def _terminate_parent(
    client: MlflowClient,
    run_id: str,
    experiment_name: str,
    status: str,
    *,
    require_active: bool,
) -> None:
    active = mlflow.active_run()
    if active is None or active.info.run_id != run_id:
        try:
            client.set_terminated(run_id, status)
        except Exception as error:
            if is_cancellation(error):
                raise
            raise TrackingError(
                f"Could not finish MLflow parent Run {run_id!r} for experiment "
                f"{experiment_name!r} as {status}: {error}"
            ) from error
        if require_active:
            observed = "no Run" if active is None else f"Run {active.info.run_id!r}"
            raise TrackingError(
                f"MLflow parent Run {run_id!r} for experiment {experiment_name!r} was no "
                f"longer the active Run ({observed} was active); its {status} status was "
                "persisted without terminating another Run."
            )
        return

    try:
        mlflow.end_run(status)
    except Exception as error:
        if is_cancellation(error):
            raise
        recovery_error = _recover_parent_status(client, run_id, status)
        recovery = (
            ""
            if recovery_error is None
            else f"; direct status recovery also failed: {recovery_error}"
        )
        raise TrackingError(
            f"Could not finish MLflow parent Run {run_id!r} for experiment "
            f"{experiment_name!r} as {status}: {error}{recovery}"
        ) from error
