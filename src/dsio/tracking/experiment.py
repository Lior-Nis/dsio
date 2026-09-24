"""Resolve native MLflow Experiments for project-owned Prefect flows."""

from __future__ import annotations

from mlflow import MlflowClient
from mlflow.entities import Experiment
from mlflow.exceptions import MlflowException

from dsio.tracking._lifecycle import TrackingError, is_cancellation


def resolve_experiment(experiment_name: str) -> Experiment:
    """Resolve or create a native MLflow Experiment without creating a Run."""
    client = MlflowClient()
    try:
        stored = client.get_experiment_by_name(experiment_name)
        if stored is not None:
            return stored
        try:
            experiment_id = client.create_experiment(experiment_name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
            stored = client.get_experiment_by_name(experiment_name)
            if stored is None:
                raise
            return stored
        return client.get_experiment(experiment_id)
    except BaseException as error:
        if is_cancellation(error) or not isinstance(error, Exception):
            raise
        raise TrackingError(
            f"Could not resolve MLflow experiment {experiment_name!r}: {error}"
        ) from error
