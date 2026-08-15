"""MLflow sink. Imported lazily so mlflow is never a hard dependency of a run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dsio.tracking.base import flatten_params


class MlflowTracker:
    """Mirrors a dsio run into an MLflow experiment.

    The dsio ``run_id`` is written as a tag rather than used as MLflow's own id, so the
    ledger stays the join key between the two systems even if the MLflow store is reset.
    """

    def __init__(self, experiment: str = "dsio", tracking_uri: str | None = None) -> None:
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._active = False

    def start_run(self, run_id: str, name: str, tags: dict[str, str]) -> None:
        self._mlflow.start_run(run_name=f"{name}-{run_id[-12:]}")
        self._mlflow.set_tags({**tags, "dsio.run_id": run_id})
        self._active = True

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(flatten_params(params))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        finite = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float) and float(value) == float(value)
        }
        if finite:
            self._mlflow.log_metrics(finite, step=step)

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def finish(self, status: str) -> None:
        if self._active:
            self._mlflow.end_run(status="FINISHED" if status == "completed" else "FAILED")
            self._active = False
