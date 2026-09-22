"""Isolation shared by executable reference-project tests."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pytest


@pytest.fixture
def reference_services(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith(("PREFECT_", "MLFLOW_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))
    monkeypatch.setenv("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    monkeypatch.setenv("PREFECT_LOGGING_LEVEL", "ERROR")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    tracking_uri = (tmp_path / "mlruns").resolve().as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
