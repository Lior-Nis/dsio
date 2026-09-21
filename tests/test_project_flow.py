"""A consumer project owns its Prefect flow and calls DSio as a library."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_project_owned_flow_calls_public_dsio_function(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in tuple(os.environ):
        if name.startswith("PREFECT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))
    monkeypatch.setenv("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    from mlflow import MlflowClient
    from prefect import flow, task
    from prefect.testing.utilities import prefect_test_harness

    from dsio.contracts import sha256_of
    from dsio.tracking import experiment

    @task
    def identify_dataset(dataset: dict[str, object], parent_run_id: str) -> tuple[str, str]:
        return sha256_of(dataset), parent_run_id

    @flow
    def project_flow() -> tuple[str, str]:
        with experiment("algae-training") as parent:
            digest, observed_parent_id = identify_dataset(
                {"name": "algae", "revision": 1}, parent.info.run_id
            )
            assert digest == "2ca217b09c12c36031e1078aab6a81e705f1281b1eb841c876f98cef5cb03556"
            return digest, observed_parent_id

    with prefect_test_harness():
        digest, parent_run_id = project_flow()

    assert digest == "2ca217b09c12c36031e1078aab6a81e705f1281b1eb841c876f98cef5cb03556"
    assert MlflowClient().get_run(parent_run_id).info.status == "FINISHED"
