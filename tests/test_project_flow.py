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
    from dsio.tracking import attempt, experiment, record_provenance

    @task
    def identify_dataset(dataset: dict[str, object], parent_run_id: str) -> tuple[str, str]:
        with attempt(parent_run_id) as child:
            digest = sha256_of(dataset)
            record_provenance(
                child.info.run_id,
                {"dataset": dataset, "seed": 7},
                components={"identity": "dsio.contracts:sha256_of"},
            )
            MlflowClient().log_param(child.info.run_id, "dataset_digest", digest)
            return digest, child.info.run_id

    @flow
    def project_flow() -> tuple[str, str]:
        with experiment("algae-training") as parent:
            digest, child_run_id = identify_dataset(
                {"name": "algae", "revision": 1}, parent.info.run_id
            )
            assert digest == "2ca217b09c12c36031e1078aab6a81e705f1281b1eb841c876f98cef5cb03556"
            return digest, child_run_id

    with prefect_test_harness():
        digest, child_run_id = project_flow()

    assert digest == "2ca217b09c12c36031e1078aab6a81e705f1281b1eb841c876f98cef5cb03556"
    child = MlflowClient().get_run(child_run_id)
    assert child.info.status == "FINISHED"
    assert child.data.params["dataset_digest"] == digest
    assert len(child.data.tags["dsio.execution_identity"]) == 64
    assert child.data.tags["dsio.version"]
    parent_run_id = child.data.tags["mlflow.parentRunId"]
    assert MlflowClient().get_run(parent_run_id).info.status == "FINISHED"
