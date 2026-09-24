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
    from dsio.tracking import (
        attempt,
        evidence_uri,
        record_provenance,
        resolve_evidence,
        resolve_experiment,
    )

    @task
    def identify_dataset(dataset: dict[str, object], experiment_id: str) -> tuple[str, str]:
        with attempt(experiment_id) as run:
            digest = sha256_of(dataset)
            record_provenance(
                run.info.run_id,
                {"dataset": dataset, "seed": 7},
                components={"identity": "dsio.contracts:sha256_of"},
            )
            MlflowClient().log_param(run.info.run_id, "dataset_digest", digest)
            MlflowClient().log_dict(
                run.info.run_id,
                {"dataset_digest": digest},
                "outputs/dataset.json",
            )
            return digest, run.info.run_id

    @flow
    def project_flow() -> tuple[str, str]:
        resolved = resolve_experiment("algae-training")
        digest, run_id = identify_dataset({"name": "algae", "revision": 1}, resolved.experiment_id)
        assert digest == "2ca217b09c12c36031e1078aab6a81e705f1281b1eb841c876f98cef5cb03556"
        return digest, run_id

    with prefect_test_harness():
        digest, run_id = project_flow()

    assert digest == "2ca217b09c12c36031e1078aab6a81e705f1281b1eb841c876f98cef5cb03556"
    run = MlflowClient().get_run(run_id)
    assert run.info.status == "FINISHED"
    assert "mlflow.parentRunId" not in run.data.tags
    assert run.data.tags["dsio.prefect.flow_run_id"]
    assert run.data.params["dataset_digest"] == digest
    assert len(run.data.tags["dsio.execution_identity"]) == 64
    assert run.data.tags["dsio.version"]
    resolved = resolve_evidence(
        run.data.params["dsio.execution_identity"],
        required_artifacts={"outputs/dataset.json"},
    )
    assert resolved is not None
    assert resolved.info.run_id == run_id
    assert evidence_uri(resolved.info.run_id, "outputs/dataset.json") == (
        f"runs:/{run_id}/outputs/dataset.json"
    )
