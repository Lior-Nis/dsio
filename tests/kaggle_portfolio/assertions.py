"""Small assertions shared by the independent Kaggle consumer tests."""

from __future__ import annotations

import json
from pathlib import Path

from mlflow import MlflowClient


def assert_execution_evidence(
    run_id: str,
    destination: Path,
    *,
    optimizer: str,
    optimizer_parameters: dict[str, float],
) -> None:
    client = MlflowClient()
    run = client.get_run(run_id)
    artifact = client.download_artifacts(run_id, "provenance.json", str(destination))
    provenance = json.loads(Path(artifact).read_text())
    configuration = provenance["configuration"]
    execution = configuration["execution"]

    assert execution["requested_accelerator"] == "cpu"
    assert execution["requested_devices"] == "1"
    assert execution["requested_precision"] == "32-true"
    assert execution["resolved_device"] == "cpu"
    assert execution["resolved_devices"] == "1"
    assert execution["resolved_precision"] == "32-true"
    for key, value in execution.items():
        assert run.data.params[f"execution.{key}"] == value
    assert configuration["fold"] == 0
    assert configuration["batch_size"] == 4
    assert configuration["num_workers"] == 0
    assert configuration["optimizer_parameters"] == optimizer_parameters
    assert provenance["components"]["optimizer"] == optimizer


def assert_replay_run_ids_differ(first: dict[str, object], second: dict[str, object]) -> None:
    keys = {key for key in first if key == "parent_run_id" or key.endswith("_run_id")}
    assert keys
    assert keys <= second.keys()
    for key in keys:
        assert first[key] != second[key]


def assert_downstream_evidence(result: dict[str, object], destination: Path) -> None:
    client = MlflowClient()
    identities = result["identities"]
    assert isinstance(identities, dict)
    for stage in ("evaluation", "inference"):
        run_id = result[f"{stage}_run_id"]
        assert isinstance(run_id, str)
        artifact = client.download_artifacts(run_id, "provenance.json", str(destination / stage))
        configuration = json.loads(Path(artifact).read_text())["configuration"]
        assert configuration["export_identity"] == identities["export"]

    inference_run_id = result["inference_run_id"]
    assert isinstance(inference_run_id, str)
    artifact = client.download_artifacts(
        inference_run_id,
        "outputs/submission.json",
        str(destination / "submission"),
    )
    submission = json.loads(Path(artifact).read_text())
    assert submission["digest"] == result["submission_digest"]
