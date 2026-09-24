"""Immutable native evidence references and consuming provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_evidence_uri_contains_an_exact_run_id_and_relative_artifact_path() -> None:
    from dsio.tracking import evidence_uri

    run_id = "a" * 32
    assert evidence_uri(run_id, "model/model.pkl") == f"runs:/{run_id}/model/model.pkl"


@pytest.mark.parametrize(
    ("reference", "path"),
    [
        ("models:/predictor@champion", "model"),
        ("models:/predictor/Production", "model"),
        ("not-a-run", "model"),
        ("a" * 32, "../model"),
        ("a" * 32, "/model"),
        ("a" * 32, "model\\weights"),
        ("a" * 32, "."),
        ("a" * 32, "outputs/a#b.json"),
        ("a" * 32, "outputs/a?b.json"),
        ("a" * 32, "outputs/a\x00b.json"),
    ],
)
def test_mutable_or_malformed_references_are_rejected(
    reference: str,
    path: str,
) -> None:
    from dsio.tracking import TrackingError, evidence_uri

    with pytest.raises(TrackingError, match="immutable|relative POSIX"):
        evidence_uri(reference, path)


def test_accepted_uri_round_trips_through_mlflow_without_path_changes() -> None:
    from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

    from dsio.tracking import evidence_uri

    run_id = "a" * 32
    uri = evidence_uri(run_id, "outputs/value.json")

    assert RunsArtifactRepository.parse_runs_uri(uri) == (run_id, "outputs/value.json")


def test_consuming_attempt_records_source_reference_without_copying_evidence(
    tmp_path: Path,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import evidence_uri, record_provenance

    client = MlflowClient()
    experiment_id = client.create_experiment("reuse-provenance")
    source = client.create_run(experiment_id)
    client.log_dict(source.info.run_id, {"weights": [1, 2]}, "model/model.json")
    client.set_terminated(source.info.run_id, "FINISHED")
    consumer = client.create_run(experiment_id)
    uri = evidence_uri(source.info.run_id, "model")

    record_provenance(
        consumer.info.run_id,
        {"source_run_id": source.info.run_id, "source_uri": uri},
    )

    artifact = Path(
        client.download_artifacts(consumer.info.run_id, "provenance.json", tmp_path)
    )
    configuration = json.loads(artifact.read_text())["configuration"]
    assert configuration == {"source_run_id": source.info.run_id, "source_uri": uri}
    artifact_paths = {item.path for item in client.list_artifacts(consumer.info.run_id)}
    assert artifact_paths <= {"git.patch", "provenance.json"}
    assert "provenance.json" in artifact_paths
    assert [item.path for item in client.list_artifacts(source.info.run_id)] == ["model"]
