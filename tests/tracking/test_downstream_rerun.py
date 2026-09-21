"""Project-owned downstream reruns consume validated immutable evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _isolate_prefect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith("PREFECT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))
    monkeypatch.setenv("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    monkeypatch.setenv("DO_NOT_TRACK", "1")


def test_project_reruns_only_downstream_work_from_validated_source_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_prefect(tmp_path, monkeypatch)

    from mlflow import MlflowClient
    from prefect import flow, task
    from prefect.testing.utilities import prefect_test_harness

    from dsio.tracking import (
        TrackingError,
        attempt,
        evidence_uri,
        experiment,
        record_provenance,
        require_evidence,
    )

    client = MlflowClient()
    experiment_id = client.create_experiment("downstream-source")
    source = client.create_run(experiment_id)
    source_identity = record_provenance(source.info.run_id, {"dataset": "algae-v1"})
    client.log_dict(source.info.run_id, {"weights": [1, 2]}, "model/model.json")
    client.log_dict(source.info.run_id, {"samples": ["a", "b"]}, "data/dataset.json")
    client.set_terminated(source.info.run_id, "FINISHED")
    failed_source = client.create_run(experiment_id)
    assert record_provenance(
        failed_source.info.run_id, {"dataset": "algae-v1"}
    ) == source_identity
    client.set_terminated(failed_source.info.run_id, "FAILED")
    incomplete_source = client.create_run(experiment_id)
    assert record_provenance(
        incomplete_source.info.run_id, {"dataset": "algae-v1"}
    ) == source_identity
    client.set_terminated(incomplete_source.info.run_id, "FINISHED")
    incompatible_source = client.create_run(experiment_id)
    record_provenance(incompatible_source.info.run_id, {"dataset": "other"})
    client.set_terminated(incompatible_source.info.run_id, "FINISHED")

    def source_snapshot() -> tuple[dict[str, object], dict[str, bytes]]:
        run = client.get_run(source.info.run_id)
        metadata: dict[str, object] = {
            "status": run.info.status,
            "lifecycle_stage": run.info.lifecycle_stage,
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
            "metrics": dict(run.data.metrics),
        }
        artifacts: dict[str, bytes] = {}
        pending: list[str | None] = [None]
        destination = tmp_path / f"snapshot-{len(list(tmp_path.glob('snapshot-*')))}"
        destination.mkdir()
        while pending:
            parent = pending.pop()
            for item in client.list_artifacts(source.info.run_id, parent):
                if item.is_dir:
                    pending.append(item.path)
                else:
                    path = Path(
                        client.download_artifacts(
                            source.info.run_id, item.path, destination
                        )
                    )
                    artifacts[item.path] = path.read_bytes()
        return metadata, artifacts

    before_source = source_snapshot()
    computations: list[float] = []

    @task
    def evaluate(
        parent_run_id: str,
        source_run_id: str,
        model_uri: str,
        dataset_uri: str,
        threshold: float,
    ) -> tuple[str, str]:
        computations.append(threshold)
        with attempt(parent_run_id) as child:
            identity = record_provenance(
                child.info.run_id,
                {
                    "source_run_id": source_run_id,
                    "model_uri": model_uri,
                    "dataset_uri": dataset_uri,
                    "threshold": threshold,
                },
            )
            MlflowClient().log_dict(
                child.info.run_id,
                {"score": threshold + 0.5},
                "evaluation/result.json",
            )
            return child.info.run_id, identity

    @flow
    def downstream_only(threshold: float, run_id: str) -> tuple[str, str, str]:
        with experiment("downstream-rerun") as parent:
            verified = require_evidence(
                run_id,
                identity=source_identity,
                required_artifacts={"model/model.json", "data/dataset.json"},
            )
            model_uri = evidence_uri(verified.info.run_id, "model/model.json")
            dataset_uri = evidence_uri(verified.info.run_id, "data/dataset.json")
            child_run_id, identity = evaluate(
                parent.info.run_id,
                verified.info.run_id,
                model_uri,
                dataset_uri,
                threshold,
            )
            return parent.info.run_id, child_run_id, identity

    with prefect_test_harness():
        first = downstream_only(0.4, source.info.run_id)
        second = downstream_only(0.7, source.info.run_id)
        repeated = downstream_only(0.4, source.info.run_id)
        invalid_sources = (
            failed_source.info.run_id,
            incomplete_source.info.run_id,
            incompatible_source.info.run_id,
            "models:/predictor@champion",
            "0" * 32,
        )
        for invalid_source in invalid_sources:
            with pytest.raises(TrackingError):
                downstream_only(0.9, invalid_source)

    assert computations == [0.4, 0.7, 0.4]
    downstream_experiment = client.get_experiment_by_name("downstream-rerun")
    assert downstream_experiment is not None
    downstream_runs = client.search_runs([downstream_experiment.experiment_id])
    parents = [
        run for run in downstream_runs if "mlflow.parentRunId" not in run.data.tags
    ]
    children = [run for run in downstream_runs if "mlflow.parentRunId" in run.data.tags]
    assert sorted(run.info.status for run in parents) == [
        "FAILED",
        "FAILED",
        "FAILED",
        "FAILED",
        "FAILED",
        "FINISHED",
        "FINISHED",
        "FINISHED",
    ]
    assert [run.info.status for run in children] == ["FINISHED"] * 3
    assert first[0] != second[0]
    assert first[1] != second[1]
    assert first[2] != second[2]
    assert repeated[0] not in {first[0], second[0]}
    assert repeated[1] not in {first[1], second[1]}
    assert repeated[2] == first[2]
    result_payloads: list[bytes] = []
    for (parent_run_id, child_run_id, _), threshold in zip(
        (first, second, repeated),
        (0.4, 0.7, 0.4),
        strict=True,
    ):
        parent = client.get_run(parent_run_id)
        child = client.get_run(child_run_id)
        assert parent.info.status == child.info.status == "FINISHED"
        assert child.data.tags["mlflow.parentRunId"] == parent_run_id
        assert client.list_artifacts(child_run_id, "evaluation")[0].path == (
            "evaluation/result.json"
        )
        result = Path(
            client.download_artifacts(
                child_run_id, "evaluation/result.json", tmp_path
            )
        )
        result_payloads.append(result.read_bytes())
        assert json.loads(result.read_text()) == {"score": threshold + 0.5}
        provenance = Path(
            client.download_artifacts(child_run_id, "provenance.json", tmp_path)
        )
        assert json.loads(provenance.read_text())["configuration"] == {
            "dataset_uri": evidence_uri(source.info.run_id, "data/dataset.json"),
            "model_uri": evidence_uri(source.info.run_id, "model/model.json"),
            "source_run_id": source.info.run_id,
            "threshold": threshold,
        }
    assert result_payloads[0] != result_payloads[1]
    assert result_payloads[0] == result_payloads[2]
    assert source_snapshot() == before_source
