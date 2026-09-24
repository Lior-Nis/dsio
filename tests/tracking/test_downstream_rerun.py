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
        record_provenance,
        require_evidence,
        resolve_experiment,
    )

    client = MlflowClient()
    experiment_id = client.create_experiment("downstream-source")
    source = client.create_run(experiment_id)
    source_identity = record_provenance(source.info.run_id, {"dataset": "algae-v1"})
    client.log_dict(source.info.run_id, {"weights": [1, 2]}, "model/model.json")
    client.log_dict(source.info.run_id, {"samples": ["a", "b"]}, "data/dataset.json")
    client.set_terminated(source.info.run_id, "FINISHED")
    failed_source = client.create_run(experiment_id)
    assert record_provenance(failed_source.info.run_id, {"dataset": "algae-v1"}) == source_identity
    client.set_terminated(failed_source.info.run_id, "FAILED")
    incomplete_source = client.create_run(experiment_id)
    assert (
        record_provenance(incomplete_source.info.run_id, {"dataset": "algae-v1"}) == source_identity
    )
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
                        client.download_artifacts(source.info.run_id, item.path, destination)
                    )
                    artifacts[item.path] = path.read_bytes()
        return metadata, artifacts

    before_source = source_snapshot()
    computations: list[float] = []

    @task
    def evaluate(
        experiment_id: str,
        source_run_id: str,
        model_uri: str,
        dataset_uri: str,
        threshold: float,
    ) -> tuple[str, str]:
        computations.append(threshold)
        with attempt(experiment_id) as run:
            identity = record_provenance(
                run.info.run_id,
                {
                    "source_run_id": source_run_id,
                    "model_uri": model_uri,
                    "dataset_uri": dataset_uri,
                    "threshold": threshold,
                },
            )
            MlflowClient().log_dict(
                run.info.run_id,
                {"score": threshold + 0.5},
                "evaluation/result.json",
            )
            return run.info.run_id, identity

    @flow
    def downstream_only(threshold: float, run_id: str) -> tuple[str, str, str]:
        resolved = resolve_experiment("downstream-rerun")
        verified = require_evidence(
            run_id,
            identity=source_identity,
            required_artifacts={"model/model.json", "data/dataset.json"},
        )
        model_uri = evidence_uri(verified.info.run_id, "model/model.json")
        dataset_uri = evidence_uri(verified.info.run_id, "data/dataset.json")
        attempt_run_id, identity = evaluate(
            resolved.experiment_id,
            verified.info.run_id,
            model_uri,
            dataset_uri,
            threshold,
        )
        return resolved.experiment_id, attempt_run_id, identity

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
    assert len(downstream_runs) == 3
    assert [run.info.status for run in downstream_runs] == ["FINISHED"] * 3
    assert all("mlflow.parentRunId" not in run.data.tags for run in downstream_runs)
    assert first[0] == second[0] == repeated[0] == downstream_experiment.experiment_id
    assert first[1] != second[1]
    assert first[2] != second[2]
    assert repeated[1] not in {first[1], second[1]}
    assert repeated[2] == first[2]
    assert (
        len(
            {
                client.get_run(result[1]).data.tags["dsio.prefect.flow_run_id"]
                for result in (first, second, repeated)
            }
        )
        == 3
    )
    result_payloads: list[bytes] = []
    for (_, attempt_run_id, _), threshold in zip(
        (first, second, repeated),
        (0.4, 0.7, 0.4),
        strict=True,
    ):
        run = client.get_run(attempt_run_id)
        assert run.info.status == "FINISHED"
        assert client.list_artifacts(attempt_run_id, "evaluation")[0].path == (
            "evaluation/result.json"
        )
        result = Path(client.download_artifacts(attempt_run_id, "evaluation/result.json", tmp_path))
        result_payloads.append(result.read_bytes())
        assert json.loads(result.read_text()) == {"score": threshold + 0.5}
        provenance = Path(client.download_artifacts(attempt_run_id, "provenance.json", tmp_path))
        assert json.loads(provenance.read_text())["configuration"] == {
            "dataset_uri": evidence_uri(source.info.run_id, "data/dataset.json"),
            "model_uri": evidence_uri(source.info.run_id, "model/model.json"),
            "source_run_id": source.info.run_id,
            "threshold": threshold,
        }
    assert result_payloads[0] != result_payloads[1]
    assert result_payloads[0] == result_payloads[2]
    assert source_snapshot() == before_source
