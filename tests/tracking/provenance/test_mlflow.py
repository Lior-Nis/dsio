"""Native MLflow persistence for deterministic safe provenance."""

from __future__ import annotations

import importlib
import json
from importlib.metadata import version
from pathlib import Path

import pytest


def _running_run(name: str) -> object:
    from mlflow import MlflowClient

    client = MlflowClient()
    experiment_id = client.create_experiment(name)
    return client.create_run(experiment_id)


def test_record_provenance_logs_one_safe_native_artifact_and_searchable_tags(
    tmp_path: Path,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import record_provenance

    run = _running_run("native-provenance")
    identity = record_provenance(
        run.info.run_id,
        {
            "dataset": "sha256:data",
            "seed": 7,
            "database": {"user": "reader", "password": "SENTINEL-PASSWORD"},
            "task_run_id": "ephemeral-id",
        },
        components={
            "model": {
                "reference": "torch.nn:Linear",
                "parameters": {
                    "in_features": 3,
                    "out_features": 2,
                    "password": "SENTINEL-COMPONENT-PASSWORD",
                    "task_run_id": "component-ephemeral-id",
                },
            }
        },
        secrets="password",
        ephemeral="task_run_id",
    )

    client = MlflowClient()
    stored = client.get_run(run.info.run_id)
    assert stored.data.tags["dsio.execution_identity"] == identity
    assert stored.data.tags["dsio.version"] == version("dsio")
    assert stored.data.params["dsio.execution_identity"] == identity
    artifact = Path(client.download_artifacts(run.info.run_id, "provenance.json", tmp_path))
    payload = json.loads(artifact.read_text())
    execution = payload["execution"]
    assert set(execution) == {"command", "environment", "git", "package_sha256"}
    assert len(execution["package_sha256"]) == 64
    assert execution["command"]
    assert execution["environment"]["python"]
    assert payload == {
        "components": {
            "model": {
                "parameters": {"in_features": 3, "out_features": 2},
                "reference": "torch.nn:Linear",
            }
        },
        "configuration": {
            "database": {"user": "reader"},
            "dataset": "sha256:data",
            "seed": 7,
        },
        "dsio_version": version("dsio"),
        "execution": execution,
        "execution_identity": identity,
        "schema_version": 2,
    }
    all_evidence = json.dumps(
        {
            "params": stored.data.params,
            "tags": stored.data.tags,
            "artifact": payload,
        },
        sort_keys=True,
    )
    assert "SENTINEL-PASSWORD" not in all_evidence
    assert "password" not in all_evidence
    assert "ephemeral-id" not in all_evidence
    assert "task_run_id" not in all_evidence


def test_record_provenance_captures_consumer_git_patch_and_root_lock(
    tmp_path: Path,
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.contracts import sha256_of_file
    from dsio.tracking import TrackingError, record_provenance, require_evidence

    (git_repo / "uv.lock").write_text("version = 1\n")
    (git_repo / "tracked.txt").write_text("dirty consumer code\n")
    nested = git_repo / "project" / "tasks"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    run = _running_run("consumer-execution-provenance")

    record_provenance(run.info.run_id, {"seed": 7})

    client = MlflowClient()
    artifact = Path(client.download_artifacts(run.info.run_id, "provenance.json", tmp_path))
    payload = json.loads(artifact.read_text())
    execution = payload["execution"]
    assert execution["git"]["dirty"] is True
    assert execution["git"]["code_hash"].startswith(execution["git"]["sha"] + "-dirty-")
    assert execution["environment"]["lock_sha256"] == sha256_of_file(
        str(git_repo / "uv.lock")
    )
    patch = Path(client.download_artifacts(run.info.run_id, "git.patch", tmp_path))
    assert b"dirty consumer code" in patch.read_bytes()
    client.set_terminated(run.info.run_id, "FINISHED")
    require_evidence(run.info.run_id)

    client.log_text(run.info.run_id, "corrupt", "git.patch")
    with pytest.raises(TrackingError, match="Git patch digest"):
        require_evidence(run.info.run_id)


@pytest.mark.parametrize("terminal_status", ["FINISHED", "FAILED", "KILLED"])
def test_record_provenance_rejects_a_terminal_run(terminal_status: str) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, record_provenance

    run = _running_run(f"terminal-provenance-{terminal_status.lower()}")
    MlflowClient().set_terminated(run.info.run_id, terminal_status)

    with pytest.raises(TrackingError, match="must be RUNNING"):
        record_provenance(run.info.run_id, {"seed": 7})


def test_recording_failure_does_not_expose_secret_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, record_provenance

    run = _running_run("failed-provenance-write")
    provenance = importlib.import_module("dsio.tracking.provenance")
    real_client = MlflowClient()

    class FailingClient:
        def __getattr__(self, name: str) -> object:
            return getattr(real_client, name)

        def log_dict(
            self,
            run_id: str,
            dictionary: dict[str, object],
            artifact_file: str,
        ) -> None:
            raise RuntimeError("artifact store unavailable")

    monkeypatch.setattr(provenance, "MlflowClient", FailingClient)

    with pytest.raises(TrackingError) as caught:
        record_provenance(
            run.info.run_id,
            {"password": "SENTINEL-PASSWORD", "seed": 7},
            secrets={"password"},
        )

    message = str(caught.value)
    assert "artifact store unavailable" in message
    assert run.info.run_id in message
    assert "SENTINEL-PASSWORD" not in message
    stored = real_client.get_run(run.info.run_id)
    assert "SENTINEL-PASSWORD" not in json.dumps(
        {"params": stored.data.params, "tags": stored.data.tags}
    )


def test_second_identity_cannot_overwrite_a_runs_first_provenance(tmp_path: Path) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, record_provenance

    run = _running_run("immutable-provenance")
    first = record_provenance(run.info.run_id, {"seed": 1})

    with pytest.raises(TrackingError):
        record_provenance(run.info.run_id, {"seed": 2})

    client = MlflowClient()
    stored = client.get_run(run.info.run_id)
    artifact = Path(client.download_artifacts(run.info.run_id, "provenance.json", tmp_path))
    payload = json.loads(artifact.read_text())
    assert stored.data.params["dsio.execution_identity"] == first
    assert stored.data.tags["dsio.execution_identity"] == first
    assert payload["execution_identity"] == first
    assert payload["configuration"] == {"seed": 1}


def test_run_that_finishes_during_persistence_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, record_provenance

    run = _running_run("terminal-during-provenance")
    provenance = importlib.import_module("dsio.tracking.provenance")
    real_client = MlflowClient()

    class FinishDuringWriteClient:
        def __getattr__(self, name: str) -> object:
            return getattr(real_client, name)

        def log_dict(
            self,
            run_id: str,
            dictionary: dict[str, object],
            artifact_file: str,
        ) -> None:
            real_client.log_dict(run_id, dictionary, artifact_file)
            real_client.set_terminated(run_id, "FINISHED")

    monkeypatch.setattr(provenance, "MlflowClient", FinishDuringWriteClient)

    with pytest.raises(TrackingError, match="must be RUNNING"):
        record_provenance(run.info.run_id, {"seed": 7})
