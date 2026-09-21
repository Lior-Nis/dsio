"""Resolve only complete, provenance-compatible native MLflow evidence."""

from __future__ import annotations

import importlib
import json
from importlib.metadata import version
from pathlib import Path

import pytest


def _evidence_run(
    name: str,
    *,
    seed: int = 7,
    status: str = "FINISHED",
    artifact: str | None = "outputs/value.json",
) -> tuple[object, str]:
    from mlflow import MlflowClient

    from dsio.tracking import record_provenance

    client = MlflowClient()
    experiment_id = client.create_experiment(name)
    run = client.create_run(experiment_id)
    identity = record_provenance(run.info.run_id, {"seed": seed})
    if artifact is not None:
        client.log_dict(run.info.run_id, {"value": seed}, artifact)
    client.set_terminated(run.info.run_id, status)
    return run, identity


def test_resolve_evidence_returns_a_native_verified_run() -> None:
    from mlflow.entities import Run

    from dsio.tracking import resolve_evidence

    source, identity = _evidence_run("verified-evidence")

    resolved = resolve_evidence(identity, required_artifacts={"outputs/value.json"})

    assert isinstance(resolved, Run)
    assert resolved.info.run_id == source.info.run_id


def test_invalid_newer_candidate_does_not_shadow_an_older_valid_run() -> None:
    from dsio.tracking import resolve_evidence

    source, identity = _evidence_run("older-valid", seed=11)
    invalid, same_identity = _evidence_run("newer-invalid", seed=11, status="FAILED")
    assert same_identity == identity

    resolved = resolve_evidence(identity, required_artifacts={"outputs/value.json"})

    assert resolved is not None
    assert resolved.info.run_id == source.info.run_id
    assert resolved.info.run_id != invalid.info.run_id


def test_missing_or_incomplete_evidence_is_a_cache_miss() -> None:
    from dsio.tracking import resolve_evidence

    _, identity = _evidence_run("missing-output", artifact=None)

    assert resolve_evidence(identity, required_artifacts={"outputs/value.json"}) is None
    assert resolve_evidence("0" * 64) is None


@pytest.mark.parametrize("status", ["RUNNING", "FAILED", "KILLED"])
def test_require_evidence_rejects_non_successful_runs(status: str) -> None:
    from dsio.tracking import TrackingError, require_evidence

    run, identity = _evidence_run(f"unusable-{status.lower()}", status=status)

    with pytest.raises(TrackingError, match=status):
        require_evidence(run.info.run_id, identity=identity)


def test_require_evidence_rejects_identity_or_provenance_disagreement(
    tmp_path: Path,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, require_evidence

    run, identity = _evidence_run("mismatched-evidence")
    client = MlflowClient()
    client.set_tag(run.info.run_id, "dsio.execution_identity", "0" * 64)

    with pytest.raises(TrackingError, match="identity tag"):
        require_evidence(run.info.run_id, identity=identity)

    artifact = Path(client.download_artifacts(run.info.run_id, "provenance.json", tmp_path))
    payload = json.loads(artifact.read_text())
    assert payload["execution_identity"] == identity


def test_require_evidence_rejects_provenance_content_changed_after_recording() -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, require_evidence

    run, identity = _evidence_run("tampered-provenance")
    MlflowClient().log_dict(
        run.info.run_id,
        {
            "schema_version": 1,
            "dsio_version": "0.1.0",
            "configuration": {"seed": 999},
            "components": {},
            "execution_identity": identity,
        },
        "provenance.json",
    )

    with pytest.raises(TrackingError, match="content does not match"):
        require_evidence(run.info.run_id, identity=identity)


def test_require_evidence_rejects_dsio_version_tag_disagreement() -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, require_evidence

    run, identity = _evidence_run("mismatched-version-tag")
    MlflowClient().set_tag(run.info.run_id, "dsio.version", "different-version")

    with pytest.raises(TrackingError, match="DSio version tag"):
        require_evidence(run.info.run_id, identity=identity)


def test_require_evidence_rejects_unhashed_provenance_fields() -> None:
    from importlib.metadata import version

    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, require_evidence

    run, identity = _evidence_run("extra-provenance-field")
    MlflowClient().log_dict(
        run.info.run_id,
        {
            "schema_version": 1,
            "dsio_version": version("dsio"),
            "configuration": {"seed": 7},
            "components": {},
            "execution_identity": identity,
            "unhashed_extra": "MUTATED",
        },
        "provenance.json",
    )

    with pytest.raises(TrackingError, match="fields do not match"):
        require_evidence(run.info.run_id, identity=identity)


def test_deleted_evidence_is_not_reusable() -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, require_evidence

    run, identity = _evidence_run("deleted-evidence")
    MlflowClient().delete_run(run.info.run_id)

    with pytest.raises(TrackingError, match="lifecycle stage.*deleted"):
        require_evidence(run.info.run_id, identity=identity)


def test_artifact_store_failure_during_validation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import TrackingError, resolve_evidence

    _, identity = _evidence_run("unavailable-artifacts")
    evidence = importlib.import_module("dsio.tracking.evidence.resolution")
    real_client = MlflowClient()

    class UnavailableArtifactClient:
        def __getattr__(self, name: str) -> object:
            return getattr(real_client, name)

        def list_artifacts(self, run_id: str, path: str | None = None) -> list[object]:
            raise RuntimeError("artifact store unavailable")

    monkeypatch.setattr(evidence, "MlflowClient", UnavailableArtifactClient)

    with pytest.raises(TrackingError, match="artifact store unavailable"):
        resolve_evidence(identity)


def test_run_deleted_during_validation_is_not_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import resolve_evidence

    run, identity = _evidence_run("deleted-during-validation")
    evidence = importlib.import_module("dsio.tracking.evidence.resolution")
    real_client = MlflowClient()

    class DeleteAfterRefreshClient:
        def __init__(self) -> None:
            self.refreshes = 0

        def __getattr__(self, name: str) -> object:
            return getattr(real_client, name)

        def get_run(self, run_id: str) -> object:
            current = real_client.get_run(run_id)
            self.refreshes += 1
            if self.refreshes == 1:
                real_client.delete_run(run_id)
            return current

    monkeypatch.setattr(evidence, "MlflowClient", DeleteAfterRefreshClient)

    assert resolve_evidence(identity) is None
    assert real_client.get_run(run.info.run_id).info.lifecycle_stage == "deleted"


@pytest.mark.parametrize("malformation", ["bytes", "directory", "large-integer"])
def test_malformed_newer_provenance_does_not_shadow_older_valid_evidence(
    malformation: str,
    tmp_path: Path,
) -> None:
    from mlflow import MlflowClient

    from dsio.tracking import resolve_evidence

    source, identity = _evidence_run(f"older-valid-{malformation}", seed=23)
    client = MlflowClient()
    experiment_id = client.create_experiment(f"newer-malformed-{malformation}")
    malformed = client.create_run(experiment_id)
    client.log_param(malformed.info.run_id, "dsio.execution_identity", identity)
    client.set_tag(malformed.info.run_id, "dsio.execution_identity", identity)
    client.set_tag(malformed.info.run_id, "dsio.version", version("dsio"))
    local = tmp_path / malformation
    if malformation == "bytes":
        local.mkdir()
        provenance = local / "provenance.json"
        provenance.write_bytes(b"\xff\xfe\xfd")
        client.log_artifact(malformed.info.run_id, str(provenance))
    elif malformation == "directory":
        provenance = local / "provenance.json"
        provenance.mkdir(parents=True)
        (provenance / "unexpected.json").write_text("{}")
        client.log_artifacts(
            malformed.info.run_id,
            str(provenance),
            artifact_path="provenance.json",
        )
    else:
        local.mkdir()
        provenance = local / "provenance.json"
        provenance.write_text('{"too_large":' + "9" * 5_000 + "}")
        client.log_artifact(malformed.info.run_id, str(provenance))
    client.set_terminated(malformed.info.run_id, "FINISHED")

    resolved = resolve_evidence(identity)

    assert resolved is not None
    assert resolved.info.run_id == source.info.run_id


@pytest.mark.parametrize(
    "document",
    [
        {
            "schema_version": True,
            "dsio_version": "0.1.0",
            "configuration": {},
            "components": {},
        },
        {
            "schema_version": 1,
            "dsio_version": "",
            "configuration": {},
            "components": {},
        },
        {
            "schema_version": 1,
            "dsio_version": "0.1.0",
            "configuration": {},
            "components": {"model": 7},
        },
    ],
)
def test_schema_invalid_self_consistent_provenance_is_rejected(
    document: dict[str, object],
) -> None:
    from mlflow import MlflowClient

    from dsio.contracts import sha256_of
    from dsio.tracking import TrackingError, require_evidence

    identity = sha256_of(document)
    client = MlflowClient()
    experiment_id = client.create_experiment(f"invalid-schema-{identity[:8]}")
    run = client.create_run(experiment_id)
    client.log_param(run.info.run_id, "dsio.execution_identity", identity)
    client.set_tag(run.info.run_id, "dsio.execution_identity", identity)
    client.set_tag(run.info.run_id, "dsio.version", str(document["dsio_version"]))
    client.log_dict(
        run.info.run_id,
        {**document, "execution_identity": identity},
        "provenance.json",
    )
    client.set_terminated(run.info.run_id, "FINISHED")

    with pytest.raises(TrackingError, match="schema|version|component"):
        require_evidence(run.info.run_id, identity=identity)


def test_mlflow_query_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from dsio.tracking import TrackingError, resolve_evidence

    evidence = importlib.import_module("dsio.tracking.evidence.resolution")

    class UnavailableClient:
        def search_experiments(self, **_: object) -> list[object]:
            raise RuntimeError("tracking unavailable")

    monkeypatch.setattr(evidence, "MlflowClient", UnavailableClient)

    with pytest.raises(TrackingError, match="tracking unavailable"):
        resolve_evidence("0" * 64)


@pytest.mark.parametrize("operation", ["resolve", "require"])
def test_mlflow_client_creation_failure_fails_closed(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dsio.tracking import TrackingError, require_evidence, resolve_evidence

    evidence = importlib.import_module("dsio.tracking.evidence.resolution")

    class UnavailableClient:
        def __init__(self) -> None:
            raise RuntimeError("invalid tracking URI")

    monkeypatch.setattr(evidence, "MlflowClient", UnavailableClient)

    with pytest.raises(TrackingError, match="invalid tracking URI"):
        if operation == "resolve":
            resolve_evidence("0" * 64)
        else:
            require_evidence("a" * 32)
