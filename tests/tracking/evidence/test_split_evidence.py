"""Native MLflow split artifacts and dataset-input lineage."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from mlflow import MlflowClient
from mlflow.entities import Dataset, DatasetInput, InputTag
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

from dsio.data.adapters import TableExamples
from dsio.data.splits import generate
from dsio.data.splits.models import SplitError, SplitFile
from dsio.tracking import (
    TrackingError,
    evidence_uri,
    load_split_evidence,
    record_provenance,
    record_split_evidence,
)


def _examples(*, digest: str = "cohort-v1") -> TableExamples:
    return TableExamples(
        name="cohort",
        sample_ids=[f"sample-{index:02d}" for index in range(12)],
        groups=[f"group-{index // 2}" for index in range(12)],
        digest=digest,
    )


def _manifest(examples: TableExamples) -> SplitFile:
    return generate(
        examples,
        "group_kfold",
        name="three-fold",
        seed=7,
        parameters={"n_splits": 3},
    )


def _run(client: MlflowClient, experiment_id: str, name: str) -> str:
    run = client.create_run(experiment_id, run_name=name)
    record_provenance(run.info.run_id, {"task": name})
    return run.info.run_id


def _experiment(client: MlflowClient) -> str:
    return client.create_experiment(f"split-evidence-{uuid4().hex}")


def _recorded_source(
    tmp_path: Path,
    *,
    status: str = "FINISHED",
) -> tuple[MlflowClient, str, str, TableExamples, SplitFile, str]:
    client = MlflowClient()
    experiment_id = _experiment(client)
    source_run_id = _run(client, experiment_id, "split")
    examples = _examples()
    manifest = _manifest(examples)
    uri = record_split_evidence(
        source_run_id,
        examples,
        manifest,
        source=tmp_path / "cohort.store",
    )
    if status != "RUNNING":
        client.set_terminated(source_run_id, status)
    return client, experiment_id, source_run_id, examples, manifest, uri


def _tags(dataset_input: DatasetInput) -> dict[str, str]:
    return {tag.key: tag.value for tag in dataset_input.tags}


def test_split_evidence_round_trips_as_native_mlflow_lineage(tmp_path: Path) -> None:
    import mlflow

    client, experiment_id, source_run_id, examples, manifest, uri = _recorded_source(tmp_path)
    referenced_run_id, artifact_path = RunsArtifactRepository.parse_runs_uri(uri)
    consumer_run_id = _run(client, experiment_id, "train")

    restored = load_split_evidence(
        uri,
        examples,
        consumer_run_id=consumer_run_id,
    )

    assert restored == manifest
    assert referenced_run_id == source_run_id
    assert artifact_path == f"split-evidence/{manifest.digest}/manifest.yaml"
    source = client.get_run(source_run_id)
    consumer = client.get_run(consumer_run_id)
    assert len(source.inputs.dataset_inputs) == 1
    assert len(consumer.inputs.dataset_inputs) == 1
    source_input = source.inputs.dataset_inputs[0]
    consumer_input = consumer.inputs.dataset_inputs[0]
    assert source_input.dataset == consumer_input.dataset
    assert source_input.dataset.name == examples.name
    assert source_input.dataset.digest == examples.digest
    assert _tags(source_input) == {
        "dsio.split.manifest_digest": manifest.digest,
        "dsio.split.manifest_uri": uri,
        "mlflow.data.context": "split",
    }
    assert _tags(consumer_input) == {
        "dsio.split.manifest_digest": manifest.digest,
        "dsio.split.manifest_uri": uri,
        "dsio.split.source_run_id": source_run_id,
        "mlflow.data.context": "split_reuse",
    }
    assert [entry.path for entry in client.list_artifacts(consumer_run_id)] == ["provenance.json"]
    assert sorted(entry.path for entry in client.list_artifacts(source_run_id)) == [
        "provenance.json",
        "split-evidence",
    ]
    assert mlflow.active_run() is None


def test_recording_validates_before_writing_mlflow_evidence(tmp_path: Path) -> None:
    client = MlflowClient()
    experiment_id = _experiment(client)
    run_id = _run(client, experiment_id, "split")
    examples = _examples()
    mismatched = _examples(digest="other-dataset")

    with pytest.raises(SplitError, match="dataset digest"):
        record_split_evidence(
            run_id,
            mismatched,
            _manifest(examples),
            source=tmp_path / "cohort.store",
        )

    run = client.get_run(run_id)
    assert run.inputs.dataset_inputs == []
    assert [entry.path for entry in client.list_artifacts(run_id)] == ["provenance.json"]


@pytest.mark.parametrize("status", ["RUNNING", "FAILED", "KILLED"])
def test_reuse_rejects_a_mutable_or_unsuccessful_source_run(
    tmp_path: Path,
    status: str,
) -> None:
    client, experiment_id, source_run_id, examples, _, uri = _recorded_source(
        tmp_path,
        status=status,
    )
    consumer_run_id = _run(client, experiment_id, "train")

    with pytest.raises(TrackingError, match=status):
        load_split_evidence(
            uri,
            examples,
            consumer_run_id=consumer_run_id,
        )

    assert client.get_run(consumer_run_id).inputs.dataset_inputs == []


def test_reuse_rejects_deleted_missing_and_wrongly_addressed_evidence(tmp_path: Path) -> None:
    client, experiment_id, source_run_id, examples, manifest, uri = _recorded_source(tmp_path)
    consumer_run_id = _run(client, experiment_id, "train")

    with pytest.raises(TrackingError, match="required artifact.*missing"):
        load_split_evidence(
            evidence_uri(source_run_id, "split-evidence/missing/manifest.yaml"),
            examples,
            consumer_run_id=consumer_run_id,
        )

    wrong_path = "split-evidence/wrong/manifest.yaml"
    local = tmp_path / "manifest.yaml"
    manifest.save(local)
    client.log_artifact(source_run_id, str(local), artifact_path=str(Path(wrong_path).parent))
    with pytest.raises(TrackingError, match="does not match its content digest"):
        load_split_evidence(
            evidence_uri(source_run_id, wrong_path),
            examples,
            consumer_run_id=consumer_run_id,
        )

    client.delete_run(source_run_id)
    with pytest.raises(TrackingError, match="lifecycle stage"):
        load_split_evidence(
            uri,
            examples,
            consumer_run_id=consumer_run_id,
        )


def test_reuse_rejects_corrupt_or_dataset_mismatched_manifests(tmp_path: Path) -> None:
    client, experiment_id, source_run_id, examples, _, uri = _recorded_source(tmp_path)
    _, artifact_path = RunsArtifactRepository.parse_runs_uri(uri)
    consumer_run_id = _run(client, experiment_id, "train")

    with pytest.raises(SplitError, match="dataset digest"):
        load_split_evidence(
            uri,
            _examples(digest="changed"),
            consumer_run_id=consumer_run_id,
        )

    client.log_text(source_run_id, "not: [valid", artifact_path)
    with pytest.raises(SplitError, match="cannot read split manifest"):
        load_split_evidence(
            uri,
            examples,
            consumer_run_id=consumer_run_id,
        )
    assert client.get_run(consumer_run_id).inputs.dataset_inputs == []


def test_reuse_requires_matching_native_dataset_input(tmp_path: Path) -> None:
    client = MlflowClient()
    experiment_id = _experiment(client)
    source_run_id = _run(client, experiment_id, "split")
    examples = _examples()
    manifest = _manifest(examples)
    artifact_path = f"split-evidence/{manifest.digest}/manifest.yaml"
    local = tmp_path / "manifest.yaml"
    manifest.save(local)
    client.log_artifact(source_run_id, str(local), artifact_path=str(Path(artifact_path).parent))
    client.set_terminated(source_run_id, "FINISHED")
    consumer_run_id = _run(client, experiment_id, "train")
    uri = evidence_uri(source_run_id, artifact_path)

    with pytest.raises(TrackingError, match="no native split dataset input"):
        load_split_evidence(
            uri,
            examples,
            consumer_run_id=consumer_run_id,
        )


@pytest.mark.parametrize("mismatch", ["dataset", "manifest_tag"])
def test_reuse_rejects_mismatched_native_dataset_lineage(
    tmp_path: Path,
    mismatch: str,
) -> None:
    client = MlflowClient()
    experiment_id = _experiment(client)
    source_run_id = _run(client, experiment_id, "split")
    examples = _examples()
    manifest = _manifest(examples)
    artifact_path = f"split-evidence/{manifest.digest}/manifest.yaml"
    uri = evidence_uri(source_run_id, artifact_path)
    local = tmp_path / "manifest.yaml"
    manifest.save(local)
    client.log_artifact(source_run_id, str(local), artifact_path=str(Path(artifact_path).parent))
    dataset = Dataset(
        name=examples.name,
        digest="wrong" if mismatch == "dataset" else examples.digest,
        source_type="local",
        source='{"uri": "cohort.store"}',
    )
    client.log_inputs(
        source_run_id,
        datasets=[
            DatasetInput(
                dataset,
                [
                    InputTag("mlflow.data.context", "split"),
                    InputTag("dsio.split.manifest_uri", uri),
                    InputTag(
                        "dsio.split.manifest_digest",
                        "wrong" if mismatch == "manifest_tag" else manifest.digest,
                    ),
                ],
            )
        ],
    )
    client.set_terminated(source_run_id, "FINISHED")
    consumer_run_id = _run(client, experiment_id, "train")

    message = "manifest digest" if mismatch == "manifest_tag" else "dataset identity"
    with pytest.raises(TrackingError, match=message):
        load_split_evidence(
            uri,
            examples,
            consumer_run_id=consumer_run_id,
        )
    assert client.get_run(consumer_run_id).inputs.dataset_inputs == []


def test_reuse_rejects_a_non_running_consumer_before_logging_lineage(tmp_path: Path) -> None:
    client, experiment_id, _, examples, _, uri = _recorded_source(tmp_path)
    consumer_run_id = _run(client, experiment_id, "train")
    client.set_terminated(consumer_run_id, "FAILED")

    with pytest.raises(TrackingError, match="must be RUNNING.*FAILED"):
        load_split_evidence(
            uri,
            examples,
            consumer_run_id=consumer_run_id,
        )


@pytest.mark.parametrize(
    "uri",
    [
        "models:/split@champion",
        "runs:/short/split.yaml",
        f"runs:/{'a' * 32}",
        f"runs:/{'a' * 32}/../split.yaml",
    ],
)
def test_reuse_accepts_only_exact_immutable_runs_artifact_uris(uri: str) -> None:
    with pytest.raises(TrackingError, match="immutable|artifact path|relative POSIX"):
        load_split_evidence(
            uri,
            _examples(),
            consumer_run_id="b" * 32,
        )


def test_mlflow_client_failure_is_actionable_and_creates_no_implicit_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlflow

    import dsio.tracking.evidence.splits as split_evidence

    class UnavailableClient:
        def __init__(self) -> None:
            raise RuntimeError("tracking unavailable")

    monkeypatch.setattr(split_evidence, "MlflowClient", UnavailableClient)

    with pytest.raises(TrackingError, match="tracking unavailable"):
        record_split_evidence(
            "a" * 32,
            _examples(),
            _manifest(_examples()),
            source="cohort.store",
        )
    assert mlflow.active_run() is None


def test_partial_mlflow_write_fails_and_cannot_masquerade_as_complete_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dsio.tracking.evidence.splits as split_evidence

    real_client = MlflowClient()
    experiment_id = _experiment(real_client)
    run_id = _run(real_client, experiment_id, "split")

    class FailingInputsClient:
        def __getattr__(self, name: str) -> object:
            return getattr(real_client, name)

        def log_inputs(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("dataset lineage unavailable")

    monkeypatch.setattr(split_evidence, "MlflowClient", FailingInputsClient)

    with pytest.raises(TrackingError, match="dataset lineage unavailable"):
        record_split_evidence(
            run_id,
            _examples(),
            _manifest(_examples()),
            source=tmp_path / "cohort.store",
        )

    current = real_client.get_run(run_id)
    assert current.inputs.dataset_inputs == []
    assert sorted(entry.path for entry in real_client.list_artifacts(run_id)) == [
        "provenance.json",
        "split-evidence",
    ]
