"""Native MLflow persistence and reuse for governed split manifests."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NoReturn

from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.meta_dataset import MetaDataset
from mlflow.data.sources import LocalArtifactDatasetSource  # type: ignore[attr-defined]
from mlflow.entities import Dataset, DatasetInput, InputTag, Run

from dsio.data.examples import Examples
from dsio.data.splits.models import SplitError, SplitFile
from dsio.data.splits.validation import validate
from dsio.tracking._lifecycle import (
    TrackingError,
    is_cancellation,
    require_writable_run,
)
from dsio.tracking.evidence.references import (
    evidence_uri,
    require_run_id,
    validate_artifact_path,
)
from dsio.tracking.evidence.resolution import require_evidence

_ARTIFACT_ROOT = "split-evidence"
_CONTEXT = "mlflow.data.context"
_MANIFEST_URI = "dsio.split.manifest_uri"
_MANIFEST_DIGEST = "dsio.split.manifest_digest"
_DATASET_DIGEST = "dsio.dataset.digest"
_SOURCE_RUN = "dsio.split.source_run_id"
_MLFLOW_DIGEST_MAX_LENGTH = 36


def record_split_evidence(
    run_id: str,
    examples: Examples,
    manifest: SplitFile,
    *,
    source: DatasetSource | Path | str,
) -> str:
    """Log one native dataset input and its content-addressed split manifest."""
    validate(examples, manifest)
    client = _client("record split evidence")
    run = require_writable_run(client, run_id, action="record split evidence")
    artifact_path = _artifact_path(manifest.digest)
    uri = evidence_uri(run_id, artifact_path)
    dataset = _dataset_entity(examples, source)
    dataset_input = _dataset_input(
        dataset,
        context="split",
        manifest_uri=uri,
        manifest_digest=manifest.digest,
        dataset_digest=examples.digest,
    )
    already_recorded = _preflight_input(run, dataset_input)

    try:
        with TemporaryDirectory(prefix="dsio-split-evidence-") as directory:
            local = Path(directory) / "manifest.yaml"
            manifest.save(local)
            client.log_artifact(run_id, str(local), artifact_path=str(Path(artifact_path).parent))
        if not already_recorded:
            client.log_inputs(run_id, datasets=[dataset_input])
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error(f"record split evidence on MLflow Run {run_id!r}", error)
    current = require_writable_run(client, run_id, action="record split evidence")
    _require_persisted_input(current, dataset_input)
    return uri


def canonical_dataset_digest(dataset_input: DatasetInput) -> str:
    """Return DSIO's full digest while validating MLflow's shortened identity."""
    tags = _input_tags(dataset_input)
    digest = tags.get(_DATASET_DIGEST, dataset_input.dataset.digest)
    if dataset_input.dataset.digest != _mlflow_digest(digest):
        raise TrackingError(
            "Native MLflow dataset identity does not match its canonical DSIO digest."
        )
    return digest


def load_split_evidence(
    manifest_uri: str,
    examples: Examples,
    *,
    consumer_run_id: str,
) -> SplitFile:
    """Verify a split artifact and link its native dataset into a consumer Run."""
    source_run_id, path = _parse_evidence_uri(manifest_uri)
    client = _client("load split evidence")
    require_writable_run(client, consumer_run_id, action="record reused split lineage")
    source_run = require_evidence(source_run_id, required_artifacts={path})
    manifest = _download_manifest(client, source_run_id, path)
    expected_path = _artifact_path(manifest.digest)
    if path != expected_path:
        raise TrackingError(
            f"Split manifest artifact path {path!r} does not match its content digest; "
            f"expected {expected_path!r}."
        )
    validate(examples, manifest)
    uri = evidence_uri(source_run_id, path)
    _source_dataset(source_run, manifest, uri)
    refreshed_source = require_evidence(source_run_id, required_artifacts={path})
    dataset = _source_dataset(refreshed_source, manifest, uri)
    consumer_input = _dataset_input(
        dataset,
        context="split_reuse",
        manifest_uri=uri,
        manifest_digest=manifest.digest,
        dataset_digest=examples.digest,
        source_run_id=source_run_id,
    )
    consumer = require_writable_run(
        client,
        consumer_run_id,
        action="record reused split lineage",
    )
    already_linked = _preflight_input(consumer, consumer_input)
    try:
        if not already_linked:
            client.log_inputs(consumer_run_id, datasets=[consumer_input])
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error(f"record split lineage on MLflow Run {consumer_run_id!r}", error)
    current = require_writable_run(client, consumer_run_id, action="record reused split lineage")
    _require_persisted_input(current, consumer_input)
    final_source = require_evidence(source_run_id, required_artifacts={path})
    _source_dataset(final_source, manifest, uri)
    return manifest


def _artifact_path(digest: str) -> str:
    return f"{_ARTIFACT_ROOT}/{digest}/manifest.yaml"


def _parse_evidence_uri(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("runs:/"):
        raise TrackingError("Split evidence requires an immutable MLflow runs:/ artifact URI.")
    run_id, separator, path = uri.removeprefix("runs:/").partition("/")
    if not separator:
        raise TrackingError("Split evidence URI must include a relative artifact path.")
    require_run_id(run_id)
    return run_id, validate_artifact_path(path)


def _dataset_entity(
    examples: Examples,
    source: DatasetSource | Path | str,
) -> Dataset:
    try:
        if isinstance(source, Path):
            resolved_source: DatasetSource = LocalArtifactDatasetSource(str(source.resolve()))
        elif isinstance(source, str):
            if not source:
                raise TrackingError("Split dataset source must not be empty.")
            resolved_source = LocalArtifactDatasetSource(source)
        elif isinstance(source, DatasetSource):
            resolved_source = source
        else:
            raise TrackingError(
                f"Split dataset source must be a native MLflow DatasetSource or path, got "
                f"{type(source).__name__}."
            )
        config = MetaDataset(  # type: ignore[abstract]
            source=resolved_source,
            name=examples.name,
            digest=_mlflow_digest(examples.digest),
        ).to_dict()
        return Dataset(
            name=config["name"],
            digest=config["digest"],
            source_type=config["source_type"],
            source=config["source"],
            schema=config.get("schema"),
            profile=config.get("profile"),
        )
    except TrackingError:
        raise
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error("construct native MLflow split dataset", error)


def _dataset_input(
    dataset: Dataset,
    *,
    context: str,
    manifest_uri: str,
    manifest_digest: str,
    dataset_digest: str,
    source_run_id: str | None = None,
) -> DatasetInput:
    tags = [
        InputTag(_CONTEXT, context),
        InputTag(_MANIFEST_URI, manifest_uri),
        InputTag(_MANIFEST_DIGEST, manifest_digest),
        InputTag(_DATASET_DIGEST, dataset_digest),
    ]
    if source_run_id is not None:
        tags.append(InputTag(_SOURCE_RUN, source_run_id))
    return DatasetInput(dataset=dataset, tags=tags)


def _download_manifest(
    client: MlflowClient,
    run_id: str,
    artifact_path: str,
) -> SplitFile:
    try:
        with TemporaryDirectory(prefix="dsio-split-reuse-") as directory:
            local = Path(client.download_artifacts(run_id, artifact_path, directory))
            if not local.is_file():
                raise TrackingError(
                    f"Split manifest artifact {artifact_path!r} on MLflow Run "
                    f"{run_id!r} is not a file."
                )
            return SplitFile.load(local)
    except (SplitError, TrackingError):
        raise
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error(
            f"download split manifest {artifact_path!r} from MLflow Run {run_id!r}",
            error,
        )


def _source_dataset(run: Run, manifest: SplitFile, manifest_uri: str) -> Dataset:
    dataset_digest = manifest.store_manifest_sha256
    if dataset_digest is None:
        raise TrackingError("Split manifest has no dataset digest.")
    inputs = run.inputs
    if inputs is None:
        raise TrackingError(f"MLflow Run {run.info.run_id!r} has no native dataset inputs.")
    matches: list[DatasetInput] = []
    for dataset_input in inputs.dataset_inputs:
        tags = _input_tags(dataset_input)
        if tags.get(_CONTEXT) != "split" or tags.get(_MANIFEST_URI) != manifest_uri:
            continue
        matches.append(dataset_input)
        if tags.get(_MANIFEST_DIGEST) != manifest.digest:
            raise TrackingError("Source dataset input manifest digest does not match the artifact.")
        if canonical_dataset_digest(dataset_input) != dataset_digest:
            raise TrackingError("Source dataset input digest does not match the artifact.")
        if tags != {
            _CONTEXT: "split",
            _MANIFEST_URI: manifest_uri,
            _MANIFEST_DIGEST: manifest.digest,
            _DATASET_DIGEST: dataset_digest,
        }:
            raise TrackingError("Source native split dataset input tags do not match the artifact.")
        dataset = dataset_input.dataset
        if dataset.name != manifest.store:
            raise TrackingError(
                "Source native MLflow dataset identity does not match the manifest."
            )
    if not matches:
        raise TrackingError(
            f"MLflow Run {run.info.run_id!r} has no native split dataset input for "
            f"{manifest_uri!r}."
        )
    expected = matches[0]
    if any(not _same_input(candidate, expected) for candidate in matches[1:]):
        raise TrackingError("Source Run has conflicting native split dataset inputs.")
    return expected.dataset


def _preflight_input(run: Run, expected: DatasetInput) -> bool:
    """Reject MLflow's silent dataset-input deduplication before a write."""
    collisions = _colliding_inputs(run, expected)
    if not collisions:
        return False
    if all(_same_input(item, expected) for item in collisions):
        return True
    raise TrackingError(f"MLflow Run {run.info.run_id!r} already has conflicting split lineage.")


def _require_persisted_input(run: Run, expected: DatasetInput) -> None:
    collisions = _colliding_inputs(run, expected)
    if not collisions or any(not _same_input(item, expected) for item in collisions):
        raise TrackingError(
            f"MLflow did not persist the exact split lineage on Run {run.info.run_id!r}."
        )


def _colliding_inputs(run: Run, expected: DatasetInput) -> list[DatasetInput]:
    expected_uri = _input_tags(expected).get(_MANIFEST_URI)
    return [
        item
        for item in _dataset_inputs(run)
        if _same_dataset_identity(item.dataset, expected.dataset)
        or _input_tags(item).get(_MANIFEST_URI) == expected_uri
    ]


def _dataset_inputs(run: Run) -> list[DatasetInput]:
    return [] if run.inputs is None else run.inputs.dataset_inputs


def _same_input(left: DatasetInput, right: DatasetInput) -> bool:
    return _same_dataset(left.dataset, right.dataset) and _input_tags(left) == _input_tags(right)


def _same_dataset(left: Dataset, right: Dataset) -> bool:
    return (
        left.name,
        left.digest,
        left.source_type,
        left.source,
        left.schema,
        left.profile,
    ) == (
        right.name,
        right.digest,
        right.source_type,
        right.source,
        right.schema,
        right.profile,
    )


def _same_dataset_identity(left: Dataset, right: Dataset) -> bool:
    return (left.name, left.digest) == (right.name, right.digest)


def _mlflow_digest(digest: str) -> str:
    """Fit DSIO's canonical digest into MLflow's native 36-character field."""
    return digest[:_MLFLOW_DIGEST_MAX_LENGTH]


def _input_tags(dataset_input: DatasetInput) -> dict[str, str]:
    return {tag.key: tag.value for tag in dataset_input.tags}


def _client(operation: str) -> MlflowClient:
    try:
        return MlflowClient()
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error(f"create an MLflow client to {operation}", error)


def _raise_tracking_error(operation: str, error: BaseException) -> NoReturn:
    if is_cancellation(error) or not isinstance(error, Exception):
        raise error
    raise TrackingError(f"Could not {operation}: {error}") from error
