"""Resolve only complete, provenance-compatible native MLflow Runs."""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, NoReturn

from mlflow import MlflowClient
from mlflow.entities import Experiment, Run, ViewType

from dsio.contracts import NonCanonicalValueError, sha256_of
from dsio.tracking._lifecycle import TrackingError, is_cancellation
from dsio.tracking.evidence.references import artifact_paths, require_run_id
from dsio.tracking.provenance import normalize

_IDENTITY = re.compile(r"[0-9a-f]{64}")
_IDENTITY_KEY = "dsio.execution_identity"
_PROVENANCE = "provenance.json"


class _UnusableEvidence(Exception):
    """Internal distinction between a cache miss and unavailable MLflow."""


def resolve_evidence(
    identity: str,
    *,
    required_artifacts: Collection[str] = (),
) -> Run | None:
    """Return the newest verified native Run for an identity, or a cache miss."""
    _require_identity(identity)
    artifacts = artifact_paths(required_artifacts)
    client = _client()
    try:
        experiments = _all_experiments(client)
        if not experiments:
            return None
        candidates = _all_matching_runs(
            client,
            [experiment.experiment_id for experiment in experiments],
            identity,
        )
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error("query MLflow evidence", error)

    ordered = sorted(
        candidates,
        key=lambda run: (
            run.info.end_time or run.info.start_time or -1,
            run.info.run_id,
        ),
        reverse=True,
    )
    for run in ordered:
        try:
            current = _get_run(client, run.info.run_id)
            return _validate_run(client, current, identity, artifacts, {})
        except _UnusableEvidence:
            continue
    return None


def require_evidence(
    run_id: str,
    *,
    identity: str | None = None,
    required_artifacts: Collection[str] = (),
    expected_configuration: Mapping[str, Any] | None = None,
) -> Run:
    """Validate and return one exact immutable MLflow Run reference."""
    require_run_id(run_id)
    if identity is not None:
        _require_identity(identity)
    artifacts = artifact_paths(required_artifacts)
    try:
        expected = normalize(expected_configuration or {})
    except NonCanonicalValueError as error:
        raise TrackingError(f"Expected evidence configuration is invalid: {error}") from error
    client = _client()
    run = _get_run(client, run_id)
    try:
        return _validate_run(client, run, identity, artifacts, expected)
    except _UnusableEvidence as error:
        raise TrackingError(f"MLflow Run {run_id!r} is not reusable: {error}") from error


def _validate_run(
    client: MlflowClient,
    run: Run,
    identity: str | None,
    required_artifacts: tuple[str, ...],
    expected_configuration: Mapping[str, Any],
) -> Run:
    run_id = run.info.run_id
    parameter_identity = _validate_metadata(run, identity)

    provenance = _read_provenance(client, run_id)
    if provenance.get("execution_identity") != parameter_identity:
        raise _UnusableEvidence("provenance identity does not match its parameter")
    expected_keys = {
        "schema_version",
        "dsio_version",
        "configuration",
        "components",
        "execution_identity",
    }
    if provenance.keys() != expected_keys:
        raise _UnusableEvidence("provenance fields do not match schema version 1")
    schema_version = provenance.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise _UnusableEvidence("provenance schema version is not supported")
    dsio_version = provenance.get("dsio_version")
    if not isinstance(dsio_version, str) or not dsio_version:
        raise _UnusableEvidence("provenance DSio version is missing")
    configuration = provenance.get("configuration")
    if not isinstance(configuration, dict):
        raise _UnusableEvidence("provenance configuration is missing")
    for name, expected in expected_configuration.items():
        if name not in configuration or configuration[name] != expected:
            raise _UnusableEvidence(
                f"provenance configuration field {name!r} does not match the expected value"
            )
    components = provenance.get("components")
    if not isinstance(components, dict):
        raise _UnusableEvidence("provenance component references are missing")
    if any(
        not isinstance(name, str) or not isinstance(reference, str)
        for name, reference in components.items()
    ):
        raise _UnusableEvidence("provenance component references must be strings")
    document = {
        "schema_version": provenance["schema_version"],
        "dsio_version": provenance["dsio_version"],
        "configuration": provenance["configuration"],
        "components": provenance["components"],
    }
    try:
        computed_identity = sha256_of(document)
    except NonCanonicalValueError as error:
        raise _UnusableEvidence("provenance content is not canonical") from error
    if computed_identity != parameter_identity:
        raise _UnusableEvidence("provenance content does not match its identity")
    if run.data.tags.get("dsio.version") != provenance["dsio_version"]:
        raise _UnusableEvidence("DSio version tag does not match provenance")

    for path in required_artifacts:
        if not _artifact_exists(client, run_id, path):
            raise _UnusableEvidence(f"required artifact {path!r} is missing")

    final = _get_run(client, run_id)
    _validate_metadata(final, identity)
    if final.data.tags.get("dsio.version") != provenance["dsio_version"]:
        raise _UnusableEvidence("DSio version tag changed during validation")
    return final


def _validate_metadata(run: Run, identity: str | None) -> str:
    if run.info.lifecycle_stage != "active":
        raise _UnusableEvidence(f"lifecycle stage is {run.info.lifecycle_stage!r}")
    if run.info.status != "FINISHED":
        raise _UnusableEvidence(f"status is {run.info.status!r}, not 'FINISHED'")

    parameter_identity = run.data.params.get(_IDENTITY_KEY)
    if parameter_identity is None or _IDENTITY.fullmatch(parameter_identity) is None:
        raise _UnusableEvidence("immutable execution identity parameter is missing or invalid")
    if identity is not None and parameter_identity != identity:
        raise _UnusableEvidence("immutable execution identity parameter does not match")
    if run.data.tags.get(_IDENTITY_KEY) != parameter_identity:
        raise _UnusableEvidence("execution identity tag does not match its parameter")
    return parameter_identity


def _read_provenance(client: MlflowClient, run_id: str) -> dict[str, Any]:
    provenance_info = _artifact_info(client, run_id, _PROVENANCE)
    if provenance_info is None:
        raise _UnusableEvidence(f"required artifact {_PROVENANCE!r} is missing")
    if provenance_info.is_dir:
        raise _UnusableEvidence(f"required artifact {_PROVENANCE!r} is not a file")
    try:
        with TemporaryDirectory(prefix="dsio-provenance-") as directory:
            path = Path(client.download_artifacts(run_id, _PROVENANCE, directory))
            try:
                value = json.loads(path.read_text())
            except (UnicodeError, ValueError, RecursionError) as error:
                raise _UnusableEvidence("provenance artifact is not valid JSON") from error
    except _UnusableEvidence:
        raise
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error(f"read provenance for MLflow Run {run_id!r}", error)
    if not isinstance(value, dict):
        raise _UnusableEvidence("provenance artifact is not a JSON object")
    return value


def _artifact_exists(client: MlflowClient, run_id: str, path: str) -> bool:
    return _artifact_info(client, run_id, path) is not None


def _artifact_info(client: MlflowClient, run_id: str, path: str) -> Any:
    parent = str(PurePosixPath(path).parent)
    try:
        entries = client.list_artifacts(run_id, None if parent == "." else parent)
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error(f"list artifacts for MLflow Run {run_id!r}", error)
    return next((entry for entry in entries if entry.path == path), None)


def _require_identity(identity: str) -> None:
    if not isinstance(identity, str) or _IDENTITY.fullmatch(identity) is None:
        raise TrackingError("Evidence resolution requires a SHA-256 execution identity.")


def _client() -> MlflowClient:
    try:
        return MlflowClient()
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error("create an MLflow client for evidence validation", error)


def _get_run(client: MlflowClient, run_id: str) -> Run:
    try:
        return client.get_run(run_id)
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error(f"refresh MLflow Run {run_id!r}", error)


def _all_experiments(client: MlflowClient) -> list[Experiment]:
    experiments: list[Experiment] = []
    token: str | None = None
    while True:
        page = client.search_experiments(view_type=ViewType.ALL, page_token=token)
        experiments.extend(page)
        token = page.token
        if token is None:
            return experiments


def _all_matching_runs(
    client: MlflowClient,
    experiment_ids: list[str],
    identity: str,
) -> list[Run]:
    runs: list[Run] = []
    token: str | None = None
    while True:
        page = client.search_runs(
            experiment_ids,
            filter_string=f"params.`{_IDENTITY_KEY}` = '{identity}'",
            run_view_type=ViewType.ALL,
            page_token=token,
        )
        runs.extend(page)
        token = page.token
        if token is None:
            return runs


def _raise_tracking_error(operation: str, error: BaseException) -> NoReturn:
    if is_cancellation(error) or not isinstance(error, Exception):
        raise error
    raise TrackingError(f"Could not {operation}: {error}") from error
