"""Resolve only complete, provenance-compatible native MLflow Runs."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from typing import Any, NoReturn

from mlflow import MlflowClient
from mlflow.entities import Experiment, Run, ViewType

from dsio.contracts import NonCanonicalValueError
from dsio.tracking._lifecycle import TrackingError, is_cancellation
from dsio.tracking.evidence.references import artifact_paths, require_run_id
from dsio.tracking.evidence.validation import UnusableEvidence, validate_run
from dsio.tracking.provenance import normalize

_IDENTITY = re.compile(r"[0-9a-f]{64}")
_IDENTITY_KEY = "dsio.execution_identity"


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
            return validate_run(client, current, identity, artifacts, {})[0]
        except UnusableEvidence:
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
    return _require_evidence(
        run_id,
        identity=identity,
        required_artifacts=required_artifacts,
        expected_configuration=expected_configuration,
    )[0]


def require_provenance(
    run_id: str,
    *,
    identity: str | None = None,
    required_artifacts: Collection[str] = (),
    expected_configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return provenance only after its exact MLflow Run passes evidence validation."""
    return _require_evidence(
        run_id,
        identity=identity,
        required_artifacts=required_artifacts,
        expected_configuration=expected_configuration,
    )[1]


def _require_evidence(
    run_id: str,
    *,
    identity: str | None,
    required_artifacts: Collection[str],
    expected_configuration: Mapping[str, Any] | None,
) -> tuple[Run, dict[str, Any]]:
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
        return validate_run(client, run, identity, artifacts, expected)
    except UnusableEvidence as error:
        raise TrackingError(f"MLflow Run {run_id!r} is not reusable: {error}") from error


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
