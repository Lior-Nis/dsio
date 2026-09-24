"""Validate one exact native MLflow Run as reusable immutable evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, NoReturn

from mlflow import MlflowClient
from mlflow.entities import Run

from dsio.config.components import ComponentError, validate_component_config
from dsio.contracts import NonCanonicalValueError, sha256_of, sha256_of_bytes
from dsio.tracking._lifecycle import TrackingError, is_cancellation

_IDENTITY = re.compile(r"[0-9a-f]{64}")
_IDENTITY_KEY = "dsio.execution_identity"
_PROVENANCE = "provenance.json"


class UnusableEvidence(Exception):
    """Internal distinction between a cache miss and unavailable MLflow."""


def validate_run(
    client: MlflowClient,
    run: Run,
    identity: str | None,
    required_artifacts: tuple[str, ...],
    expected_configuration: dict[str, Any],
) -> Run:
    run_id = run.info.run_id
    parameter_identity = _validate_metadata(run, identity)

    provenance = _read_provenance(client, run_id)
    if provenance.get("execution_identity") != parameter_identity:
        raise UnusableEvidence("provenance identity does not match its parameter")
    schema_version = provenance.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise UnusableEvidence("provenance schema version is not supported")
    expected_keys = {
        "schema_version",
        "dsio_version",
        "configuration",
        "components",
        "execution_identity",
    }
    if schema_version == 2:
        expected_keys.add("execution")
    if provenance.keys() != expected_keys:
        raise UnusableEvidence(f"provenance fields do not match schema version {schema_version}")
    dsio_version = provenance.get("dsio_version")
    if not isinstance(dsio_version, str) or not dsio_version:
        raise UnusableEvidence("provenance DSio version is missing")
    configuration = provenance.get("configuration")
    if not isinstance(configuration, dict):
        raise UnusableEvidence("provenance configuration is missing")
    for name, expected in expected_configuration.items():
        if name not in configuration or configuration[name] != expected:
            raise UnusableEvidence(
                f"provenance configuration field {name!r} does not match the expected value"
            )
    components = provenance.get("components")
    if not isinstance(components, dict):
        raise UnusableEvidence("provenance component references are missing")
    _validate_components(components)
    document = {
        "schema_version": provenance["schema_version"],
        "dsio_version": provenance["dsio_version"],
        "configuration": provenance["configuration"],
        "components": provenance["components"],
    }
    execution: dict[str, Any] | None = None
    if schema_version == 2:
        execution = _validate_execution(provenance.get("execution"))
        document["execution"] = execution
    try:
        computed_identity = sha256_of(document)
    except NonCanonicalValueError as error:
        raise UnusableEvidence("provenance content is not canonical") from error
    if computed_identity != parameter_identity:
        raise UnusableEvidence("provenance content does not match its identity")
    if run.data.tags.get("dsio.version") != provenance["dsio_version"]:
        raise UnusableEvidence("DSio version tag does not match provenance")
    if execution is not None:
        _validate_execution_tags(run, execution)
        git = execution["git"]
        if git["dirty"]:
            patch = _read_artifact_bytes(client, run_id, "git.patch")
            if sha256_of_bytes(patch) != git["patch_sha256"]:
                raise UnusableEvidence("Git patch digest does not match provenance")

    for path in required_artifacts:
        if not _artifact_exists(client, run_id, path):
            raise UnusableEvidence(f"required artifact {path!r} is missing")

    final = _get_run(client, run_id)
    _validate_metadata(final, identity)
    if final.data.tags.get("dsio.version") != provenance["dsio_version"]:
        raise UnusableEvidence("DSio version tag changed during validation")
    if execution is not None:
        _validate_execution_tags(final, execution)
    return final


def _validate_components(components: dict[Any, Any]) -> None:
    for name, reference in components.items():
        if not isinstance(name, str):
            raise UnusableEvidence("provenance component names must be strings")
        if isinstance(reference, str):
            continue
        try:
            validate_component_config(reference)
        except ComponentError as error:
            raise UnusableEvidence(f"provenance component {name!r} is invalid: {error}") from error


def _validate_execution(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UnusableEvidence("provenance execution context is missing")
    if set(value) != {"command", "environment", "git", "package_sha256"}:
        raise UnusableEvidence("provenance execution fields do not match schema version 2")
    command = value["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(token, str) for token in command)
    ):
        raise UnusableEvidence("provenance command must be a non-empty list of strings")
    environment = value["environment"]
    if not isinstance(environment, dict):
        raise UnusableEvidence("provenance environment is missing")
    expected_environment = {
        "python",
        "platform",
        "hostname",
        "lock_sha256",
        "torch",
        "cuda",
        "cudnn",
        "gpu",
        "extra",
    }
    if set(environment) != expected_environment:
        raise UnusableEvidence("provenance environment fields do not match schema version 2")
    if any(
        not isinstance(environment[name], str) or not environment[name]
        for name in ("python", "platform", "hostname")
    ):
        raise UnusableEvidence("provenance environment identity is invalid")
    optional_environment = expected_environment - {"python", "platform", "hostname"}
    if any(
        environment[name] is not None and not isinstance(environment[name], str)
        for name in optional_environment
    ):
        raise UnusableEvidence("provenance environment values must be strings or null")
    git = value["git"]
    if not isinstance(git, dict):
        raise UnusableEvidence("provenance Git state is missing")
    if set(git) != {"sha", "branch", "dirty", "code_hash", "patch_sha256"}:
        raise UnusableEvidence("provenance Git fields do not match schema version 2")
    if type(git["dirty"]) is not bool:
        raise UnusableEvidence("provenance Git dirty flag is invalid")
    if any(
        git[name] is not None and not isinstance(git[name], str)
        for name in ("sha", "branch", "code_hash", "patch_sha256")
    ):
        raise UnusableEvidence("provenance Git values must be strings or null")
    if git["dirty"] and (git["code_hash"] is None or git["patch_sha256"] is None):
        raise UnusableEvidence("dirty Git provenance requires code and patch digests")
    package = value["package_sha256"]
    if not isinstance(package, str) or _IDENTITY.fullmatch(package) is None:
        raise UnusableEvidence("provenance DSio package digest is invalid")
    return value


def _validate_execution_tags(run: Run, execution: dict[str, Any]) -> None:
    expected = {"dsio.package_sha256": execution["package_sha256"]}
    git = execution["git"]
    environment = execution["environment"]
    if git.get("code_hash") is not None:
        expected["dsio.code_hash"] = git["code_hash"]
    if environment.get("lock_sha256") is not None:
        expected["dsio.lock_sha256"] = environment["lock_sha256"]
    for name, value in expected.items():
        if run.data.tags.get(name) != value:
            raise UnusableEvidence(f"execution tag {name!r} does not match provenance")


def _validate_metadata(run: Run, identity: str | None) -> str:
    if run.info.lifecycle_stage != "active":
        raise UnusableEvidence(f"lifecycle stage is {run.info.lifecycle_stage!r}")
    if run.info.status != "FINISHED":
        raise UnusableEvidence(f"status is {run.info.status!r}, not 'FINISHED'")
    parameter_identity = run.data.params.get(_IDENTITY_KEY)
    if parameter_identity is None or _IDENTITY.fullmatch(parameter_identity) is None:
        raise UnusableEvidence("immutable execution identity parameter is missing or invalid")
    if identity is not None and parameter_identity != identity:
        raise UnusableEvidence("immutable execution identity parameter does not match")
    if run.data.tags.get(_IDENTITY_KEY) != parameter_identity:
        raise UnusableEvidence("execution identity tag does not match its parameter")
    return parameter_identity


def _read_provenance(client: MlflowClient, run_id: str) -> dict[str, Any]:
    provenance_info = _artifact_info(client, run_id, _PROVENANCE)
    if provenance_info is None:
        raise UnusableEvidence(f"required artifact {_PROVENANCE!r} is missing")
    if provenance_info.is_dir:
        raise UnusableEvidence(f"required artifact {_PROVENANCE!r} is not a file")
    try:
        with TemporaryDirectory(prefix="dsio-provenance-") as directory:
            path = Path(client.download_artifacts(run_id, _PROVENANCE, directory))
            try:
                value = json.loads(path.read_text())
            except (UnicodeError, ValueError, RecursionError) as error:
                raise UnusableEvidence("provenance artifact is not valid JSON") from error
    except UnusableEvidence:
        raise
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error(f"read provenance for MLflow Run {run_id!r}", error)
    if not isinstance(value, dict):
        raise UnusableEvidence("provenance artifact is not a JSON object")
    return value


def _read_artifact_bytes(client: MlflowClient, run_id: str, path: str) -> bytes:
    artifact = _artifact_info(client, run_id, path)
    if artifact is None:
        raise UnusableEvidence(f"required artifact {path!r} is missing")
    if artifact.is_dir:
        raise UnusableEvidence(f"required artifact {path!r} is not a file")
    try:
        with TemporaryDirectory(prefix="dsio-evidence-") as directory:
            local = Path(client.download_artifacts(run_id, path, directory))
            return local.read_bytes()
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error(f"read artifact {path!r} for MLflow Run {run_id!r}", error)


def _artifact_exists(client: MlflowClient, run_id: str, path: str) -> bool:
    return _artifact_info(client, run_id, path) is not None


def _artifact_info(client: MlflowClient, run_id: str, path: str) -> Any:
    parent = str(PurePosixPath(path).parent)
    try:
        entries = client.list_artifacts(run_id, None if parent == "." else parent)
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error(f"list artifacts for MLflow Run {run_id!r}", error)
    return next((entry for entry in entries if entry.path == path), None)


def _get_run(client: MlflowClient, run_id: str) -> Run:
    try:
        return client.get_run(run_id)
    except BaseException as error:  # noqa: BLE001 - preserve cancellation
        _raise_tracking_error(f"refresh MLflow Run {run_id!r}", error)


def _raise_tracking_error(operation: str, error: BaseException) -> NoReturn:
    if is_cancellation(error) or not isinstance(error, Exception):
        raise error
    raise TrackingError(f"Could not {operation}: {error}") from error
