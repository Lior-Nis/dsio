"""Deterministic execution identity and safe native MLflow provenance."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from importlib.metadata import version
from typing import Any, cast

from mlflow import MlflowClient

from dsio.config.components import ComponentError, validate_component_config
from dsio.contracts import NonCanonicalValueError, canonical_json, sha256_of
from dsio.tracking._lifecycle import TrackingError, is_cancellation, require_writable_run

_SCHEMA_VERSION = 1
_SET_MARKER = "$dsio.set"


def normalize(
    config: Mapping[str, Any],
    *,
    secrets: Collection[str] = (),
    ephemeral: Collection[str] = (),
) -> dict[str, Any]:
    """Return canonical JSON-safe configuration with declared fields removed."""
    omitted = _field_names(secrets) | _field_names(ephemeral)
    filtered = _filter_fields(config, omitted)
    return cast(dict[str, Any], json.loads(canonical_json(filtered)))


def execution_identity(
    config: Mapping[str, Any],
    *,
    components: Mapping[str, str | Mapping[str, Any]] | None = None,
    secrets: Collection[str] = (),
    ephemeral: Collection[str] = (),
) -> str:
    """Hash the complete safe identity document for one experiment node."""
    return sha256_of(
        _identity_document(
            config,
            components=components,
            secrets=secrets,
            ephemeral=ephemeral,
        )
    )


def record_provenance(
    run_id: str,
    config: Mapping[str, Any],
    *,
    components: Mapping[str, str | Mapping[str, Any]] | None = None,
    secrets: Collection[str] = (),
    ephemeral: Collection[str] = (),
) -> str:
    """Record one safe identity document on an explicit native MLflow Run."""
    document = _identity_document(
        config,
        components=components,
        secrets=secrets,
        ephemeral=ephemeral,
    )
    identity = sha256_of(document)
    client = MlflowClient()
    require_writable_run(client, run_id, action="record provenance")

    try:
        client.log_param(run_id, "dsio.execution_identity", identity)
        client.log_dict(
            run_id,
            {**document, "execution_identity": identity},
            "provenance.json",
        )
        client.set_tag(run_id, "dsio.execution_identity", identity)
        client.set_tag(run_id, "dsio.version", document["dsio_version"])
    except BaseException as error:
        if is_cancellation(error) or not isinstance(error, Exception):
            error.add_note(f"Interrupted while recording provenance for MLflow Run {run_id!r}.")
            raise
        raise TrackingError(
            f"Could not record provenance for MLflow Run {run_id!r}: {error}"
        ) from error
    require_writable_run(client, run_id, action="record provenance")
    return identity


def _identity_document(
    config: Mapping[str, Any],
    *,
    components: Mapping[str, str | Mapping[str, Any]] | None,
    secrets: Collection[str],
    ephemeral: Collection[str],
) -> dict[str, Any]:
    component_references = dict(components or {})
    invalid_key_types = sorted(
        {type(name).__name__ for name in component_references if not isinstance(name, str)}
    )
    if invalid_key_types:
        types = ", ".join(invalid_key_types)
        raise NonCanonicalValueError(f"component names must be str, got: {types}")
    normalized_components: dict[str, Any] = {}
    for name, component in component_references.items():
        if isinstance(component, str):
            normalized_components[name] = component
            continue
        try:
            normalized_components[name] = validate_component_config(component)
        except ComponentError as error:
            raise NonCanonicalValueError(
                f"component reference or configuration is invalid for {name!r}: {error}"
            ) from None
    return {
        "schema_version": _SCHEMA_VERSION,
        "dsio_version": version("dsio"),
        "configuration": normalize(config, secrets=secrets, ephemeral=ephemeral),
        "components": normalize(
            normalized_components,
            secrets=secrets,
            ephemeral=ephemeral,
        ),
    }


def _filter_fields(
    value: Any,
    omitted: frozenset[str],
    active: set[int] | None = None,
) -> Any:
    if not isinstance(value, Mapping | list | tuple | set | frozenset):
        return value

    active = set() if active is None else active
    container_id = id(value)
    if container_id in active:
        raise NonCanonicalValueError("cyclic container has no canonical encoding")
    active.add(container_id)
    try:
        if isinstance(value, Mapping):
            filtered: dict[str, Any] = {}
            for key in value:
                if not isinstance(key, str):
                    raise NonCanonicalValueError(
                        f"dict keys must be str for canonical encoding, got "
                        f"{type(key).__name__}"
                    )
                if key in omitted:
                    continue
                if key == _SET_MARKER:
                    raise NonCanonicalValueError(
                        f"configuration key {_SET_MARKER!r} is reserved for canonical sets"
                    )
                filtered[key] = _filter_fields(value[key], omitted, active)
            return filtered
        if isinstance(value, list | tuple):
            return [_filter_fields(item, omitted, active) for item in value]
        items = [_filter_fields(item, omitted, active) for item in value]
        return {_SET_MARKER: sorted(items, key=canonical_json)}
    finally:
        active.remove(container_id)


def _field_names(fields: Collection[str]) -> frozenset[str]:
    if isinstance(fields, str):
        return frozenset({fields})
    values = tuple(fields)
    invalid_types = sorted(
        {type(field).__name__ for field in values if not isinstance(field, str)}
    )
    if invalid_types:
        types = ", ".join(invalid_types)
        raise NonCanonicalValueError(f"field selectors must be str, got: {types}")
    return frozenset(values)
