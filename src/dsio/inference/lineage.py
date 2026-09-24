"""Bind a checkpoint to the training inputs that produced it."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from dsio.config.components import ComponentConfig, ComponentError, validate_component_config
from dsio.inference.predictor import PredictorError
from dsio.tracking import TrackingError
from dsio.tracking.evidence.resolution import require_provenance
from dsio.tracking.execution import capture_execution
from dsio.train.artifacts import ArtifactRef


def require_checkpoint_lineage(
    checkpoint: ArtifactRef,
    *,
    training_run_id: str,
    training_identity: str,
    dataset_digest: str,
    split_digest: str,
    components: Collection[str] = (),
) -> dict[str, ComponentConfig]:
    """Verify checkpoint lineage and return requested reconstructible components."""
    if checkpoint.run_id != training_run_id:
        raise PredictorError("checkpoint evidence belongs to a different training run")
    try:
        provenance = require_provenance(
            training_run_id,
            identity=training_identity,
            required_artifacts=(checkpoint.path,),
            expected_configuration={
                "dataset_digest": dataset_digest,
                "split_digest": split_digest,
            },
        )
    except TrackingError as error:
        raise PredictorError(f"checkpoint lineage is invalid: {error}") from error
    if not components:
        return {}
    _require_compatible_execution(provenance)
    recorded = provenance["components"]
    resolved: dict[str, ComponentConfig] = {}
    for name in components:
        if not isinstance(name, str) or not name:
            raise PredictorError("checkpoint component names must be non-empty strings")
        try:
            resolved[name] = validate_component_config(recorded.get(name))
        except ComponentError as error:
            raise PredictorError(
                f"checkpoint component {name!r} is not reconstructible: {error}"
            ) from error
    return resolved


def _require_compatible_execution(provenance: Mapping[str, Any]) -> None:
    expected = provenance.get("execution")
    if not isinstance(expected, Mapping):
        raise PredictorError("checkpoint components require versioned execution evidence")
    current, _ = capture_execution()
    checks = {
        "consumer code": (expected["git"]["code_hash"], current["git"]["code_hash"]),
        "dependency lock": (
            expected["environment"]["lock_sha256"],
            current["environment"]["lock_sha256"],
        ),
        "DSIO package": (expected["package_sha256"], current["package_sha256"]),
    }
    for role, (recorded, observed) in checks.items():
        if recorded != observed:
            raise PredictorError(
                f"checkpoint {role} identity does not match the current export environment"
            )
