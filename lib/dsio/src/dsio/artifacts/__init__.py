"""Versioned model artifacts with pinned references and fail-closed integrity."""

from dsio.artifacts.store import (
    ModelRef,
    ModelRegistry,
    ModelVersion,
    RegistryIntegrityError,
    compute_provenance_digest,
    promotion_blockers,
)

__all__ = [
    "ModelRef",
    "ModelRegistry",
    "ModelVersion",
    "RegistryIntegrityError",
    "compute_provenance_digest",
    "promotion_blockers",
]
