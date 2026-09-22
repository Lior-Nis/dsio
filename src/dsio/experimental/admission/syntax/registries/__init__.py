"""Closed-dispatcher and runtime-registration checks."""

from dsio.experimental.admission.syntax.registries.known import (
    is_registry_api,
    registry_assignment,
    registry_mutation,
    registry_reference,
)
from dsio.experimental.admission.syntax.registries.surfaces import (
    defines_registration_surface,
)

__all__ = [
    "defines_registration_surface",
    "is_registry_api",
    "registry_assignment",
    "registry_mutation",
    "registry_reference",
]
