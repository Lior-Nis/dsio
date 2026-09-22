"""Static, mechanically provable checks for experimental component admission."""

from __future__ import annotations

from collections.abc import Sequence

from dsio.config.components import (
    ComponentError,
    resolve_reference,
    validate_component_config,
)
from dsio.experimental.admission.source import audit_source, component_source


class AdmissionError(ValueError):
    """A component violates one or more experimental admission rules."""


def audit_component(
    reference: str,
    *,
    project_names: Sequence[str] = (),
) -> tuple[str, ...]:
    """Audit one named component before importing any candidate code."""
    try:
        validated = validate_component_config({"reference": reference})
    except ComponentError as error:
        return (f"importability: {error}",)
    module = validated["reference"].partition(":")[0]
    if module != "dsio.experimental" and not module.startswith("dsio.experimental."):
        return (
            f"namespace: proposed component module {module!r} must live under "
            "'dsio.experimental'",
        )
    path = component_source(module)
    if path is None:
        return (f"importability: component {reference!r} has no inspectable Python source",)
    failures = audit_source(path, module=module, project_names=project_names)
    if failures:
        return failures
    try:
        component = resolve_reference(reference)
    except Exception as error:  # noqa: BLE001 - admission translates candidate import failures
        return (f"importability: could not resolve component {reference!r}: {error}",)
    component_module = getattr(component, "__module__", "")
    if component_module != "dsio.experimental" and not component_module.startswith(
        "dsio.experimental."
    ):
        return (
            f"namespace: resolved component lives in {component_module!r}, not "
            "'dsio.experimental'",
        )
    return ()


def require_admissible_component(
    reference: str,
    *,
    project_names: Sequence[str] = (),
) -> None:
    """Raise one actionable error if the named component fails static admission."""
    failures = audit_component(reference, project_names=project_names)
    if failures:
        raise AdmissionError("component admission failed:\n- " + "\n- ".join(failures))


__all__ = [
    "AdmissionError",
    "audit_component",
    "audit_source",
    "require_admissible_component",
]
