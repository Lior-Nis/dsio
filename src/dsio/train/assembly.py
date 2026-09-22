"""Shared assembly helpers for concrete training runners."""

from __future__ import annotations

from typing import Any

from dsio.config.components import (
    ComponentConfig,
    resolve_component,
    resolve_reference,
    validate_component_config,
)


def component_factory(component: ComponentConfig) -> tuple[Any, dict[str, Any]]:
    """Return the named native callable and its declared parameters."""
    validated = validate_component_config(component)
    return resolve_reference(validated["reference"]), validated["parameters"]


def build_optional_component(component: ComponentConfig | None) -> Any:
    return None if component is None else resolve_component(component)


def accepted_shape_arguments(factory: Any, shape: dict[str, Any]) -> dict[str, Any]:
    """Keep only the shape arguments this factory actually takes.

    Framework-supplied shape and user-supplied params are filtered differently on purpose.
    A backbone that pools over time is genuinely length-agnostic, and making it declare a
    ``length`` it ignores would be a lie that later reads as a constraint. User params stay
    strict and unfiltered, so a typo in a config still fails loudly instead of vanishing
    into a catch-all.
    """
    import inspect

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        return shape
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        # A factory that takes **kwargs has told us nothing, so pass everything and let it
        # decide. Registering the class itself gives a real signature and avoids this.
        return shape
    return {key: value for key, value in shape.items() if key in signature.parameters}
