"""Shared component configuration primitives for concrete training runners."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from dsio.contracts import DsioModel


class Component(DsioModel):
    """A registered component plus its keyword arguments.

    Name and parameters travel together so a config records exactly what was built. The
    alternative -- a name here and a parameter block somewhere else -- is how a config
    directory fills with files that differ only in one exponent.
    """

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


def build_optional_component(component: Component | None, registry: Any) -> Any:
    return None if component is None else registry.get(component.name)(**component.params)


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
