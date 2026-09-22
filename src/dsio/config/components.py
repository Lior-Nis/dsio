"""Named import boundaries for reusable native components.

A component configuration is deliberately just data::

    {"reference": "package.module:qualname", "parameters": {...}}

There is no registry or DSio component base class. The imported object remains an ordinary
PyTorch, Lightning, TorchMetrics, or project-owned class or factory.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from functools import partial
from typing import Any, NotRequired, TypedDict

from dsio.contracts import NonCanonicalValueError, canonical_json


class ComponentConfig(TypedDict):
    """The plain serializable shape accepted by resolution and provenance."""

    reference: str
    parameters: NotRequired[dict[str, Any]]


class ComponentError(ValueError):
    """A reusable component cannot be named, imported, configured, or validated."""


def validate_component_config(config: object) -> ComponentConfig:
    """Validate and copy one plain component mapping without importing it."""
    if not isinstance(config, Mapping):
        raise ComponentError(
            f"component configuration must be a mapping, got {type(config).__name__}"
        )
    keys = set(config)
    if keys - {"reference", "parameters"}:
        raise ComponentError("component configuration accepts only reference and parameters")
    reference = config.get("reference")
    if not isinstance(reference, str) or not reference:
        raise ComponentError("component reference must be a non-empty module:qualname string")
    _split_reference(reference)
    raw_parameters = config.get("parameters", {})
    if not isinstance(raw_parameters, Mapping):
        raise ComponentError("component parameters must be a mapping")
    parameters = dict(raw_parameters)
    try:
        canonical_json(parameters)
    except NonCanonicalValueError as error:
        raise ComponentError(f"component parameters must be canonical: {error}") from None
    return {"reference": reference, "parameters": parameters}


def resolve_component[T](
    config: object,
    *runtime_args: Any,
    expected: type[T] | tuple[type[Any], ...] | None = None,
    **runtime_parameters: Any,
) -> T:
    """Import, construct, and type-check one configured native component."""
    validated = validate_component_config(config)
    reference = validated["reference"]
    target = resolve_reference(reference)
    parameters = {**runtime_parameters, **validated["parameters"]}
    try:
        component = target(*runtime_args, **parameters)
    except Exception as error:
        raise ComponentError(f"could not construct {reference}: {error}") from error
    if expected is not None and not isinstance(component, expected):
        expected_name = _expected_name(expected)
        raise ComponentError(
            f"{reference} constructed {type(component).__name__}; expected {expected_name}"
        )
    return component


def importable_reference(value: object) -> str:
    """Return a callable's stable reference, rejecting ambiguous runtime objects."""
    if not callable(value):
        raise ComponentError(
            f"reusable component must be a named importable callable, got {type(value).__name__}"
        )
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if (
        not isinstance(module, str)
        or not module
        or module == "__main__"
        or not isinstance(qualname, str)
        or not qualname
        or qualname == "<lambda>"
        or "<locals>" in qualname
    ):
        raise ComponentError(
            "reusable component must be a named importable module:qualname; "
            "lambdas, closures, and local objects are not reproducible"
        )
    reference = f"{module}:{qualname}"
    resolved = resolve_reference(reference)
    if resolved is not value:
        raise ComponentError(
            f"{reference} does not resolve to the supplied object; use a named importable callable"
        )
    return reference


def require_importable_component(value: object, role: str) -> str:
    """Validate an injected callable or callable instance before training starts."""
    candidate = (
        value
        if inspect.isfunction(value) or inspect.isclass(value) or isinstance(value, partial)
        else type(value)
    )
    try:
        return importable_reference(candidate)
    except ComponentError as error:
        raise ComponentError(f"{role}: {error}") from None


def resolve_reference(reference: str) -> Any:
    """Import one named callable without constructing it."""
    module_name, qualname = _split_reference(reference)
    try:
        value: Any = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise ComponentError(
            f"could not import component module {module_name!r}: {error}"
        ) from None
    for part in qualname.split("."):
        try:
            value = getattr(value, part)
        except AttributeError:
            raise ComponentError(f"component reference {reference!r} does not resolve") from None
    if not callable(value):
        raise ComponentError(f"component reference {reference!r} is not callable")
    return value


def _split_reference(reference: str) -> tuple[str, str]:
    if reference.count(":") != 1:
        raise ComponentError(
            f"component reference {reference!r} must use module:qualname form"
        )
    module, qualname = reference.split(":", 1)
    if not module or not qualname or "<" in qualname or ">" in qualname:
        raise ComponentError(
            f"component reference {reference!r} must use named module:qualname form"
        )
    return module, qualname


def _expected_name(expected: type[Any] | tuple[type[Any], ...]) -> str:
    choices = expected if isinstance(expected, tuple) else (expected,)
    names = []
    for choice in choices:
        module = choice.__module__.removesuffix(".optimizer")
        names.append(f"{module}.{choice.__name__}")
    return " or ".join(names)
