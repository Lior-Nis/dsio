"""Every public experimental object must pass the static admission audit.

Only ``calibrate_execution`` used to be audited. Location is maturity, so everything
exported from ``dsio.experimental`` and its domain packages is audited against every real
consumer project name. The synthetic ``supervised``/``self_supervised`` references are
fixtures named after generic ML terms, not consumers, and are deliberately not supplied.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

import dsio.experimental
from dsio.experimental import require_admissible_component

KAGGLE = Path(__file__).resolve().parents[2] / "reference_projects" / "kaggle"
CONSUMER_PROJECTS = tuple(
    sorted(path.name for path in KAGGLE.iterdir() if (path / "__init__.py").is_file())
)


def _public_objects() -> dict[str, list[str]]:
    """Public experimental callables, grouped by defining module."""
    by_module: dict[str, set[str]] = {}
    for module_info in pkgutil.walk_packages(
        dsio.experimental.__path__, prefix="dsio.experimental."
    ):
        if ".admission" in module_info.name:
            continue  # the auditor itself, not a component
        module = importlib.import_module(module_info.name)
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            if not (inspect.isclass(value) or inspect.isfunction(value)):
                continue  # type aliases are not components
            by_module.setdefault(value.__module__, set()).add(value.__qualname__)
    return {module: sorted(names) for module, names in sorted(by_module.items())}


PUBLIC = _public_objects()


def test_consumer_names_cover_every_kaggle_consumer() -> None:
    assert len(CONSUMER_PROJECTS) >= 8
    assert {"child_mind", "parkinsons_fog", "titanic"} <= set(CONSUMER_PROJECTS)


def test_proven_experimental_components_are_audited() -> None:
    audited = {f"{module}:{name}" for module, names in PUBLIC.items() for name in names}
    assert {
        "dsio.experimental.training:calibrate_training_execution",
        "dsio.experimental.telemetry:measure_phase",
        "dsio.experimental.telemetry:log_phase_evidence",
    } <= audited


def test_every_experimental_domain_exports_components() -> None:
    domains = {module.split(".")[2] for module in PUBLIC}
    assert {"data", "model", "train", "inference", "eval"} <= domains


@pytest.mark.parametrize("module", PUBLIC)
def test_public_experimental_module_is_admissible(module: str) -> None:
    # The source audit covers the whole defining module, so auditing one object per module
    # is complete; every other object only needs the (cheap) namespace resolution check.
    names = PUBLIC[module]
    require_admissible_component(f"{module}:{names[0]}", project_names=CONSUMER_PROJECTS)
    for name in names[1:]:
        assert audit_component_namespace(f"{module}:{name}") == ()


def audit_component_namespace(reference: str) -> tuple[str, ...]:
    from dsio.config.components import resolve_reference

    component = resolve_reference(reference)
    module = getattr(component, "__module__", "")
    if not module.startswith("dsio.experimental."):
        return (f"namespace: {reference} resolves to {module!r}",)
    return ()
