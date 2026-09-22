"""Static, mechanically provable checks for experimental component admission."""

from __future__ import annotations

import ast
import inspect
import re
from collections.abc import Sequence
from pathlib import Path

from dsio.config.components import ComponentError, resolve_reference

_PUBLIC_DEPENDENCIES = frozenset(
    {
        "dsio",
        "lightning",
        "mlflow",
        "numpy",
        "prefect",
        "pydantic",
        "sklearn",
        "torch",
        "torchmetrics",
        "yaml",
    }
)
_KNOWN_PROJECTS = ("pulse", "algae", "algua")


class AdmissionError(ValueError):
    """A component violates one or more experimental admission rules."""


def audit_component(
    reference: str,
    *,
    project_names: Sequence[str] = (),
) -> tuple[str, ...]:
    """Audit the source module containing one named experimental component."""
    try:
        component = resolve_reference(reference)
    except ComponentError as error:
        return (f"importability: {error}",)
    module = getattr(component, "__module__", "")
    path = inspect.getsourcefile(component)
    if path is None:
        return (f"source: component {reference!r} has no inspectable Python source",)
    return audit_source(path, module=module, project_names=project_names)


def require_admissible_component(
    reference: str,
    *,
    project_names: Sequence[str] = (),
) -> None:
    """Raise one actionable error if the named component fails static admission."""
    failures = audit_component(reference, project_names=project_names)
    if failures:
        raise AdmissionError("component admission failed:\n- " + "\n- ".join(failures))


def audit_source(
    path: str | Path,
    *,
    module: str,
    project_names: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return deterministic rule messages for one proposed component module."""
    failures: list[str] = []
    if module != "dsio.experimental" and not module.startswith("dsio.experimental."):
        failures.append(
            f"namespace: proposed component module {module!r} must live under 'dsio.experimental'"
        )
    source_path = Path(path)
    try:
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
    except (OSError, SyntaxError, UnicodeError) as error:
        failures.append(f"source: cannot parse {source_path}: {error}")
        return tuple(failures)

    for imported in _imports(tree, module):
        root = imported.partition(".")[0]
        if root == "dsio" and any(part.startswith("_") for part in imported.split(".")):
            failures.append(f"stable-contract: import {imported!r} uses a private DSIO module")
        if imported == "dsio.config.registry" or imported.startswith("dsio.config.registry."):
            failures.append(
                "runtime-registration: experimental components cannot depend on the "
                "closed DSIO registry implementation"
            )
        elif root not in _PUBLIC_DEPENDENCIES and root not in _stdlib_roots():
            failures.append(
                f"dependency: import {imported!r} is not a DSIO public dependency; "
                "private consumer-project imports are forbidden"
            )

    forbidden_projects = {
        name.casefold()
        for name in (*_KNOWN_PROJECTS, *project_names)
        if isinstance(name, str) and name
    }
    for condition in _conditions(tree):
        for value in (
            node.value
            for node in ast.walk(condition)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ):
            matched = next(
                (
                    name
                    for name in sorted(forbidden_projects)
                    if re.search(rf"\b{re.escape(name)}\b", value.casefold())
                ),
                None,
            )
            if matched is not None:
                failures.append(f"genericity: conditional branches on consumer project {matched!r}")

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if _registry_mutation(call.func):
            failures.append(
                "runtime-registration: component source mutates a registry; contribute "
                "through the governed DSIO dispatcher instead"
            )
    return tuple(dict.fromkeys(failures))


def _imports(tree: ast.AST, module: str) -> tuple[str, ...]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module.split(".")[: -node.level]
                imports.append(".".join([*base, node.module or ""]).rstrip("."))
            elif node.module:
                imports.append(node.module)
    return tuple(imports)


def _conditions(tree: ast.AST) -> tuple[ast.AST, ...]:
    return tuple(
        node.test for node in ast.walk(tree) if isinstance(node, ast.If | ast.IfExp | ast.While)
    )


def _registry_mutation(function: ast.expr) -> bool:
    if not isinstance(function, ast.Attribute):
        return False
    if function.attr == "register":
        return True
    return (
        function.attr == "add"
        and isinstance(function.value, ast.Name)
        and "registry" in function.value.id.casefold()
    )


def _stdlib_roots() -> frozenset[str]:
    import sys

    return frozenset(sys.stdlib_module_names)
