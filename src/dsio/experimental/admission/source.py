"""Source-file admission policy and recursive experimental dependency audit."""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from pathlib import Path

from dsio.experimental.admission import syntax

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
_EXPERIMENTAL_ROOT = Path(__file__).parents[1]


def audit_source(
    path: str | Path,
    *,
    module: str,
    project_names: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return deterministic rule messages for one proposed component module."""
    if isinstance(project_names, str):
        project_names = (project_names,)
    return _audit_source(
        Path(path), module=module, project_names=project_names, seen_modules=set()
    )


def _audit_source(
    source_path: Path,
    *,
    module: str,
    project_names: Sequence[str],
    seen_modules: set[str],
) -> tuple[str, ...]:
    failures: list[str] = []
    if module in seen_modules:
        return ()
    seen_modules.add(module)
    if module != "dsio.experimental" and not module.startswith("dsio.experimental."):
        failures.append(
            f"namespace: proposed component module {module!r} must live under 'dsio.experimental'"
        )
    try:
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
    except (OSError, SyntaxError, UnicodeError) as error:
        failures.append(f"source: cannot parse {source_path}: {error}")
        return tuple(failures)

    is_package = source_path.name == "__init__.py"
    aliases = syntax.assigned_aliases(tree, syntax.aliases(tree, module, is_package=is_package))
    dynamic_imports = syntax.dynamic_imports(tree, aliases)
    imported_modules = (
        *syntax.imports(tree, module, is_package=is_package),
        *(target for target in dynamic_imports if target is not None),
    )
    if any(target is None for target in dynamic_imports):
        failures.append(
            "dependency: dynamic imports must name one literal public module for static audit"
        )
    for imported in imported_modules:
        root = imported.partition(".")[0]
        if imported.endswith(".*"):
            failures.append(
                f"stable-contract: star import {imported!r} prevents static dependency audit"
            )
        if root == "dsio" and any(part.startswith("_") for part in imported.split(".")):
            failures.append(f"stable-contract: import {imported!r} uses a private DSIO module")
        if syntax.is_registry_api(imported):
            failures.append(
                f"runtime-registration: import {imported!r} exposes a closed DSIO dispatcher"
            )
        elif imported == "dsio.config.registry" or imported.startswith(
            "dsio.config.registry."
        ):
            failures.append(
                "runtime-registration: experimental components cannot depend on the "
                "closed DSIO registry implementation"
            )
        elif root not in _PUBLIC_DEPENDENCIES and root not in _stdlib_roots():
            failures.append(
                f"dependency: import {imported!r} is not a DSIO public dependency; "
                "private consumer-project imports are forbidden"
            )

    explicit_projects = {
        name.casefold()
        for name in project_names
        if isinstance(name, str) and name
    }
    known_projects = set(_KNOWN_PROJECTS) - explicit_projects
    string_constants = syntax.string_constants(tree)
    for branch, has_project_context in syntax.project_branches(tree):
        matched = _matched_project(branch, explicit_projects, string_constants)
        if matched is None and has_project_context:
            matched = _matched_project(branch, known_projects, string_constants)
        if matched is not None:
            failures.append(f"genericity: conditional branches on consumer project {matched!r}")

    if any(
        syntax.registry_mutation(node, aliases)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ) or any(syntax.registry_assignment(node, aliases) for node in ast.walk(tree)):
        failures.append(
            "runtime-registration: component source mutates a registry; contribute "
            "through the governed DSIO dispatcher instead"
        )

    for imported in imported_modules:
        dependency = _experimental_source(imported)
        if dependency is None:
            continue
        dependency_module, dependency_path = dependency
        failures.extend(
            _audit_source(
                dependency_path,
                module=dependency_module,
                project_names=project_names,
                seen_modules=seen_modules,
            )
        )
    return tuple(dict.fromkeys(failures))


def component_source(module: str) -> Path | None:
    """Locate one experimental module without importing candidate code."""
    suffix = module.removeprefix("dsio.experimental").lstrip(".")
    if not suffix:
        return _EXPERIMENTAL_ROOT / "__init__.py"
    parts = suffix.split(".")
    if not all(part.isidentifier() for part in parts):
        return None
    candidate = _EXPERIMENTAL_ROOT.joinpath(*parts)
    package_path = candidate / "__init__.py"
    if package_path.is_file():
        return package_path
    module_path = candidate.with_suffix(".py")
    return module_path if module_path.is_file() else None


def _experimental_source(imported: str) -> tuple[str, Path] | None:
    candidate = imported
    while candidate == "dsio.experimental" or candidate.startswith("dsio.experimental."):
        path = component_source(candidate)
        if path is not None:
            return candidate, path
        candidate = candidate.rpartition(".")[0]
    return None


def _matched_project(
    branch: ast.AST, projects: set[str], string_constants: dict[str, str]
) -> str | None:
    values = [
        node.value.casefold()
        for node in ast.walk(branch)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    values.extend(
        string_constants[node.id].casefold()
        for node in ast.walk(branch)
        if isinstance(node, ast.Name) and node.id in string_constants
    )
    return next(
        (
            project
            for value in values
            for project in sorted(projects)
            if re.search(rf"\b{re.escape(project)}\b", value)
        ),
        None,
    )


def _stdlib_roots() -> frozenset[str]:
    import sys

    return frozenset(sys.stdlib_module_names)
