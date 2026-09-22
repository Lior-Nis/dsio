"""Small AST helpers used by the experimental source audit."""

from __future__ import annotations

import ast
import re

_REGISTRATION_CALLABLES = frozenset(
    {
        "dsio.eval.metrics.metric",
        "dsio.train.runner.preflight",
        "dsio.train.runner.runner",
    }
)
_REGISTRIES = frozenset(
    {
        "dsio.config.schema.TASKS",
        "dsio.eval.metrics.METRICS",
        "dsio.train.runner.PREFLIGHTS",
        "dsio.train.runner.RUNNERS",
    }
)


def imports(tree: ast.AST, module: str, *, is_package: bool = False) -> tuple[str, ...]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = import_from_base(node, module, is_package=is_package)
            imported.extend(
                f"{base}.*" if alias.name == "*" else f"{base}.{alias.name}".strip(".")
                for alias in node.names
            )
    return tuple(imported)


def project_branches(tree: ast.AST) -> tuple[tuple[ast.AST, bool], ...]:
    project_names = _project_aliases(tree)
    branches: list[tuple[ast.AST, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If | ast.IfExp | ast.While):
            branches.append((node.test, mentions_project(node.test, project_names)))
        elif isinstance(node, ast.Match):
            subject_is_project = mentions_project(node.subject, project_names)
            branches.extend((case.pattern, subject_is_project) for case in node.cases)
            branches.extend(
                (
                    case.guard,
                    subject_is_project or mentions_project(case.guard, project_names),
                )
                for case in node.cases
                if case.guard is not None
            )
        elif isinstance(node, ast.comprehension):
            branches.extend(
                (condition, mentions_project(condition, project_names))
                for condition in node.ifs
            )
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            value = node.value
            if value is not None:
                branches.append((value, mentions_project(value, project_names)))
    return tuple(branches)


def mentions_project(node: ast.AST, aliases: set[str] | None = None) -> bool:
    aliases = aliases or set()
    return any(
        candidate.id in aliases or _project_identifier(candidate.id)
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Name)
    ) or any(
        _project_identifier(candidate.attr)
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Attribute)
    )


def registry_mutation(call: ast.Call, known_aliases: dict[str, str]) -> bool:
    name = expression_name(call.func, known_aliases)
    if name in _REGISTRATION_CALLABLES:
        return True
    if name in {"setattr", "delattr"} and call.args:
        return _is_registry_receiver(expression_name(call.args[0], known_aliases))
    receiver, separator, method = name.rpartition(".")
    return bool(separator) and _is_registry_receiver(receiver) and method in {
        "add",
        "clear",
        "pop",
        "register",
        "setdefault",
        "update",
    }


def registry_assignment(node: ast.AST, known_aliases: dict[str, str]) -> bool:
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign | ast.AugAssign):
        targets.append(node.target)
    elif isinstance(node, ast.Delete):
        targets.extend(node.targets)
    for target in targets:
        if isinstance(target, ast.Subscript | ast.Attribute):
            receiver = expression_name(target.value, known_aliases)
        else:
            continue
        if _is_registry_receiver(receiver):
            return True
    return False


def is_registry_api(imported: str) -> bool:
    terminal = imported.rpartition(".")[2]
    registry_names = {registry.rpartition(".")[2] for registry in _REGISTRIES}
    return (
        imported in _REGISTRATION_CALLABLES
        or imported in _REGISTRIES
        or (imported.startswith("dsio.") and terminal in registry_names)
    )


def aliases(
    tree: ast.AST, module: str, *, is_package: bool = False
) -> dict[str, str]:
    known: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                known[alias.asname or alias.name.partition(".")[0]] = (
                    alias.name if alias.asname else alias.name.partition(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            base = import_from_base(node, module, is_package=is_package)
            for alias in node.names:
                if alias.name != "*":
                    known[alias.asname or alias.name] = f"{base}.{alias.name}".strip(".")
    return known


def assigned_aliases(tree: ast.AST, known_aliases: dict[str, str]) -> dict[str, str]:
    resolved = dict(known_aliases)
    assignments: list[tuple[str, ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            assignments.extend(
                (target.id, node.value) for target in node.targets if isinstance(target, ast.Name)
            )
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            assignments.append((node.target.id, node.value))
    for _ in assignments:
        changed = False
        for target, value in assignments:
            name = expression_name(value, resolved)
            if name and resolved.get(target) != name:
                resolved[target] = name
                changed = True
        if not changed:
            break
    return resolved


def expression_name(expression: ast.expr, known_aliases: dict[str, str]) -> str:
    if isinstance(expression, ast.Name):
        return known_aliases.get(expression.id, expression.id)
    if isinstance(expression, ast.Attribute):
        parent = expression_name(expression.value, known_aliases)
        return f"{parent}.{expression.attr}" if parent else expression.attr
    return ""


def dynamic_imports(
    tree: ast.AST, known_aliases: dict[str, str]
) -> tuple[str | None, ...]:
    imported: list[str | None] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = expression_name(node.func, known_aliases)
        if function not in {"__import__", "importlib.import_module"}:
            continue
        target = node.args[0] if node.args else next(
            (keyword.value for keyword in node.keywords if keyword.arg == "name"), None
        )
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            imported.append(target.value)
        else:
            imported.append(None)
    return tuple(imported)


def import_from_base(
    node: ast.ImportFrom, module: str, *, is_package: bool = False
) -> str:
    if not node.level:
        return node.module or ""
    package = module if is_package else module.rpartition(".")[0]
    parts = package.split(".") if package else []
    remove = node.level - 1
    if remove:
        parts = parts[:-remove]
    return ".".join([*parts, node.module or ""]).rstrip(".")


def string_constants(tree: ast.AST) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        constants.update(
            (target.id, node.value.value) for target in targets if isinstance(target, ast.Name)
        )
    return constants


def _project_aliases(tree: ast.AST) -> set[str]:
    aliases: set[str] = set()
    assignments = [
        (target.id, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    ]
    for _ in assignments:
        changed = False
        for target, value in assignments:
            if target not in aliases and mentions_project(value, aliases):
                aliases.add(target)
                changed = True
        if not changed:
            break
    return aliases


def _project_identifier(name: str) -> bool:
    return "project" in re.split(r"_+", name.casefold())


def _is_registry_receiver(name: str) -> bool:
    parts = name.split(".")
    return any(
        name == registry or name.startswith(f"{registry}.") for registry in _REGISTRIES
    ) or any(part.casefold() == "registry" for part in parts)
