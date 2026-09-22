"""Small AST helpers used by the experimental source audit."""

from __future__ import annotations

import ast

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


def imports(tree: ast.AST, module: str) -> tuple[str, ...]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = import_from_base(node, module)
            imported.extend(
                base if alias.name == "*" else f"{base}.{alias.name}".strip(".")
                for alias in node.names
            )
    return tuple(imported)


def project_branches(tree: ast.AST) -> tuple[ast.AST, ...]:
    branches: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If | ast.IfExp | ast.While):
            branches.append(node.test)
        elif isinstance(node, ast.Match) and mentions_project(node.subject):
            branches.extend(case.pattern for case in node.cases)
            branches.extend(case.guard for case in node.cases if case.guard is not None)
        elif isinstance(node, ast.comprehension):
            branches.extend(node.ifs)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            value = node.value
            if value is not None and mentions_project(value):
                branches.append(value)
    return tuple(branches)


def mentions_project(node: ast.AST) -> bool:
    return any(
        "project" in candidate.id.casefold()
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Name)
    ) or any(
        "project" in candidate.attr.casefold()
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Attribute)
    )


def registry_mutation(function: ast.expr, known_aliases: dict[str, str]) -> bool:
    name = expression_name(function, known_aliases)
    if name in _REGISTRATION_CALLABLES:
        return True
    receiver, separator, method = name.rpartition(".")
    return bool(separator) and receiver in _REGISTRIES and method in {
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
        if receiver in _REGISTRIES:
            return True
    return False


def aliases(tree: ast.AST, module: str) -> dict[str, str]:
    known: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                known[alias.asname or alias.name.partition(".")[0]] = (
                    alias.name if alias.asname else alias.name.partition(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            base = import_from_base(node, module)
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


def dynamic_imports(tree: ast.AST, known_aliases: dict[str, str]) -> tuple[str, ...]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = expression_name(node.func, known_aliases)
        target = node.args[0]
        if function not in {"__import__", "importlib.import_module"}:
            continue
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            imported.append(target.value)
    return tuple(imported)


def import_from_base(node: ast.ImportFrom, module: str) -> str:
    if not node.level:
        return node.module or ""
    package = module.rpartition(".")[0]
    parts = package.split(".") if package else []
    remove = node.level - 1
    if remove:
        parts = parts[:-remove]
    return ".".join([*parts, node.module or ""]).rstrip(".")
