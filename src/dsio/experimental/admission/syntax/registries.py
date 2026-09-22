"""Closed-dispatcher and runtime-registration checks."""

from __future__ import annotations

import ast

from dsio.experimental.admission.syntax.imports import expression_name

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


def registry_reference(node: ast.AST, known_aliases: dict[str, str]) -> bool:
    if isinstance(node, ast.Name | ast.Attribute | ast.Subscript | ast.Call):
        name = expression_name(node, known_aliases)
        if name in _REGISTRATION_CALLABLES or _is_registry_receiver(name):
            return True
    if isinstance(node, ast.Call) and expression_name(node.func, known_aliases) == "getattr":
        return bool(node.args) and expression_name(node.args[0], known_aliases).startswith("dsio.")
    if isinstance(node, ast.Subscript):
        return expression_name(node.value, known_aliases).startswith("dsio.") and isinstance(
            node.value, ast.Attribute
        ) and node.value.attr == "__dict__"
    return False


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


def defines_registration_surface(tree: ast.AST) -> bool:
    if any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.casefold() in {"register", "register_component", "register_plugin"}
        for node in ast.walk(tree)
    ):
        return True
    if not isinstance(tree, ast.Module):
        return False
    tables = {
        target.id
        for statement in tree.body
        if isinstance(statement, ast.Assign | ast.AnnAssign)
        and statement.value is not None
        and _mutable_mapping(statement.value)
        for target in (
            statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        )
        if isinstance(target, ast.Name)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign | ast.Delete):
            targets = (
                node.targets
                if isinstance(node, ast.Assign | ast.Delete)
                else [node.target]
            )
            if any(_mutates_table(target, tables) for target in targets):
                return True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and expression_name(node.func.value, {}).partition(".")[0] in tables
            and node.func.attr in {"clear", "pop", "setdefault", "update"}
        ):
            return True
    return False


def _is_registry_receiver(name: str) -> bool:
    parts = name.split(".")
    registry_names = {registry.rpartition(".")[2] for registry in _REGISTRIES}
    return any(
        name == registry or name.startswith(f"{registry}.") for registry in _REGISTRIES
    ) or any(part in registry_names and name.startswith("dsio.") for part in parts)


def _mutable_mapping(value: ast.expr) -> bool:
    return isinstance(value, ast.Dict) or (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"dict", "defaultdict"}
    )


def _mutates_table(target: ast.expr, tables: set[str]) -> bool:
    return isinstance(target, ast.Subscript) and expression_name(target.value, {}).partition(".")[
        0
    ] in tables
