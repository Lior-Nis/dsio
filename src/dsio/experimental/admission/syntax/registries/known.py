"""Checks for access to DSIO's known closed dispatchers."""

from __future__ import annotations

import ast

from dsio.experimental.admission.syntax.imports import expression_name

_REGISTRATION_CALLABLES = frozenset(
    {
        "dsio.eval.metrics.metric",
    }
)
_REGISTRIES = frozenset(
    {
        "dsio.eval.metrics.METRICS",
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
        if (
            name in _REGISTRATION_CALLABLES
            or _is_registry_receiver(name)
            or (name.startswith("dsio.") and ".__dict__" in name)
        ):
            return True
    if isinstance(node, ast.Call) and expression_name(node.func, known_aliases) in {
        "getattr",
        "inspect.getattr_static",
    }:
        return bool(node.args) and expression_name(node.args[0], known_aliases).startswith("dsio.")
    if isinstance(node, ast.Call) and expression_name(node.func, known_aliases) in {
        "object.__getattribute__",
        "vars",
    }:
        return bool(node.args) and expression_name(node.args[0], known_aliases).startswith("dsio.")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Call):
        reflector = expression_name(node.func.func, known_aliases)
        return reflector in {"operator.attrgetter", "operator.methodcaller"} and bool(
            node.args
        ) and expression_name(node.args[0], known_aliases).startswith("dsio.")
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


def _is_registry_receiver(name: str) -> bool:
    parts = name.split(".")
    registry_names = {registry.rpartition(".")[2] for registry in _REGISTRIES}
    return any(
        name == registry or name.startswith(f"{registry}.") for registry in _REGISTRIES
    ) or any(part in registry_names and name.startswith("dsio.") for part in parts)
