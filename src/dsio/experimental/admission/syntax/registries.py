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
        if (
            name in _REGISTRATION_CALLABLES
            or _is_registry_receiver(name)
            or (name.startswith("dsio.") and ".__dict__" in name)
        ):
            return True
    if isinstance(node, ast.Call) and expression_name(node.func, known_aliases) == "getattr":
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


def defines_registration_surface(tree: ast.AST) -> bool:
    if any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.casefold() in {"register", "register_component", "register_plugin"}
        for node in ast.walk(tree)
    ):
        return True
    if not isinstance(tree, ast.Module):
        return False
    all_tables = {
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
    aliases = set(all_tables)
    table_assignments = [
        (target.id, expression_name(statement.value, {}))
        for statement in ast.walk(tree)
        if isinstance(statement, ast.Assign | ast.AnnAssign)
        and statement.value is not None
        for target in (
            statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        )
        if isinstance(target, ast.Name)
    ]
    for _ in table_assignments:
        changed = False
        for target, value in table_assignments:
            if value in aliases and target not in aliases:
                aliases.add(target)
                changed = True
        if not changed:
            break
    semantic_tables = {name for name in all_tables if _registration_table(name)}
    for _ in table_assignments:
        changed = False
        for target, value in table_assignments:
            if value in semantic_tables and target not in semantic_tables:
                semantic_tables.add(target)
                changed = True
        if not changed:
            break
    if _parameterized_registration(tree, aliases):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.Delete):
            targets = (
                node.targets
                if isinstance(node, ast.Assign | ast.Delete)
                else [node.target]
            )
            if any(_mutates_table(target, semantic_tables) for target in targets):
                return True
        elif isinstance(node, ast.AugAssign):
            if _mutates_table(node.target, semantic_tables) or expression_name(
                node.target, {}
            ) in semantic_tables:
                return True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and expression_name(node.func.value, {}).partition(".")[0] in semantic_tables
            and node.func.attr
            in {"__setitem__", "append", "clear", "extend", "insert", "pop", "setdefault", "update"}
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
    return isinstance(value, ast.Dict | ast.List | ast.Set) or (
        isinstance(value, ast.Call)
        and expression_name(value.func, {}).rpartition(".")[2]
        in {"OrderedDict", "defaultdict", "dict", "list", "set"}
    )


def _mutates_table(target: ast.expr, tables: set[str]) -> bool:
    return isinstance(target, ast.Subscript) and expression_name(target.value, {}).partition(".")[
        0
    ] in tables


def _registration_table(name: str) -> bool:
    return name.casefold() in {
        "components",
        "dispatch",
        "factories",
        "handlers",
        "metrics",
        "plugins",
        "registries",
        "registry",
        "runners",
        "tasks",
        "transforms",
    }


def _parameterized_registration(tree: ast.Module, tables: set[str]) -> bool:
    component_roles = {
        "callback",
        "collator",
        "component",
        "dataset",
        "factory",
        "handler",
        "loss",
        "metric",
        "model",
        "objective",
        "plugin",
        "runner",
        "sampler",
        "splitter",
        "transform",
    }
    parameter_mutators: set[str] = set()
    for function in (
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ):
        parameters = {
            argument.arg for argument in (*function.args.posonlyargs, *function.args.args)
        } | {argument.arg for argument in function.args.kwonlyargs}
        component_parameters = parameters & component_roles
        if not component_parameters:
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(target, ast.Subscript) for target in targets):
                continue
            value_names = {
                candidate.id
                for candidate in ast.walk(node.value)
                if isinstance(candidate, ast.Name)
            }
            if not value_names & component_parameters:
                continue
            receivers = {
                expression_name(target.value, {}).partition(".")[0]
                for target in targets
                if isinstance(target, ast.Subscript)
            }
            if receivers & tables:
                return True
            if receivers & parameters:
                parameter_mutators.add(function.name)
    return any(
        isinstance(node, ast.Call)
        and expression_name(node.func, {}) in parameter_mutators
        and any(expression_name(argument, {}) in tables for argument in node.args)
        for node in ast.walk(tree)
    )
