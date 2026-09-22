"""Checks for candidate-defined runtime registration surfaces."""

from __future__ import annotations

import ast

from dsio.experimental.admission.syntax.imports import expression_name


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
        for statement in ast.walk(tree)
        if isinstance(statement, ast.Assign | ast.AnnAssign)
        and statement.value is not None
        and _mutable_mapping(statement.value)
        for target in (
            statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        )
        if isinstance(target, ast.Name)
    }
    declared_registration_tables = {
        target.id
        for statement in ast.walk(tree)
        if isinstance(statement, ast.Assign | ast.AnnAssign)
        for target in (
            statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        )
        if isinstance(target, ast.Name) and _registration_table(target.id)
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
    _expand_aliases(aliases, table_assignments)
    semantic_tables = declared_registration_tables | {
        name for name in all_tables if _registration_table(name)
    }
    _expand_aliases(semantic_tables, table_assignments)
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
            and _receiver_table(node.func.value, semantic_tables)
            and node.func.attr
            in {
                "__setitem__",
                "add",
                "append",
                "clear",
                "extend",
                "insert",
                "pop",
                "register",
                "setdefault",
                "update",
            }
        ):
            return True
    return False


def _expand_aliases(names: set[str], assignments: list[tuple[str, str]]) -> None:
    for _ in assignments:
        changed = False
        for target, value in assignments:
            if value in names and target not in names:
                names.add(target)
                changed = True
        if not changed:
            break


def _mutable_mapping(value: ast.expr) -> bool:
    return isinstance(value, ast.Dict | ast.List | ast.Set) or (
        isinstance(value, ast.Call)
        and expression_name(value.func, {}).rpartition(".")[2]
        in {"OrderedDict", "defaultdict", "dict", "list", "set"}
    )


def _mutates_table(target: ast.expr, tables: set[str]) -> bool:
    return isinstance(target, ast.Subscript) and _receiver_table(target.value, tables)


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
    functions = (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )
    for function in functions:
        parameters = {
            argument.arg for argument in (*function.args.posonlyargs, *function.args.args)
        } | {argument.arg for argument in function.args.kwonlyargs}
        component_parameters = parameters & component_roles
        if not component_parameters:
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
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
                if any(
                    _receiver_table(target.value, tables)
                    for target in targets
                    if isinstance(target, ast.Subscript)
                ):
                    return True
                if receivers & parameters:
                    parameter_mutators.add(function.name)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                value_names = {
                    candidate.id
                    for argument in node.args
                    for candidate in ast.walk(argument)
                    if isinstance(candidate, ast.Name)
                }
                if value_names & component_parameters:
                    if _receiver_table(node.func.value, tables):
                        return True
                    receiver = expression_name(node.func.value, {}).partition(".")[0]
                    if receiver in parameters:
                        parameter_mutators.add(function.name)
    return any(
        isinstance(node, ast.Call)
        and expression_name(node.func, {}) in parameter_mutators
        and any(expression_name(argument, {}) in tables for argument in node.args)
        for node in ast.walk(tree)
    )


def _receiver_table(receiver: ast.expr, tables: set[str]) -> bool:
    name = expression_name(receiver, {})
    return name.partition(".")[0] in tables or name.rpartition(".")[2] in tables
