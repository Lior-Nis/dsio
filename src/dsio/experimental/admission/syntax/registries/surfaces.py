"""Checks for candidate-defined runtime registration surfaces."""

from __future__ import annotations

import ast

from dsio.experimental.admission.syntax.imports import expression_name
from dsio.experimental.admission.syntax.registries.escaping import escaping_local_mutator
from dsio.experimental.admission.syntax.registries.initializers import initialized_registration
from dsio.experimental.admission.syntax.registries.parameters import parameterized_registration


def defines_registration_surface(tree: ast.AST) -> bool:
    if not isinstance(tree, ast.Module):
        return False
    declarations = tuple(_container_assignments(tree))
    all_tables = {
        target.id
        for statement in declarations
        if statement.value is not None
        and _mutable_mapping(statement.value)
        for target in (
            statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        )
        if isinstance(target, ast.Name)
    }
    declared_registration_tables = {
        target.id
        for statement in declarations
        for target in (
            statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        )
        if isinstance(target, ast.Name) and _registration_table(target.id)
    }
    aliases = set(all_tables)
    table_assignments = [
        (target.id, expression_name(statement.value, {}))
        for statement in declarations
        if statement.value is not None
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
    if any(initialized_registration(statement, semantic_tables) for statement in declarations):
        return True
    if parameterized_registration(tree, aliases) or escaping_local_mutator(tree):
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


def _container_assignments(
    tree: ast.Module | ast.ClassDef,
) -> tuple[ast.Assign | ast.AnnAssign, ...]:
    assignments: list[ast.Assign | ast.AnnAssign] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign | ast.AnnAssign):
            assignments.append(statement)
        elif isinstance(statement, ast.ClassDef):
            assignments.extend(_container_assignments(statement))
    return tuple(assignments)


def _mutable_mapping(value: ast.expr) -> bool:
    return isinstance(value, ast.Dict | ast.List | ast.Set) or (
        isinstance(value, ast.Call)
        and expression_name(value.func, {}).rpartition(".")[2]
        in {"OrderedDict", "defaultdict", "dict", "list", "set"}
    )


def _mutates_table(target: ast.expr, tables: set[str]) -> bool:
    return isinstance(target, ast.Subscript) and _receiver_table(target.value, tables)


def _registration_table(name: str) -> bool:
    normalized = name.casefold()
    return normalized in {
        "catalog",
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
    } or normalized.endswith(("_catalog", "_registries", "_registry"))


def _receiver_table(receiver: ast.expr, tables: set[str]) -> bool:
    name = expression_name(receiver, {})
    return name.partition(".")[0] in tables or name.rpartition(".")[2] in tables
