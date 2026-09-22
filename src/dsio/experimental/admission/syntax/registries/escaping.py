"""Detection of function-local registries exposed through closing mutators."""

from __future__ import annotations

import ast

from dsio.experimental.admission.syntax.imports import expression_name


def escaping_local_mutator(tree: ast.Module) -> bool:
    functions = (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )
    for function in functions:
        tables = {
            target.id
            for statement in function.body
            if isinstance(statement, ast.Assign | ast.AnnAssign)
            and statement.value is not None
            and _mutable_collection(statement.value)
            for target in (
                statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            )
            if isinstance(target, ast.Name) and _registration_name(target.id)
        }
        if tables and _returns_mutator(function, tables):
            return True
    return False


def _returns_mutator(
    function: ast.FunctionDef | ast.AsyncFunctionDef, tables: set[str]
) -> bool:
    returned = {
        expression_name(node.value, {})
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and node.value is not None
    }
    for nested in function.body:
        if not isinstance(nested, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if nested.name in returned and any(
            isinstance(node, ast.Assign | ast.AnnAssign | ast.Delete)
            and any(
                _mutates(target, tables)
                for target in (
                    node.targets
                    if isinstance(node, ast.Assign | ast.Delete)
                    else [node.target]
                )
            )
            for node in ast.walk(nested)
        ):
            return True
    return False


def _mutable_collection(value: ast.expr) -> bool:
    return isinstance(value, ast.Dict | ast.List | ast.Set) or (
        isinstance(value, ast.Call)
        and expression_name(value.func, {}).rpartition(".")[2]
        in {"OrderedDict", "defaultdict", "dict", "list", "set"}
    )


def _registration_name(name: str) -> bool:
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


def _mutates(target: ast.expr, tables: set[str]) -> bool:
    if not isinstance(target, ast.Subscript):
        return False
    receiver = expression_name(target.value, {})
    return receiver.partition(".")[0] in tables or receiver.rpartition(".")[2] in tables
