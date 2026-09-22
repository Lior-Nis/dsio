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
        if nested.name not in returned or not _component_registration_evidence(nested, tables):
            continue
        if any(_mutates_node(node, tables) for node in ast.walk(nested)):
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


def _mutates_node(node: ast.AST, tables: set[str]) -> bool:
    if isinstance(node, ast.Assign | ast.AnnAssign | ast.Delete):
        targets = node.targets if isinstance(node, ast.Assign | ast.Delete) else [node.target]
        return any(_mutates(target, tables) for target in targets)
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    receiver = expression_name(node.func.value, {})
    return (
        receiver.partition(".")[0] in tables or receiver.rpartition(".")[2] in tables
    ) and node.func.attr in {
        "__setitem__",
        "add",
        "append",
        "extend",
        "insert",
        "register",
        "setdefault",
        "update",
    }


def _component_registration_evidence(
    function: ast.FunctionDef | ast.AsyncFunctionDef, tables: set[str]
) -> bool:
    explicit_table = any(
        name.casefold() in {"catalog", "registries", "registry"}
        or name.casefold().endswith(("_catalog", "_registries", "_registry"))
        for name in tables
    )
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
    parameters = (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
    component_parameters = {
        argument.arg
        for argument in parameters
        if argument.arg in component_roles and not _primitive_annotation(argument.annotation)
    }
    return explicit_table or bool(component_parameters)


def _primitive_annotation(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    name = expression_name(annotation, {}).rpartition(".")[2]
    return name in {"bool", "bytes", "complex", "float", "int", "str"}
