"""Detection of helpers that add components to candidate registry tables."""

from __future__ import annotations

import ast

from dsio.experimental.admission.syntax.imports import expression_name

_COMPONENT_ROLES = frozenset(
    {
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
)


def parameterized_registration(tree: ast.Module, tables: set[str]) -> bool:
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
        component_parameters = parameters & _COMPONENT_ROLES
        if not component_parameters:
            continue
        if _function_registers(function, parameters, component_parameters, tables):
            return True
        if _mutates_parameter(function, parameters, component_parameters):
            parameter_mutators.add(function.name)
    return any(
        isinstance(node, ast.Call)
        and expression_name(node.func, {}) in parameter_mutators
        and any(expression_name(argument, {}) in tables for argument in node.args)
        for node in ast.walk(tree)
    )


def _function_registers(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    parameters: set[str],
    component_parameters: set[str],
    tables: set[str],
) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value_names = {item.id for item in ast.walk(node.value) if isinstance(item, ast.Name)}
            if value_names & component_parameters and any(
                isinstance(target, ast.Subscript) and _receiver_table(target.value, tables)
                for target in targets
            ):
                return True
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value_names = {
                item.id
                for argument in node.args
                for item in ast.walk(argument)
                if isinstance(item, ast.Name)
            }
            if value_names & component_parameters and _receiver_table(node.func.value, tables):
                return True
    return False


def _mutates_parameter(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    parameters: set[str],
    component_parameters: set[str],
) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value_names = {item.id for item in ast.walk(node.value) if isinstance(item, ast.Name)}
            receivers = {
                expression_name(target.value, {}).partition(".")[0]
                for target in targets
                if isinstance(target, ast.Subscript)
            }
            if value_names & component_parameters and receivers & parameters:
                return True
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value_names = {
                item.id
                for argument in node.args
                for item in ast.walk(argument)
                if isinstance(item, ast.Name)
            }
            receiver = expression_name(node.func.value, {}).partition(".")[0]
            if value_names & component_parameters and receiver in parameters:
                return True
    return False


def _receiver_table(receiver: ast.expr, tables: set[str]) -> bool:
    name = expression_name(receiver, {})
    return name.partition(".")[0] in tables or name.rpartition(".")[2] in tables
