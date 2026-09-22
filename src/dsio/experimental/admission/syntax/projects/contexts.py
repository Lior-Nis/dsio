"""Source-order-aware consumer-project control-flow analysis."""

from __future__ import annotations

import ast

from dsio.experimental.admission.syntax.imports import expression_name
from dsio.experimental.admission.syntax.projects.identity import (
    project_identifier,
    references_project_identity,
)


def project_contexts(tree: ast.AST) -> tuple[ast.AST, ...]:
    if not isinstance(tree, ast.Module):
        return ()
    contexts: list[ast.AST] = []
    _scan_statements(tree.body, set(), contexts)
    return tuple(contexts)


def _scan_statements(
    statements: list[ast.stmt], aliases: set[str], contexts: list[ast.AST]
) -> set[str]:
    state = set(aliases)
    for statement in statements:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            _record_definition_contexts(statement, state, contexts)
            parameters = {
                argument.arg
                for argument in (
                    *statement.args.posonlyargs,
                    *statement.args.args,
                    *statement.args.kwonlyargs,
                )
            }
            if statement.args.vararg is not None:
                parameters.add(statement.args.vararg.arg)
            if statement.args.kwarg is not None:
                parameters.add(statement.args.kwarg.arg)
            local_state = state - parameters
            local_state.update(name for name in parameters if project_identifier(name))
            _scan_statements(statement.body, local_state, contexts)
            continue
        if isinstance(statement, ast.ClassDef):
            _record_definition_contexts(statement, state, contexts)
            _scan_statements(statement.body, state, contexts)
            continue
        _update_named_expressions(statement, state)
        if isinstance(statement, ast.If):
            _record_context(statement.test, state, contexts)
            body_state = _scan_statements(statement.body, state, contexts)
            else_state = _scan_statements(statement.orelse, state, contexts)
            state = body_state | else_state
            continue
        if isinstance(statement, ast.While):
            _record_context(statement.test, state, contexts)
            body_state = _scan_statements(statement.body, state, contexts)
            else_state = _scan_statements(statement.orelse, state, contexts)
            state |= body_state | else_state
            continue
        if isinstance(statement, ast.For | ast.AsyncFor):
            _record_context(statement.iter, state, contexts)
            loop_state = set(state)
            for name in _assigned_names(statement.target):
                loop_state.discard(name)
            body_state = _scan_statements(statement.body, loop_state, contexts)
            else_state = _scan_statements(statement.orelse, state, contexts)
            state |= body_state | else_state
            continue
        if isinstance(statement, ast.Match):
            _record_context(statement.subject, state, contexts)
            branch_states = [set(state)]
            for case in statement.cases:
                if case.guard is not None:
                    _record_context(case.guard, state, contexts)
                branch_states.append(_scan_statements(case.body, state, contexts))
            state = set().union(*branch_states)
            continue
        if isinstance(statement, ast.Try | ast.TryStar):
            branch_states = [_scan_statements(statement.body, state, contexts)]
            branch_states.extend(
                _scan_statements(handler.body, state, contexts)
                for handler in statement.handlers
            )
            branch_states.append(_scan_statements(statement.orelse, state, contexts))
            state = set().union(state, *branch_states)
            state = _scan_statements(statement.finalbody, state, contexts)
            continue
        if isinstance(statement, ast.With | ast.AsyncWith):
            for item in statement.items:
                _record_nested_contexts(item.context_expr, state, contexts)
                if item.optional_vars is not None:
                    _update_aliases(item.optional_vars, item.context_expr, state)
            state = _scan_statements(statement.body, state, contexts)
            continue
        if isinstance(statement, ast.Assert):
            _record_context(statement.test, state, contexts)
        _record_nested_contexts(statement, state, contexts)
        if isinstance(statement, ast.Assign | ast.AnnAssign) and statement.value is not None:
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                _update_aliases(target, statement.value, state)
        elif isinstance(statement, ast.AugAssign):
            project_derived = references_project_identity(statement.value, state)
            for name in _assigned_names(statement.target):
                if project_derived or name in state:
                    state.add(name)
                else:
                    state.discard(name)
    return state


def _record_context(node: ast.AST, aliases: set[str], contexts: list[ast.AST]) -> None:
    if references_project_identity(node, aliases):
        contexts.append(node)


def _record_nested_contexts(
    node: ast.AST, aliases: set[str], contexts: list[ast.AST]
) -> None:
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.BoolOp):
            _record_context(candidate, aliases, contexts)
        elif isinstance(candidate, ast.IfExp):
            _record_context(candidate.test, aliases, contexts)
        elif isinstance(candidate, ast.comprehension):
            _record_context(candidate.iter, aliases, contexts)
            for condition in candidate.ifs:
                _record_context(condition, aliases, contexts)


def _record_definition_contexts(
    definition: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    aliases: set[str],
    contexts: list[ast.AST],
) -> None:
    expressions: list[ast.AST] = [*definition.decorator_list]
    if isinstance(definition, ast.ClassDef):
        expressions.extend(definition.bases)
        expressions.extend(keyword.value for keyword in definition.keywords)
    else:
        expressions.extend(definition.args.defaults)
        expressions.extend(value for value in definition.args.kw_defaults if value is not None)
        if definition.returns is not None:
            expressions.append(definition.returns)
        arguments = (
            *definition.args.posonlyargs,
            *definition.args.args,
            *definition.args.kwonlyargs,
        )
        expressions.extend(
            argument.annotation for argument in arguments if argument.annotation is not None
        )
    for expression in expressions:
        _record_nested_contexts(expression, aliases, contexts)


def _update_named_expressions(node: ast.AST, aliases: set[str]) -> None:
    named_expressions = sorted(
        (candidate for candidate in ast.walk(node) if isinstance(candidate, ast.NamedExpr)),
        key=lambda candidate: (candidate.lineno, candidate.col_offset),
    )
    for expression in named_expressions:
        _update_aliases(expression.target, expression.value, aliases)


def _update_aliases(target: ast.expr, value: ast.expr, aliases: set[str]) -> None:
    if (
        isinstance(target, ast.Tuple | ast.List)
        and isinstance(value, ast.Tuple | ast.List)
        and len(target.elts) == len(value.elts)
    ):
        for target_item, value_item in zip(target.elts, value.elts, strict=True):
            _update_aliases(target_item, value_item, aliases)
        return
    project_derived = references_project_identity(value, aliases)
    for name in _assigned_names(target):
        if project_derived:
            aliases.add(name)
        else:
            aliases.discard(name)


def _assigned_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name | ast.Attribute | ast.Subscript):
        return (expression_name(target, {}),)
    if isinstance(target, ast.Tuple | ast.List):
        return tuple(name for item in target.elts for name in _assigned_names(item))
    return ()
