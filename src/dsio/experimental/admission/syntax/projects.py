"""Consumer-project identity checks for static admission."""

from __future__ import annotations

import ast
import re

from dsio.experimental.admission.syntax.imports import expression_name, fold_string


def references_project_identity(tree: ast.AST, aliases: set[str] | None = None) -> bool:
    aliases = aliases or set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            key = fold_string(node.slice)
            if expression_name(node, {}) in aliases or (
                key is not None and _project_identifier(key)
            ):
                return True
        elif isinstance(node, ast.Name | ast.Attribute):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if expression_name(node, {}) in aliases or (name and _project_identifier(name)):
                return True
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and (key := fold_string(node.args[0])) is not None
            and _project_identifier(key)
        ):
            return True
    return False


def project_contexts(tree: ast.AST) -> tuple[ast.AST, ...]:
    if not isinstance(tree, ast.Module):
        return ()
    contexts: list[ast.AST] = []
    _scan_statements(tree.body, set(), contexts)
    return tuple(contexts)


def references_consumer_names(tree: ast.AST, names: set[str]) -> str | None:
    if not names:
        return None
    identifiers = (
        candidate.id
        if isinstance(candidate, ast.Name)
        else candidate.attr
        if isinstance(candidate, ast.Attribute)
        else candidate.arg or ""
        if isinstance(candidate, ast.arg | ast.keyword)
        else candidate.asname or candidate.name
        if isinstance(candidate, ast.alias)
        else candidate.name
        for candidate in ast.walk(tree)
        if isinstance(
            candidate,
            ast.Name
            | ast.Attribute
            | ast.arg
            | ast.keyword
            | ast.alias
            | ast.FunctionDef
            | ast.ClassDef,
        )
    )
    return next(
        (
            name
            for identifier in identifiers
            for name in sorted(names)
            if f"_{name}_" in f"_{_snake_case(identifier)}_"
        ),
        None,
    )


def string_values(tree: ast.AST, constants: dict[str, set[str]]) -> tuple[str, ...]:
    values: list[str] = []
    for node in ast.walk(tree):
        value = fold_string(node)
        if value is not None:
            values.append(value)
        elif isinstance(node, ast.Name):
            values.extend(constants.get(node.id, ()))
    return tuple(values)


def _project_identifier(name: str) -> bool:
    return _snake_case(name) in {
        "consumer_project",
        "current_project",
        "project",
        "project_code",
        "project_context",
        "project_enabled",
        "project_environment",
        "project_id",
        "project_key",
        "project_name",
        "project_package",
        "project_slug",
        "project_type",
    }


def _scan_statements(
    statements: list[ast.stmt], aliases: set[str], contexts: list[ast.AST]
) -> set[str]:
    state = set(aliases)
    for statement in statements:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
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
            local_state.update(name for name in parameters if _project_identifier(name))
            _scan_statements(statement.body, local_state, contexts)
            continue
        if isinstance(statement, ast.ClassDef):
            _scan_statements(statement.body, state, contexts)
            continue
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
                    for name in _assigned_names(item.optional_vars):
                        state.discard(name)
            state = _scan_statements(statement.body, state, contexts)
            continue
        if isinstance(statement, ast.Assert):
            _record_context(statement.test, state, contexts)
        _record_nested_contexts(statement, state, contexts)
        if isinstance(statement, ast.Assign | ast.AnnAssign) and statement.value is not None:
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                _update_aliases(target, statement.value, state)
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


def _snake_case(name: str) -> str:
    separated = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated).casefold()
