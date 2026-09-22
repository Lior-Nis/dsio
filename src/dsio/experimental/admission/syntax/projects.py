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
    aliases: set[str] = set()
    assignments = (
        (target, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
    )
    for target, value in assignments:
        for name in _assigned_names(target):
            if references_project_identity(value, aliases):
                aliases.add(name)
            else:
                aliases.discard(name)
    candidates: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            candidates.append(node)
        elif isinstance(node, ast.If | ast.IfExp | ast.While):
            candidates.append(node.test)
        elif isinstance(node, ast.Match):
            candidates.append(node.subject)
            candidates.extend(case.guard for case in node.cases if case.guard is not None)
        elif isinstance(node, ast.For | ast.AsyncFor):
            candidates.append(node.iter)
        elif isinstance(node, ast.comprehension):
            candidates.append(node.iter)
            candidates.extend(node.ifs)
    return tuple(node for node in candidates if references_project_identity(node, aliases))


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
        "project_enabled",
        "project_id",
        "project_key",
        "project_name",
        "project_package",
        "project_slug",
    }


def _assigned_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name | ast.Attribute | ast.Subscript):
        return (expression_name(target, {}),)
    if isinstance(target, ast.Tuple | ast.List):
        return tuple(name for item in target.elts for name in _assigned_names(item))
    return ()


def _snake_case(name: str) -> str:
    separated = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated).casefold()
