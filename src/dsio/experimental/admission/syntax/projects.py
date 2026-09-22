"""Consumer-project identity checks for static admission."""

from __future__ import annotations

import ast
import re

from dsio.experimental.admission.syntax.imports import expression_name, fold_string


def references_project_identity(tree: ast.AST, aliases: set[str] | None = None) -> bool:
    aliases = aliases or set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if expression_name(node, {}) in aliases or fold_string(node.slice) == "project":
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
            and fold_string(node.args[0]) == "project"
        ):
            return True
    return False


def project_contexts(tree: ast.AST) -> tuple[ast.AST, ...]:
    aliases: set[str] = set()
    assignments = [
        (target, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
    ]
    for _ in assignments:
        changed = False
        for target, value in assignments:
            if references_project_identity(value, aliases):
                for name in _assigned_names(target):
                    if name not in aliases:
                        aliases.add(name)
                        changed = True
        if not changed:
            break
    candidates = (
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Compare
            | ast.If
            | ast.IfExp
            | ast.Match
            | ast.While,
        )
    )
    return tuple(node for node in candidates if references_project_identity(node, aliases))


def references_consumer_names(tree: ast.AST, names: set[str]) -> bool:
    if not names:
        return False
    identifiers = (
        candidate.id
        if isinstance(candidate, ast.Name)
        else candidate.attr
        if isinstance(candidate, ast.Attribute)
        else candidate.name
        for candidate in ast.walk(tree)
        if isinstance(candidate, ast.Name | ast.Attribute | ast.FunctionDef | ast.ClassDef)
    )
    return any(
        f"_{name}_" in f"_{_snake_case(identifier)}_"
        for identifier in identifiers
        for name in names
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
