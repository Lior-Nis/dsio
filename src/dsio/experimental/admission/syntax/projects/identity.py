"""Recognition of explicit consumer-project identity expressions."""

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
                key is not None and project_identifier(key)
            ):
                return True
        elif isinstance(node, ast.Name | ast.Attribute):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if expression_name(node, {}) in aliases or (name and project_identifier(name)):
                return True
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and (key := fold_string(node.args[0])) is not None
            and project_identifier(key)
        ):
            return True
    return False


def project_identifier(name: str) -> bool:
    return snake_case(name) in {
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


def snake_case(name: str) -> str:
    separated = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated).casefold()
