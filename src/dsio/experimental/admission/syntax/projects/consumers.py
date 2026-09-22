"""Recognition of explicitly configured consumer names."""

from __future__ import annotations

import ast

from dsio.experimental.admission.syntax.imports import fold_string
from dsio.experimental.admission.syntax.projects.identity import snake_case


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
            if f"_{name}_" in f"_{snake_case(identifier)}_"
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
