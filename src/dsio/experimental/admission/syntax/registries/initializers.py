"""Detection of populated candidate registry declarations."""

from __future__ import annotations

import ast


def initialized_registration(
    statement: ast.Assign | ast.AnnAssign, tables: set[str]
) -> bool:
    value = statement.value
    if value is None or not isinstance(value, ast.Dict | ast.List | ast.Set):
        return False
    elements = (
        [item for item in value.values if item is not None]
        if isinstance(value, ast.Dict)
        else list(value.elts)
    )
    if not elements:
        return False
    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    names = [target.id for target in targets if isinstance(target, ast.Name)]
    if not any(name in tables for name in names):
        return False
    if any(
        name.casefold() in {"catalog", "registries", "registry"}
        or name.casefold().endswith(("_catalog", "_registries", "_registry"))
        for name in names
    ):
        return True
    return isinstance(value, ast.Dict) and any(
        isinstance(element, ast.Name | ast.Attribute | ast.Lambda | ast.Call)
        for element in elements
    )
