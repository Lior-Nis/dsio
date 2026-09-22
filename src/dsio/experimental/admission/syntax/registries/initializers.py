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
    return any(isinstance(target, ast.Name) and target.id in tables for target in targets)
