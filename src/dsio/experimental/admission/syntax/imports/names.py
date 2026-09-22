"""Static expression-name and string folding."""

from __future__ import annotations

import ast


def expression_name(expression: ast.expr, known_aliases: dict[str, str]) -> str:
    if isinstance(expression, ast.Name):
        return known_aliases.get(expression.id, expression.id)
    if isinstance(expression, ast.Attribute):
        parent = expression_name(expression.value, known_aliases)
        return f"{parent}.{expression.attr}" if parent else ""
    if isinstance(expression, ast.Call):
        function = expression_name(expression.func, known_aliases)
        if function == "getattr" and len(expression.args) >= 2:
            attribute = fold_string(expression.args[1])
            parent = expression_name(expression.args[0], known_aliases)
            return f"{parent}.{attribute}" if parent and attribute else ""
        if function == "vars" and expression.args:
            return f"{expression_name(expression.args[0], known_aliases)}.__dict__"
    if isinstance(expression, ast.Subscript):
        key = fold_string(expression.slice)
        parent = expression_name(expression.value, known_aliases)
        if key and parent.endswith(".__dict__"):
            return f"{parent.removesuffix('.__dict__')}.{key}"
        if key and parent:
            return f"{parent}[{key!r}]"
    return ""


def fold_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = fold_string(node.left)
        right = fold_string(node.right)
        return left + right if left is not None and right is not None else None
    return None
