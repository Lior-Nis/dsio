"""Import and expression-name resolution for static admission."""

from __future__ import annotations

import ast


def imports(tree: ast.AST, module: str, *, is_package: bool = False) -> tuple[str, ...]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = import_from_base(node, module, is_package=is_package)
            imported.extend(
                f"{base}.*" if alias.name == "*" else f"{base}.{alias.name}".strip(".")
                for alias in node.names
            )
    return tuple(imported)


def aliases(
    tree: ast.AST, module: str, *, is_package: bool = False
) -> dict[str, str]:
    known: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                known[alias.asname or alias.name.partition(".")[0]] = (
                    alias.name if alias.asname else alias.name.partition(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            base = import_from_base(node, module, is_package=is_package)
            for alias in node.names:
                if alias.name != "*":
                    known[alias.asname or alias.name] = f"{base}.{alias.name}".strip(".")
    return known


def assigned_aliases(tree: ast.AST, known_aliases: dict[str, str]) -> dict[str, str]:
    resolved = dict(known_aliases)
    assignments: list[tuple[str, ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            assignments.extend(
                (target.id, node.value) for target in node.targets if isinstance(target, ast.Name)
            )
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            assignments.append((node.target.id, node.value))
    for _ in assignments:
        changed = False
        for target, value in assignments:
            name = expression_name(value, resolved)
            if name and resolved.get(target) != name:
                resolved[target] = name
                changed = True
        if not changed:
            break
    return resolved


def expression_name(expression: ast.expr, known_aliases: dict[str, str]) -> str:
    if isinstance(expression, ast.Name):
        return known_aliases.get(expression.id, expression.id)
    if isinstance(expression, ast.Attribute):
        parent = expression_name(expression.value, known_aliases)
        return f"{parent}.{expression.attr}" if parent else expression.attr
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


def is_dynamic_import_api(name: str) -> bool:
    return name == "builtins" or name.startswith(("builtins.", "importlib", "runpy"))


def dynamic_import_reference(node: ast.AST, known_aliases: dict[str, str]) -> bool:
    if not isinstance(node, ast.Name | ast.Attribute | ast.Subscript | ast.Call):
        return False
    name = expression_name(node, known_aliases)
    return name == "__builtins__" or name in {
        "__import__",
        "builtins.__import__",
        "builtins.compile",
        "builtins.eval",
        "builtins.exec",
        "compile",
        "eval",
        "exec",
        "importlib.import_module",
    } or name.startswith(("builtins.", "importlib.", "runpy."))


def import_from_base(
    node: ast.ImportFrom, module: str, *, is_package: bool = False
) -> str:
    if not node.level:
        return node.module or ""
    package = module if is_package else module.rpartition(".")[0]
    parts = package.split(".") if package else []
    remove = node.level - 1
    if remove:
        parts = parts[:-remove]
    return ".".join([*parts, node.module or ""]).rstrip(".")


def fold_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = fold_string(node.left)
        right = fold_string(node.right)
        return left + right if left is not None and right is not None else None
    return None
