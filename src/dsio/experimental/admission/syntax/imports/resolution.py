"""Scope- and source-order-aware import alias resolution."""

from __future__ import annotations

import ast

from dsio.experimental.admission.syntax.imports.names import expression_name


def imports(tree: ast.AST, module: str, *, is_package: bool = False) -> tuple[str, ...]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _import_from_base(node, module, is_package=is_package)
            imported.extend(
                f"{base}.*" if alias.name == "*" else f"{base}.{alias.name}".strip(".")
                for alias in node.names
            )
    return tuple(imported)


def aliases(tree: ast.AST, module: str, *, is_package: bool = False) -> dict[str, str]:
    known: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                known[alias.asname or alias.name.partition(".")[0]] = (
                    alias.name if alias.asname else alias.name.partition(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            base = _import_from_base(node, module, is_package=is_package)
            for alias in node.names:
                if alias.name != "*":
                    known[alias.asname or alias.name] = f"{base}.{alias.name}".strip(".")
    return known


def aliases_at(
    tree: ast.AST,
    node: ast.AST,
    module: str,
    *,
    is_package: bool = False,
) -> dict[str, str]:
    """Resolve import aliases visible at one node, respecting scope and source order."""
    if not isinstance(tree, ast.Module):
        return {}
    contained_scopes = [
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        and any(descendant is node for descendant in ast.walk(candidate))
    ]
    contained_scopes.sort(
        key=lambda candidate: (
            candidate.lineno,
            -(candidate.end_lineno or candidate.lineno),
        )
    )
    scopes: list[ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = [
        tree,
        *contained_scopes,
    ]
    resolved: dict[str, str] = {}
    for index, scope in enumerate(scopes):
        _remove_parameters(scope, resolved)
        cutoff = _position(node) if index == len(scopes) - 1 else (float("inf"), 0)
        for event in sorted(_scope_events(scope), key=_position):
            if _position(event) > cutoff:
                break
            _apply_alias_event(event, resolved, module, is_package=is_package)
    return resolved


def assigned_aliases(tree: ast.AST, known_aliases: dict[str, str]) -> dict[str, str]:
    resolved = dict(known_aliases)
    assignments = [
        (target.id, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    ]
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


def _remove_parameters(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    resolved: dict[str, str],
) -> None:
    if not isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
        return
    parameters = (*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs)
    for parameter in parameters:
        resolved.pop(parameter.arg, None)
    if scope.args.vararg is not None:
        resolved.pop(scope.args.vararg.arg, None)
    if scope.args.kwarg is not None:
        resolved.pop(scope.args.kwarg.arg, None)


def _scope_events(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> tuple[ast.Import | ast.ImportFrom | ast.Assign | ast.AnnAssign, ...]:
    events: list[ast.Import | ast.ImportFrom | ast.Assign | ast.AnnAssign] = []

    def collect(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
            ):
                continue
            if isinstance(child, ast.Import | ast.ImportFrom | ast.Assign | ast.AnnAssign):
                events.append(child)
            collect(child)

    for statement in scope.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        if isinstance(statement, ast.Import | ast.ImportFrom | ast.Assign | ast.AnnAssign):
            events.append(statement)
        collect(statement)
    return tuple(dict.fromkeys(events))


def _apply_alias_event(
    event: ast.Import | ast.ImportFrom | ast.Assign | ast.AnnAssign,
    resolved: dict[str, str],
    module: str,
    *,
    is_package: bool,
) -> None:
    if isinstance(event, ast.Import):
        for imported in event.names:
            resolved[imported.asname or imported.name.partition(".")[0]] = (
                imported.name if imported.asname else imported.name.partition(".")[0]
            )
        return
    if isinstance(event, ast.ImportFrom):
        base = _import_from_base(event, module, is_package=is_package)
        for imported in event.names:
            if imported.name != "*":
                resolved[imported.asname or imported.name] = (
                    f"{base}.{imported.name}".strip(".")
                )
        return
    value = event.value
    if value is None:
        return
    targets = event.targets if isinstance(event, ast.Assign) else [event.target]
    name = expression_name(value, resolved)
    for target in targets:
        if not isinstance(target, ast.Name):
            continue
        if name:
            resolved[target.id] = name
        else:
            resolved.pop(target.id, None)


def _position(node: ast.AST) -> tuple[float, int]:
    return (float(getattr(node, "lineno", 0)), getattr(node, "col_offset", 0))


def _import_from_base(
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
