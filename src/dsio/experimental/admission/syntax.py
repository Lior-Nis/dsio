"""Small AST helpers used by the experimental source audit."""

from __future__ import annotations

import ast

_REGISTRATION_CALLABLES = frozenset(
    {
        "dsio.eval.metrics.metric",
        "dsio.train.runner.preflight",
        "dsio.train.runner.runner",
    }
)
_REGISTRIES = frozenset(
    {
        "dsio.config.schema.TASKS",
        "dsio.eval.metrics.METRICS",
        "dsio.train.runner.PREFLIGHTS",
        "dsio.train.runner.RUNNERS",
    }
)


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


def references_project_identity(tree: ast.AST, aliases: set[str] | None = None) -> bool:
    aliases = aliases or set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name | ast.Attribute):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if expression_name(node, {}) in aliases or _project_identifier(name):
                return True
        elif (
            isinstance(node, ast.Subscript) and _literal_name(node.slice) == "project"
        ) or (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _literal_name(node.args[0]) == "project"
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
        if isinstance(node, ast.Compare | ast.Match | ast.Call | ast.Subscript)
    )
    return tuple(node for node in candidates if references_project_identity(node, aliases))


def registry_mutation(call: ast.Call, known_aliases: dict[str, str]) -> bool:
    name = expression_name(call.func, known_aliases)
    if name in _REGISTRATION_CALLABLES:
        return True
    if name in {"setattr", "delattr"} and call.args:
        return _is_registry_receiver(expression_name(call.args[0], known_aliases))
    receiver, separator, method = name.rpartition(".")
    return bool(separator) and _is_registry_receiver(receiver) and method in {
        "add",
        "clear",
        "pop",
        "register",
        "setdefault",
        "update",
    }


def registry_reference(node: ast.AST, known_aliases: dict[str, str]) -> bool:
    if isinstance(node, ast.Name | ast.Attribute | ast.Subscript | ast.Call):
        name = expression_name(node, known_aliases)
        if name in _REGISTRATION_CALLABLES or _is_registry_receiver(name):
            return True
    if isinstance(node, ast.Call) and expression_name(node.func, known_aliases) == "getattr":
        return bool(node.args) and expression_name(node.args[0], known_aliases).startswith("dsio.")
    if isinstance(node, ast.Subscript):
        return expression_name(node.value, known_aliases).startswith("dsio.") and isinstance(
            node.value, ast.Attribute
        ) and node.value.attr == "__dict__"
    return False


def defines_registration_surface(tree: ast.AST) -> bool:
    if any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.casefold() in {"register", "register_component", "register_plugin"}
        for node in ast.walk(tree)
    ):
        return True
    if not isinstance(tree, ast.Module):
        return False
    tables = {
        target.id
        for statement in tree.body
        if isinstance(statement, ast.Assign | ast.AnnAssign)
        and statement.value is not None
        and _mutable_mapping(statement.value)
        for target in (
            statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        )
        if isinstance(target, ast.Name)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign | ast.Delete):
            targets = (
                node.targets
                if isinstance(node, ast.Assign | ast.Delete)
                else [node.target]
            )
            if any(_mutates_table(target, tables) for target in targets):
                return True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and expression_name(node.func.value, {}).partition(".")[0] in tables
            and node.func.attr in {"clear", "pop", "setdefault", "update"}
        ):
            return True
    return False


def registry_assignment(node: ast.AST, known_aliases: dict[str, str]) -> bool:
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign | ast.AugAssign):
        targets.append(node.target)
    elif isinstance(node, ast.Delete):
        targets.extend(node.targets)
    for target in targets:
        if isinstance(target, ast.Subscript | ast.Attribute):
            receiver = expression_name(target.value, known_aliases)
        else:
            continue
        if _is_registry_receiver(receiver):
            return True
    return False


def is_registry_api(imported: str) -> bool:
    terminal = imported.rpartition(".")[2]
    registry_names = {registry.rpartition(".")[2] for registry in _REGISTRIES}
    return (
        imported in _REGISTRATION_CALLABLES
        or imported in _REGISTRIES
        or (imported.startswith("dsio.") and terminal in registry_names)
    )


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
            attribute = _literal_name(expression.args[1])
            parent = expression_name(expression.args[0], known_aliases)
            return f"{parent}.{attribute}" if parent and attribute else ""
        if function == "vars" and expression.args:
            return f"{expression_name(expression.args[0], known_aliases)}.__dict__"
    if isinstance(expression, ast.Subscript):
        key = _literal_name(expression.slice)
        parent = expression_name(expression.value, known_aliases)
        if key and parent.endswith(".__dict__"):
            return f"{parent.removesuffix('.__dict__')}.{key}"
    return ""


def is_dynamic_import_api(name: str) -> bool:
    return name in {
        "builtins.__import__",
        "builtins.compile",
        "builtins.eval",
        "builtins.exec",
        "importlib.import_module",
    }


def dynamic_import_reference(node: ast.AST, known_aliases: dict[str, str]) -> bool:
    if not isinstance(node, ast.Name | ast.Attribute | ast.Call):
        return False
    name = expression_name(node, known_aliases)
    return name in {
        "__import__",
        "builtins.__import__",
        "builtins.compile",
        "builtins.eval",
        "builtins.exec",
        "compile",
        "eval",
        "exec",
        "importlib.import_module",
    }


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


def string_constants(tree: ast.AST) -> dict[str, set[str]]:
    constants: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
            continue
        value = _fold_string(node.value)
        if value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                constants.setdefault(target.id, set()).add(value)
    return constants


def string_values(tree: ast.AST, constants: dict[str, set[str]]) -> tuple[str, ...]:
    values: list[str] = []
    for node in ast.walk(tree):
        value = _fold_string(node)
        if value is not None:
            values.append(value)
        elif isinstance(node, ast.Name):
            values.extend(constants.get(node.id, ()))
    return tuple(values)


def _project_identifier(name: str) -> bool:
    lowered = name.casefold()
    return (
        lowered == "project"
        or lowered.startswith("project_")
        or lowered.endswith("_project")
        or "_project_" in lowered
        or (name.startswith("project") and len(name) > 7 and name[7].isupper())
        or "Project" in name
    )


def _is_registry_receiver(name: str) -> bool:
    parts = name.split(".")
    registry_names = {registry.rpartition(".")[2] for registry in _REGISTRIES}
    return any(
        name == registry or name.startswith(f"{registry}.") for registry in _REGISTRIES
    ) or any(
        part in registry_names and name.startswith("dsio.") for part in parts
    )


def _literal_name(node: ast.AST) -> str | None:
    return _fold_string(node)


def _fold_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _fold_string(node.left)
        right = _fold_string(node.right)
        return left + right if left is not None and right is not None else None
    return None


def _assigned_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name | ast.Attribute):
        return (expression_name(target, {}),)
    if isinstance(target, ast.Tuple | ast.List):
        return tuple(name for item in target.elts for name in _assigned_names(item))
    return ()


def _mutable_mapping(value: ast.expr) -> bool:
    return isinstance(value, ast.Dict) or (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"dict", "defaultdict"}
    )


def _mutates_table(target: ast.expr, tables: set[str]) -> bool:
    return isinstance(target, ast.Subscript) and expression_name(target.value, {}).partition(".")[
        0
    ] in tables
