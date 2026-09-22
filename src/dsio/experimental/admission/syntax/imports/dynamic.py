"""Detection of runtime import and code-evaluation APIs."""

from __future__ import annotations

import ast

from dsio.experimental.admission.syntax.imports.names import expression_name, fold_string

_DYNAMIC_APIS = frozenset(
    {
        "builtins.__import__",
        "builtins.compile",
        "builtins.eval",
        "builtins.exec",
        "importlib.import_module",
        "importlib.metadata.EntryPoint",
        "importlib.metadata.entry_points",
        "pkgutil.resolve_name",
        "pydoc.locate",
        "runpy",
        "runpy.run_module",
        "runpy.run_path",
    }
)


def is_dynamic_import_api(name: str) -> bool:
    return name in _DYNAMIC_APIS or name.startswith(("importlib.machinery", "importlib.util"))


def dynamic_import_reference(node: ast.AST, known_aliases: dict[str, str]) -> bool:
    if not isinstance(node, ast.Name | ast.Attribute | ast.Subscript | ast.Call):
        return False
    name = expression_name(node, known_aliases)
    if "sys.modules[" in name and any(
        module in name for module in ("'builtins'", "'importlib'", "'runpy'")
    ):
        return True
    if isinstance(node, ast.Call) and expression_name(node.func, known_aliases) in {
        "getattr",
        "vars",
    }:
        root = expression_name(node.args[0], known_aliases) if node.args else ""
        if root in {"builtins", "importlib", "runpy"}:
            return True
    bare_dynamic = {"__builtins__", "__import__", "compile", "eval", "exec"}
    return (
        name in bare_dynamic
        or is_dynamic_import_api(name)
        or _composes_dynamic_api(node, known_aliases)
    )


def _composes_dynamic_api(node: ast.AST, known_aliases: dict[str, str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    function = expression_name(node.func, known_aliases)
    if function in {"operator.attrgetter", "operator.methodcaller"}:
        return any(
            fold_string(argument) in {"__import__", "compile", "eval", "exec", "import_module"}
            for argument in node.args
        )
    if function == "functools.partial":
        if len(node.args) >= 3 and expression_name(node.args[0], known_aliases) in {
            "getattr",
            "object.__getattribute__",
        }:
            root = expression_name(node.args[1], known_aliases)
            attribute = fold_string(node.args[2])
            if root in {"builtins", "importlib", "runpy"} and attribute is not None:
                return True
        return any(dynamic_import_reference(argument, known_aliases) for argument in node.args)
    if function == "object.__getattribute__" and len(node.args) >= 2:
        root = expression_name(node.args[0], known_aliases)
        attribute = fold_string(node.args[1])
        return root in {"builtins", "importlib", "runpy"} and attribute is not None
    return False
