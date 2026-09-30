"""The installed distribution metadata is DSio's only version."""

from __future__ import annotations

import ast
from importlib.metadata import version
from pathlib import Path

import dsio
import dsio.config.components
import dsio.model.chain
import dsio.model.module

PACKAGE_INIT = Path(dsio.__file__)


def test_reported_version_is_the_installed_distribution_version() -> None:
    assert dsio.__version__ == version("dsio")


def test_package_init_writes_no_version_literal() -> None:
    tree = ast.parse(PACKAGE_INIT.read_text(encoding="utf-8"))
    literals = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
    ]
    assert literals == []


def test_component_error_is_defined_once() -> None:
    canonical = dsio.config.components.ComponentError
    assert dsio.model.chain.ComponentError is canonical
    assert dsio.model.module.ComponentError is canonical
