"""Published distribution metadata is DSio's consumer-facing install contract."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RUNTIME_DEPENDENCIES = {
    "prefect",
    "torch",
    "lightning",
    "torchmetrics",
    "mlflow",
}
OBSOLETE_ORCHESTRATION_PATHS = {
    "src/dsio/application.py",
    "src/dsio/presets.py",
    "src/dsio/config/overrides.py",
    "src/dsio/config/presets.py",
    "src/dsio/cli/__init__.py",
    "Dockerfile",
}


def _project_metadata() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()


def test_training_spine_is_part_of_the_required_distribution() -> None:
    data = _project_metadata()
    names = {_requirement_name(item) for item in data["project"]["dependencies"]}

    assert REQUIRED_RUNTIME_DEPENDENCIES <= names
    assert "mlflow-skinny" not in names
    assert "typer" not in names


def test_distribution_has_no_accelerator_extras_or_console_script() -> None:
    data = _project_metadata()
    extras = data["project"].get("optional-dependencies", {})
    scripts = data["project"].get("scripts", {})

    assert "cpu" not in extras
    assert "gpu" not in extras
    assert "dsio" not in scripts


def test_sdist_has_an_explicit_narrow_include_policy() -> None:
    data = _project_metadata()
    included = set(data["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])

    assert included == {"/src/dsio", "/pyproject.toml", "/README.md"}


def test_obsolete_orchestration_surface_is_absent() -> None:
    present = {path for path in OBSOLETE_ORCHESTRATION_PATHS if (ROOT / path).exists()}

    assert present == set()
