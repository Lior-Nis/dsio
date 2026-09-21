"""Published distribution metadata is DSio's consumer-facing install contract."""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RUNTIME_DEPENDENCIES = {
    "prefect": ">=3.8,<4",
    "torch": ">=2.7,<3",
    "lightning": ">=2.5,<3",
    "torchmetrics": ">=1.7,<2",
    "mlflow": ">=3,<4",
}
OBSOLETE_ORCHESTRATION_PATHS = {
    "src/dsio/application.py",
    "src/dsio/presets.py",
    "src/dsio/config/overrides.py",
    "src/dsio/config/presets.py",
    "src/dsio/cli/__init__.py",
    "src/dsio/cli/envelope.py",
    "src/dsio/cli/main.py",
    "src/dsio/cli/run_cmd.py",
    "Dockerfile",
}


def _project_metadata() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_training_spine_is_part_of_the_required_distribution() -> None:
    data = _project_metadata()
    requirements = {
        requirement.name.lower(): requirement
        for item in data["project"]["dependencies"]
        if (requirement := Requirement(item)).name.lower() in REQUIRED_RUNTIME_DEPENDENCIES
    }

    assert set(requirements) == set(REQUIRED_RUNTIME_DEPENDENCIES)
    for name, expected_specifier in REQUIRED_RUNTIME_DEPENDENCIES.items():
        requirement = requirements[name]
        assert str(requirement.specifier) == str(Requirement(f"x{expected_specifier}").specifier)
        assert requirement.marker is None
        assert requirement.url is None
        assert not requirement.extras

    names = {Requirement(item).name.lower() for item in data["project"]["dependencies"]}
    assert "mlflow-skinny" not in names
    assert "typer" not in names
    assert data["project"]["requires-python"] == ">=3.12"


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
