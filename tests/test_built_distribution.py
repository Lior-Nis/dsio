"""The built artifacts, not the checkout, define DSio's install boundary."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DEPENDENCIES = {"prefect", "torch", "lightning", "torchmetrics", "mlflow"}
REMOVED_MEMBERS = {
    "dsio/application.py",
    "dsio/presets.py",
    "dsio/config/overrides.py",
    "dsio/config/presets.py",
}
REVIEW_REPORTS = {
    "architecture_review.md",
    "claude_review.md",
    "codex_review.md",
    "opencode_review.md",
}
EXPECTED_DIGEST = "2ca217b09c12c36031e1078aab6a81e705f1281b1eb841c876f98cef5cb03556"


def _run(*command: str, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True, text=True, capture_output=True)


def _requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()


@pytest.fixture(scope="module")
def built_artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    output = tmp_path_factory.mktemp("distribution")
    _run("uv", "build", "--out-dir", str(output))
    return next(output.glob("*.whl")), next(output.glob("*.tar.gz"))


def test_wheel_contains_only_the_public_package_and_neutral_metadata(
    built_artifacts: tuple[Path, Path],
) -> None:
    wheel, _ = built_artifacts
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        metadata_name = next(name for name in members if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        entry_points = [name for name in members if name.endswith(".dist-info/entry_points.txt")]

    requirements = metadata.get_all("Requires-Dist", [])
    names = {_requirement_name(requirement) for requirement in requirements}

    assert "dsio/__init__.py" in members
    assert REQUIRED_DEPENDENCIES <= names
    assert "mlflow-skinny" not in names
    assert not any("pytorch-cpu" in requirement for requirement in requirements)
    assert not entry_points
    forbidden_prefixes = ("tests/", "project/", "dsio/torch/", "dsio/cli/")
    assert not any(name.startswith(forbidden_prefixes) for name in members)
    assert REMOVED_MEMBERS.isdisjoint(members)


def test_sdist_excludes_repository_internal_material(
    built_artifacts: tuple[Path, Path],
) -> None:
    _, sdist = built_artifacts
    with tarfile.open(sdist, "r:gz") as archive:
        members = {Path(name).as_posix() for name in archive.getnames()}

    relative = {name.split("/", 1)[1] for name in members if "/" in name}
    assert "pyproject.toml" in relative
    assert "README.md" in relative
    assert "src/dsio/__init__.py" in relative
    # Hatch always includes the VCS ignore file in an sdist; everything else is the
    # declared source/build/readme allowlist plus generated package metadata.
    assert relative <= {".gitignore", "README.md", "pyproject.toml", "PKG-INFO"} | {
        name for name in relative if name.startswith("src/dsio/")
    }
    assert not any(name.startswith("_bmad-output/") for name in relative)
    assert REVIEW_REPORTS.isdisjoint(relative)


def test_wheel_installs_with_dependencies_and_root_import_is_inert(
    built_artifacts: tuple[Path, Path], tmp_path: Path,
) -> None:
    wheel, _ = built_artifacts
    environment = tmp_path / "environment"
    state = tmp_path / "state"
    work = state / "work"
    temp = state / "tmp"
    home = state / "home"
    prefect_home = state / "prefect"
    mlflow_store = state / "mlflow"
    work.mkdir(parents=True)
    temp.mkdir()
    home.mkdir(parents=True)

    _run("uv", "venv", "--python", sys.executable, str(environment))
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run(
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--index",
        "https://download.pytorch.org/whl/cpu",
        "torch>=2.7,<3",
    )
    _run("uv", "pip", "install", "--python", str(python), str(wheel))
    _run("uv", "pip", "check", "--python", str(python))

    probe = """
import json
import pathlib
import socket
import sys

state = pathlib.Path(__import__('os').environ['DSIO_IMPORT_STATE_ROOT'])
before = {str(path.relative_to(state)) for path in state.rglob('*')}

def refuse_service_call(*args, **kwargs):
    raise AssertionError(f'import attempted a service call: {args!r} {kwargs!r}')

socket.socket.connect = refuse_service_call
import dsio
after = {str(path.relative_to(state)) for path in state.rglob('*')}
print(json.dumps({
    'before': sorted(before),
    'after': sorted(after),
    'origin': dsio.__file__,
    'heavy': sorted({
        name.split('.', 1)[0]
        for name in sys.modules
        if name.split('.', 1)[0] in {'prefect', 'mlflow', 'torch', 'lightning', 'torchmetrics'}
    }),
}))
"""
    env = {
        **os.environ,
        "HOME": str(home),
        "PREFECT_HOME": str(prefect_home),
        "MLFLOW_TRACKING_URI": f"file:{mlflow_store}",
        "DSIO_IMPORT_STATE_ROOT": str(state),
        "PYTHONDONTWRITEBYTECODE": "1",
        "XDG_CACHE_HOME": str(state / "cache"),
        "XDG_CONFIG_HOME": str(state / "config"),
        "XDG_DATA_HOME": str(state / "data"),
        "TMPDIR": str(temp),
        "TEMP": str(temp),
        "TMP": str(temp),
    }
    completed = subprocess.run(
        [str(python), "-I", "-B", "-c", probe],
        cwd=work,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)

    assert result["before"] == result["after"]
    assert result["heavy"] == []
    assert Path(result["origin"]).is_relative_to(environment)

    flow_probe = """
from prefect import flow, task
from prefect.testing.utilities import prefect_test_harness
from dsio.contracts import sha256_of

@task
def identify_dataset(dataset):
    return sha256_of(dataset)

@flow
def project_flow():
    return identify_dataset({'name': 'algae', 'revision': 1})

with prefect_test_harness():
    print('DSIO_RESULT=' + project_flow())
"""
    flow_result = subprocess.run(
        [str(python), "-I", "-B", "-c", flow_probe],
        cwd=work,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert f"DSIO_RESULT={EXPECTED_DIGEST}" in flow_result.stdout
