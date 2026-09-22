"""The built artifacts, not the checkout, define DSio's install boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DEPENDENCIES = {
    "prefect": ">=3.8,<4",
    "torch": ">=2.7,<3",
    "lightning": ">=2.5,<3",
    "torchmetrics": ">=1.7,<2",
    "mlflow": ">=3,<4",
}
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

    raw_requirements = metadata.get_all("Requires-Dist", [])
    requirements = {
        requirement.name.lower(): requirement
        for item in raw_requirements
        if (requirement := Requirement(item)).name.lower() in REQUIRED_DEPENDENCIES
    }

    assert "dsio/__init__.py" in members
    assert "dsio/tracking/__init__.py" in members
    assert "dsio/tracking/attempt.py" in members
    assert "dsio/tracking/cache.py" in members
    assert "dsio/tracking/evidence/__init__.py" in members
    assert "dsio/tracking/evidence/references.py" in members
    assert "dsio/tracking/evidence/resolution.py" in members
    assert "dsio/tracking/provenance.py" in members
    assert "dsio/experimental/__init__.py" in members
    assert "dsio/experimental/admission/__init__.py" in members
    assert "dsio/experimental/admission/source.py" in members
    assert "dsio/experimental/admission/syntax/__init__.py" in members
    assert "dsio/experimental/admission/syntax/imports/__init__.py" in members
    assert "dsio/experimental/admission/syntax/imports/dynamic.py" in members
    assert "dsio/experimental/admission/syntax/imports/names.py" in members
    assert "dsio/experimental/admission/syntax/imports/resolution.py" in members
    assert "dsio/experimental/admission/syntax/projects.py" in members
    assert "dsio/experimental/admission/syntax/registries/__init__.py" in members
    assert "dsio/experimental/admission/syntax/registries/escaping.py" in members
    assert "dsio/experimental/admission/syntax/registries/initializers.py" in members
    assert "dsio/experimental/admission/syntax/registries/known.py" in members
    assert "dsio/experimental/admission/syntax/registries/parameters.py" in members
    assert "dsio/experimental/admission/syntax/registries/surfaces.py" in members
    assert set(requirements) == set(REQUIRED_DEPENDENCIES)
    for name, expected_specifier in REQUIRED_DEPENDENCIES.items():
        requirement = requirements[name]
        assert str(requirement.specifier) == str(Requirement(f"x{expected_specifier}").specifier)
        assert requirement.marker is None
        assert requirement.url is None
        assert not requirement.extras

    names = {Requirement(item).name.lower() for item in raw_requirements}
    assert "mlflow-skinny" not in names
    assert not any("pytorch-cpu" in requirement for requirement in raw_requirements)
    assert SpecifierSet(metadata["Requires-Python"]) == SpecifierSet(">=3.12,<3.15")
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
    built_artifacts: tuple[Path, Path],
    tmp_path: Path,
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
    uv_cache = state / "uv-cache"
    install_env = {
        key: value for key, value in os.environ.items() if not key.startswith(("UV_", "PIP_"))
    }
    install_env["UV_CACHE_DIR"] = str(uv_cache)

    _run(
        "uv",
        "venv",
        "--no-config",
        "--cache-dir",
        str(uv_cache),
        "--python",
        sys.executable,
        str(environment),
        cwd=work,
        env=install_env,
    )
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run(
        "uv",
        "pip",
        "install",
        "--no-config",
        "--cache-dir",
        str(uv_cache),
        "--python",
        str(python),
        "--default-index",
        "https://download.pytorch.org/whl/cpu",
        "torch>=2.7,<3",
        cwd=work,
        env=install_env,
    )
    _run(
        "uv",
        "pip",
        "install",
        "--no-config",
        "--cache-dir",
        str(uv_cache),
        "--python",
        str(python),
        "--default-index",
        "https://pypi.org/simple",
        str(wheel),
        cwd=work,
        env=install_env,
    )
    _run(
        "uv",
        "pip",
        "check",
        "--no-config",
        "--cache-dir",
        str(uv_cache),
        "--python",
        str(python),
        cwd=work,
        env=install_env,
    )

    probe = """
import json
import pathlib
import socket
import sys

state = pathlib.Path(__import__('os').environ['DSIO_IMPORT_STATE_ROOT'])
environment = pathlib.Path(__import__('os').environ['DSIO_IMPORT_ENV_ROOT'])

def snapshot(root):
    return {
        str(path.relative_to(root)): (details.st_mode, details.st_size, details.st_mtime_ns)
        for path in root.rglob('*')
        if (details := path.stat())
    }

state_before = snapshot(state)
environment_before = snapshot(environment)

def refuse_service_call(*args, **kwargs):
    raise AssertionError(f'import attempted a service call: {args!r} {kwargs!r}')

socket.socket.connect = refuse_service_call
import dsio
from dsio.experimental import AdmissionError, audit_component, audit_source
print(json.dumps({
    'state_unchanged': state_before == snapshot(state),
    'environment_unchanged': environment_before == snapshot(environment),
    'origin': dsio.__file__,
    'experimental_api': [AdmissionError.__name__, audit_component.__name__, audit_source.__name__],
    'heavy': sorted({
        name.split('.', 1)[0]
        for name in sys.modules
        if name.split('.', 1)[0] in {'prefect', 'mlflow', 'torch', 'lightning', 'torchmetrics'}
    }),
}))
"""
    env = {
        **{
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("PREFECT_", "MLFLOW_"))
        },
        "HOME": str(home),
        "PREFECT_HOME": str(prefect_home),
        "PREFECT_SERVER_ANALYTICS_ENABLED": "false",
        "DO_NOT_TRACK": "1",
        "MLFLOW_TRACKING_URI": f"file:{mlflow_store}",
        "MLFLOW_ALLOW_FILE_STORE": "true",
        "DSIO_IMPORT_ENV_ROOT": str(environment),
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

    assert result["state_unchanged"]
    assert result["environment_unchanged"]
    assert result["heavy"] == []
    assert result["experimental_api"] == ["AdmissionError", "audit_component", "audit_source"]
    assert Path(result["origin"]).is_relative_to(environment)

    flow_probe = """
from prefect import flow, task
from prefect.testing.utilities import prefect_test_harness
from mlflow import MlflowClient
from dsio.contracts import sha256_of
from dsio.tracking import attempt, evidence_uri, experiment, record_provenance, resolve_evidence

@task
def identify_dataset(dataset, parent_run_id):
    with attempt(parent_run_id) as child:
        digest = sha256_of(dataset)
        record_provenance(
            child.info.run_id,
            {'dataset': dataset, 'seed': 7},
            components={'identity': 'dsio.contracts:sha256_of'},
        )
        MlflowClient().log_param(child.info.run_id, 'dataset_digest', digest)
        MlflowClient().log_dict(
            child.info.run_id,
            {'dataset_digest': digest},
            'outputs/dataset.json',
        )
        return digest, child.info.run_id

@flow
def project_flow():
    with experiment('installed-wheel-flow') as parent:
        digest, child_run_id = identify_dataset(
            {'name': 'algae', 'revision': 1}, parent.info.run_id
        )
        assert digest == '2ca217b09c12c36031e1078aab6a81e705f1281b1eb841c876f98cef5cb03556'
        return digest, parent.info.run_id, child_run_id

with prefect_test_harness():
    digest, parent_run_id, child_run_id = project_flow()
client = MlflowClient()
print('DSIO_RESULT=' + digest)
print('DSIO_PARENT_STATUS=' + client.get_run(parent_run_id).info.status)
print('DSIO_CHILD_STATUS=' + client.get_run(child_run_id).info.status)
print('DSIO_CHILD_IDENTITY_LENGTH=' + str(
    len(client.get_run(child_run_id).data.tags['dsio.execution_identity'])
))
print('DSIO_CHILD_PARENT_MATCH=' + str(
    client.get_run(child_run_id).data.tags['mlflow.parentRunId'] == parent_run_id
))
identity = client.get_run(child_run_id).data.params['dsio.execution_identity']
resolved = resolve_evidence(identity, required_artifacts={'outputs/dataset.json'})
print('DSIO_REUSED_RUN_MATCH=' + str(resolved.info.run_id == child_run_id))
print('DSIO_REUSED_URI=' + evidence_uri(resolved.info.run_id, 'outputs/dataset.json'))
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
    assert "DSIO_PARENT_STATUS=FINISHED" in flow_result.stdout
    assert "DSIO_CHILD_STATUS=FINISHED" in flow_result.stdout
    assert "DSIO_CHILD_IDENTITY_LENGTH=64" in flow_result.stdout
    assert "DSIO_CHILD_PARENT_MATCH=True" in flow_result.stdout
    assert "DSIO_REUSED_RUN_MATCH=True" in flow_result.stdout
    assert "DSIO_REUSED_URI=runs:/" in flow_result.stdout
