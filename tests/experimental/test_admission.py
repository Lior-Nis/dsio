"""Component admission checks only mechanically provable source rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dsio.experimental import (
    AdmissionError,
    audit_component,
    audit_source,
    require_admissible_component,
)


def _audit(
    tmp_path: Path, source: str, *, module: str = "dsio.experimental.candidate"
) -> tuple[str, ...]:
    path = tmp_path / "candidate.py"
    path.write_text(source)
    return audit_source(path, module=module, project_names=("consumer_x",))


def test_generic_public_component_source_passes(tmp_path: Path) -> None:
    assert (
        _audit(
            tmp_path,
            """
import numpy as np
from torch import nn

class Center(nn.Module):
    def forward(self, value):
        return value - np.asarray(value).mean()
""",
        )
        == ()
    )


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ("from consumer_x.private import Model\n", "dependency"),
        ("from dsio.train._internal import run\n", "stable-contract"),
        ("if project == 'Pulse':\n    value = 1\n", "genericity"),
        (
            "from dsio.eval.metrics import METRICS\nMETRICS.register('x')(value)\n",
            "runtime-registration",
        ),
        ("from dsio.config.registry import Registry\n", "runtime-registration"),
    ],
)
def test_admission_reports_each_mechanical_rule(tmp_path: Path, source: str, rule: str) -> None:
    assert any(message.startswith(f"{rule}:") for message in _audit(tmp_path, source))


def test_component_must_start_in_experimental_namespace(tmp_path: Path) -> None:
    failures = _audit(tmp_path, "value = 1\n", module="dsio.model.candidate")

    assert failures == (
        "namespace: proposed component module 'dsio.model.candidate' must live under "
        "'dsio.experimental'",
    )


def test_named_component_audit_and_enforcement_are_plain(tmp_path: Path) -> None:
    assert audit_component("dsio.experimental.admission:audit_source") == ()
    with pytest.raises(AdmissionError, match="importability"):
        require_admissible_component("dsio.experimental.missing:Component")


@pytest.mark.parametrize(
    "source",
    [
        "from dsio import _private\n",
        "from . import _private\n",
    ],
)
def test_from_import_cannot_hide_private_dsio_dependency(
    tmp_path: Path, source: str
) -> None:
    assert any(message.startswith("stable-contract:") for message in _audit(tmp_path, source))


@pytest.mark.parametrize(
    "source",
    [
        "match project:\n    case 'pulse':\n        value = 1\n",
        "[value for value in values if project == 'algae']\n",
    ],
)
def test_all_branch_forms_reject_consumer_project_names(tmp_path: Path, source: str) -> None:
    assert any(message.startswith("genericity:") for message in _audit(tmp_path, source))


def test_indirect_project_flag_is_rejected(tmp_path: Path) -> None:
    failures = _audit(
        tmp_path,
        "is_pulse = project == 'pulse'\nif is_pulse:\n    value = 1\n",
    )
    assert any(message.startswith("genericity:") for message in failures)


def test_domain_value_named_like_project_is_not_project_branching(tmp_path: Path) -> None:
    assert _audit(tmp_path, "if signal_kind == 'pulse':\n    value = 1\n") == ()


@pytest.mark.parametrize(
    "source",
    [
        "from dsio.eval.metrics import metric\n@metric('x')\ndef score(): pass\n",
        "from dsio.train.runner import runner as register\n@register('x')\ndef run(): pass\n",
        "from dsio.eval.metrics import METRICS\n"
        "register = METRICS.register\nregister('x')(score)\n",
    ],
)
def test_aliases_cannot_hide_closed_dispatcher_mutation(tmp_path: Path, source: str) -> None:
    assert any(
        message.startswith("runtime-registration:") for message in _audit(tmp_path, source)
    )


@pytest.mark.parametrize(
    "source",
    [
        "import importlib as loader\nloader.import_module('consumer_x.private')\n",
        "from importlib import import_module as load\nload('dsio._private')\n",
        "__import__('consumer_x.private')\n",
    ],
)
def test_literal_dynamic_imports_follow_dependency_rules(tmp_path: Path, source: str) -> None:
    failures = _audit(tmp_path, source)
    assert any(
        message.startswith(("dependency:", "stable-contract:")) for message in failures
    )


def test_unrelated_register_method_is_not_a_registry_mutation(tmp_path: Path) -> None:
    assert _audit(tmp_path, "import atexit\natexit.register(cleanup)\n") == ()


@pytest.mark.parametrize(
    "mutation",
    [
        "METRICS['x'] = score",
        "METRICS.update({'x': score})",
        "METRICS.add('x', score)",
    ],
)
def test_known_registry_mutation_forms_are_rejected(
    tmp_path: Path, mutation: str
) -> None:
    source = f"from dsio.eval.metrics import METRICS\n{mutation}\n"
    assert any(
        message.startswith("runtime-registration:") for message in _audit(tmp_path, source)
    )


def test_imported_experimental_helpers_are_audited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = tmp_path / "experimental"
    package.mkdir()
    (package / "candidate.py").write_text("from dsio.experimental import helper\n")
    (package / "helper.py").write_text("from consumer_x.private import Model\n")
    from dsio.experimental.admission import source as admission_source

    monkeypatch.setattr(admission_source, "_EXPERIMENTAL_ROOT", package)

    failures = audit_source(
        package / "candidate.py",
        module="dsio.experimental.candidate",
        project_names=("consumer_x",),
    )

    assert any(message.startswith("dependency:") for message in failures)


def test_named_audit_validates_before_importing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dsio.experimental import admission

    candidate = tmp_path / "candidate.py"
    candidate.write_text("from consumer_x.private import Model\n")
    imported = False

    def resolve(_reference: str) -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("unsafe source was imported")

    monkeypatch.setattr(admission, "component_source", lambda _module: candidate)
    monkeypatch.setattr(admission, "resolve_reference", resolve)

    assert audit_component("dsio.experimental.candidate:Component", project_names=("consumer_x",))
    assert not imported


def test_named_audit_translates_import_time_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dsio.experimental import admission

    candidate = tmp_path / "candidate.py"
    candidate.write_text("class Component: pass\n")
    monkeypatch.setattr(admission, "component_source", lambda _module: candidate)

    def fail(_reference: str) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(admission, "resolve_reference", fail)

    assert audit_component("dsio.experimental.candidate:Component") == (
        "importability: could not resolve component "
        "'dsio.experimental.candidate:Component': boom",
    )


def test_named_audit_rejects_malformed_runtime_value() -> None:
    assert audit_component(None) == (  # type: ignore[arg-type]
        "importability: component reference must be a non-empty module:qualname string",
    )


def test_admission_policy_records_review_and_versioning_requirements() -> None:
    policy = Path("docs/component-admission.md").read_text().casefold()

    for requirement in (
        "unit tests",
        "integration",
        "deterministic",
        "provenance",
        "downstream",
        "adversarial agent review",
        "explicit human approval",
        "unrelated second",
        "major release",
        "migration notes",
        "never silently",
    ):
        assert requirement in policy
