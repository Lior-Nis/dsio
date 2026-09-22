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


def test_named_component_audit_and_enforcement_are_plain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dsio.experimental import admission

    candidate_source = tmp_path / "candidate.py"
    candidate_source.write_text("def component(): pass\n")

    def component() -> None:
        pass

    component.__module__ = "dsio.experimental.candidate"
    with monkeypatch.context() as patch:
        patch.setattr(admission, "component_source", lambda _module: candidate_source)
        patch.setattr(admission, "resolve_reference", lambda _reference: component)
        assert audit_component("dsio.experimental.candidate:component") == ()
    with pytest.raises(AdmissionError, match="importability"):
        require_admissible_component("dsio.experimental.missing:Component")


def test_admission_package_passes_its_own_static_policy() -> None:
    assert audit_component("dsio.experimental.admission:audit_source") == ()


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
    assert _audit(tmp_path, "if projection == 'pulse':\n    value = 1\n") == ()
    assert _audit(tmp_path, '"""Pulse transform."""\nvalue = project_axis + 1\n') == ()


def test_explicit_consumer_name_needs_no_project_identifier(tmp_path: Path) -> None:
    assert any(
        message.startswith("genericity:")
        for message in _audit(tmp_path, "if client == 'consumer_x':\n    value = 1\n")
    )


def test_explicit_consumer_name_in_helper_is_rejected(tmp_path: Path) -> None:
    source = """
def belongs(client):
    return client == 'consumer_x'
if belongs(client):
    value = 1
"""
    assert any(message.startswith("genericity:") for message in _audit(tmp_path, source))


def test_camel_case_project_identifier_is_project_context(tmp_path: Path) -> None:
    assert any(
        message.startswith("genericity:")
        for message in _audit(tmp_path, "if projectName == 'pulse':\n    value = 1\n")
    )


def test_mapping_project_access_is_project_context(tmp_path: Path) -> None:
    for source in (
        "if config['project'] == 'pulse':\n    value = 1\n",
        "if config.get('project') == 'pulse':\n    value = 1\n",
    ):
        assert any(message.startswith("genericity:") for message in _audit(tmp_path, source))


def test_project_alias_and_match_guard_are_rejected(tmp_path: Path) -> None:
    for source in (
        "p = project\nif p == 'pulse':\n    value = 1\n",
        "match project:\n    case item if item == 'pulse':\n        value = 1\n",
    ):
        assert any(message.startswith("genericity:") for message in _audit(tmp_path, source))


def test_project_name_constant_is_rejected(tmp_path: Path) -> None:
    source = "PROJECT_PULSE = 'pulse'\nif project == PROJECT_PULSE:\n    value = 1\n"
    assert any(message.startswith("genericity:") for message in _audit(tmp_path, source))
    assert any(
        message.startswith("genericity:")
        for message in _audit(tmp_path, "if project == 'pu' + 'lse':\n    value = 1\n")
    )


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


def test_importing_closed_registry_api_is_itself_rejected(tmp_path: Path) -> None:
    source = """
from dsio.eval.metrics import METRICS
def mutate(target):
    target.add('x', score)
mutate(METRICS)
"""
    assert any(
        message.startswith("runtime-registration:") for message in _audit(tmp_path, source)
    )


def test_registry_reexports_and_candidate_registries_are_rejected(tmp_path: Path) -> None:
    for source in (
        "from dsio.train.torch_task import TASKS\n",
        "REGISTRY = {}\nREGISTRY['x'] = component\n",
        "COMPONENTS = {}\ndef register(name):\n    COMPONENTS[name] = component\n",
        "HANDLERS = {}\ndef enroll(name):\n    HANDLERS[name] = component\n",
    ):
        assert any(
            message.startswith("runtime-registration:")
            for message in _audit(tmp_path, source)
        )


def test_registry_descendants_and_reflection_are_rejected(tmp_path: Path) -> None:
    for mutation in (
        "metrics.METRICS._items.update({'x': score})",
        "setattr(metrics.METRICS, '_items', {})",
    ):
        source = f"import dsio.eval.metrics as metrics\n{mutation}\n"
        assert any(
            message.startswith("runtime-registration:")
            for message in _audit(tmp_path, source)
        )


@pytest.mark.parametrize(
    "reference",
    [
        "mutate(metrics.METRICS)",
        "mutate(metrics.metric)",
        "getattr(metrics, 'METRICS').add('x', score)",
    ],
)
def test_closed_registry_references_cannot_be_passed_indirectly(
    tmp_path: Path, reference: str
) -> None:
    source = f"import dsio.eval.metrics as metrics\n{reference}\n"
    assert any(
        message.startswith("runtime-registration:") for message in _audit(tmp_path, source)
    )


def test_reflective_dsio_registry_access_is_rejected(tmp_path: Path) -> None:
    for reference in (
        "NAME = 'METRICS'\ngetattr(metrics, NAME).clear()",
        "metrics.__dict__['MET' + 'RICS'].clear()",
        "mutate(vars(metrics)['METRICS'])",
    ):
        source = f"import dsio.eval.metrics as metrics\n{reference}\n"
        assert any(
            message.startswith("runtime-registration:")
            for message in _audit(tmp_path, source)
        )


def test_generic_component_collections_are_not_registries(tmp_path: Path) -> None:
    source = """
from torch import nn
class Ensemble(nn.Module):
    def __init__(self, components):
        super().__init__()
        self.components = nn.ModuleList(components)
def apply_plugins(value, plugins):
    return tuple(plugin(value) for plugin in plugins)
"""
    assert _audit(tmp_path, source) == ()


@pytest.mark.parametrize(
    "source",
    [
        "import builtins\nbuiltins.__import__('consumer_x.private')\n",
        "import importlib\ngetattr(importlib, 'import_module')('consumer_x.private')\n",
        "import builtins\nbuiltins.eval(source)\n",
        "from builtins import exec as run\nrun(source)\n",
        "eval(source)\n",
    ],
)
def test_dynamic_import_and_code_evaluation_apis_are_rejected(
    tmp_path: Path, source: str
) -> None:
    assert any(message.startswith("dependency:") for message in _audit(tmp_path, source))


def test_star_import_is_not_statically_admissible(tmp_path: Path) -> None:
    failures = _audit(tmp_path, "from dsio.eval.metrics import *\nmetric('x')(score)\n")
    assert any(message.startswith("stable-contract:") for message in failures)


@pytest.mark.parametrize(
    "source",
    [
        "import importlib\nload = importlib.import_module\nload('consumer_x.private')\n",
        "import importlib\nimportlib.import_module(name='consumer_x.private')\n",
        "import importlib\nimportlib.import_module(module_name)\n",
    ],
)
def test_dynamic_import_aliases_keywords_and_unknown_targets_are_rejected(
    tmp_path: Path, source: str
) -> None:
    assert any(message.startswith("dependency:") for message in _audit(tmp_path, source))


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


def test_package_relative_helper_is_audited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dsio.experimental.admission import source as admission_source

    package = tmp_path / "experimental"
    candidate = package / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "__init__.py").write_text("from . import helper\n")
    (candidate / "helper.py").write_text("from consumer_x.private import Model\n")
    monkeypatch.setattr(admission_source, "_EXPERIMENTAL_ROOT", package)

    failures = audit_source(
        candidate / "__init__.py",
        module="dsio.experimental.candidate",
        project_names=("consumer_x",),
    )

    assert any(message.startswith("dependency:") for message in failures)


def test_package_source_wins_module_package_name_collision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dsio.experimental.admission import source as admission_source

    root = tmp_path / "experimental"
    package = root / "candidate"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("class Component: pass\n")
    (root / "candidate.py").write_text("class Component: pass\n")
    monkeypatch.setattr(admission_source, "_EXPERIMENTAL_ROOT", root)

    assert admission_source.component_source("dsio.experimental.candidate") == (
        package / "__init__.py"
    )


def test_one_project_name_string_is_normalized(tmp_path: Path) -> None:
    path = tmp_path / "candidate.py"
    path.write_text("if client == 'consumer_x':\n    value = 1\n")

    failures = audit_source(
        path, module="dsio.experimental.candidate", project_names="consumer_x"
    )

    assert any(message.startswith("genericity:") for message in failures)


@pytest.mark.parametrize(
    ("path", "module", "project_names"),
    [
        (None, "dsio.experimental.candidate", ()),
        ("candidate.py", None, ()),
        ("candidate.py", "dsio.experimental.candidate", b"consumer_x"),
        ("candidate.py", "dsio.experimental.candidate", ("consumer_x", 1)),
        ("bad\0.py", "dsio.experimental.candidate", ()),
    ],
)
def test_malformed_audit_inputs_return_an_input_rule(
    path: Any, module: Any, project_names: Any
) -> None:
    failures = audit_source(path, module=module, project_names=project_names)

    assert failures[0].startswith(("input:", "source:"))


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
