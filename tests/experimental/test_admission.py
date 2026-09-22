"""Component admission checks only mechanically provable source rules."""

from __future__ import annotations

from pathlib import Path

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
        ("METRICS.register('x')(value)\n", "runtime-registration"),
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
