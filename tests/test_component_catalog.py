"""The component catalog is generated, complete and honest (Component Warehouse v1 CAP-8)."""

from __future__ import annotations

import doctest
import importlib
from pathlib import Path

import pytest
from tools import catalog

ENTRIES = catalog.collect()
EVIDENCE = catalog.load_yaml(catalog.EVIDENCE)
CANDIDATES = catalog.load_yaml(catalog.CANDIDATES)
PROVEN = [
    entry
    for entry in ENTRIES
    if entry.kind == "component" and not EVIDENCE.get(entry.reference, {}).get("legacy")
]


def test_catalog_rules_hold() -> None:
    assert catalog.validate(ENTRIES, EVIDENCE, CANDIDATES) == []


def test_committed_catalog_is_current() -> None:
    rendered = catalog.render(ENTRIES, EVIDENCE, CANDIDATES)
    assert catalog.CATALOG.read_text(encoding="utf-8") == rendered, (
        "docs/component-warehouse/catalog.md is stale; run `uv run python tools/catalog.py`"
    )


def test_every_kind_is_catalogued() -> None:
    kinds = {entry.kind for entry in ENTRIES}
    assert kinds == {"spine", "component", "dispatcher"}
    assert PROVEN, "at least the calibration and telemetry components have real uses"


@pytest.mark.parametrize("entry", PROVEN, ids=lambda entry: entry.reference)
def test_component_example_executes(entry: catalog.Entry) -> None:
    module = importlib.import_module(entry.module)
    example = entry.sections["Example"]
    parsed = doctest.DocTestParser().get_doctest(
        example, dict(vars(module)), entry.reference, entry.module, 0
    )
    executed = [item for item in parsed.examples if not item.options.get(doctest.SKIP, False)]
    assert executed, f"{entry.reference}: its Example must execute at least one statement"
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    runner.run(parsed)
    assert runner.summarize(verbose=False).failed == 0


def _component(reference: str, *, sections: dict[str, str] | None = None) -> catalog.Entry:
    full = {name: "documented" for name in catalog.SECTIONS}
    return catalog.Entry(reference, "component", "Summary.", full if sections is None else sections)


def _use(tmp_path: Path, group: str, kind: str = "real") -> dict[str, object]:
    consumer = tmp_path / group
    consumer.mkdir(exist_ok=True)
    test = tmp_path / f"test_{group}.py"
    test.touch()
    return {"consumer": group, "test": test.name, "kind": kind, "group": group}


def test_missing_evidence_and_sections_are_named(tmp_path: Path) -> None:
    entries = [
        _component("dsio.experimental.model.x:Undocumented", sections={}),
        _component("dsio.experimental.model.x:Unregistered"),
    ]
    evidence = {"dsio.experimental.model.x:Undocumented": {"uses": [_use(tmp_path, "a")]}}

    problems = catalog.validate(entries, evidence, {}, root=tmp_path)

    assert any("Undocumented: missing docstring sections" in problem for problem in problems)
    assert any("Unregistered: missing evidence entry" in problem for problem in problems)


def test_stable_component_needs_two_unrelated_real_uses_and_approval(tmp_path: Path) -> None:
    entry = _component("dsio.model.blocks:Encoder")
    same_group = {"uses": [_use(tmp_path, "cmi"), {**_use(tmp_path, "cmi"), "consumer": "cmi"}]}

    problems = catalog.validate([entry], {entry.reference: same_group}, {}, root=tmp_path)
    assert any("needs two unrelated real uses, has 1" in problem for problem in problems)
    assert any("needs a recorded approval" in problem for problem in problems)

    proven = {
        "uses": [_use(tmp_path, "cmi"), _use(tmp_path, "fog")],
        "approval": "PR #1 maintainer approval",
    }
    assert catalog.validate([entry], {entry.reference: proven}, {}, root=tmp_path) == []


def test_fixture_uses_do_not_count_and_legacy_must_be_experimental(tmp_path: Path) -> None:
    experimental = _component("dsio.experimental.model.x:OnlyFixtures")
    stable_legacy = _component("dsio.model.blocks:Legacy")
    evidence = {
        experimental.reference: {"uses": [_use(tmp_path, "synthetic", kind="fixture")]},
        stable_legacy.reference: {"legacy": True, "uses": []},
    }

    problems = catalog.validate([experimental, stable_legacy], evidence, {}, root=tmp_path)

    assert any("OnlyFixtures: experimental needs one real" in problem for problem in problems)
    assert any("Legacy: legacy components must live in dsio.experimental" in p for p in problems)


def test_paths_runs_and_unknown_entries_are_checked(tmp_path: Path) -> None:
    entry = _component("dsio.experimental.model.x:Block")
    bad = {
        "uses": [
            {
                "consumer": "missing/consumer",
                "test": "missing_test.py",
                "kind": "real",
                "group": "g",
                "runs": ["latest"],
            }
        ]
    }
    evidence = {entry.reference: bad, "dsio.experimental.model.x:Ghost": {"legacy": True}}

    problems = catalog.validate([entry], evidence, {}, root=tmp_path)

    assert any("consumer path 'missing/consumer' does not exist" in p for p in problems)
    assert any("test path 'missing_test.py' does not exist" in p for p in problems)
    assert any("run 'latest' is not an immutable MLflow run URI" in p for p in problems)
    assert any("Ghost: evidence entry names no public component" in p for p in problems)


def test_sections_parse_from_docstring_layout() -> None:
    doc = "Summary.\n\nConsumes:\n    x [B, C, T].\n\nExample:\n    >>> 1 + 1\n    2\n\nNotes here."
    sections = catalog.parse_sections(doc)
    assert sections["Consumes"] == "x [B, C, T]."
    assert sections["Example"] == ">>> 1 + 1\n2"
