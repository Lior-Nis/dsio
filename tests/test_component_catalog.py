"""The component catalog is generated, complete and honest (Component Warehouse v1 CAP-8)."""

from __future__ import annotations

import doctest
import importlib
import importlib.util
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
def test_component_example_calls_the_component_and_executes(entry: catalog.Entry) -> None:
    module = importlib.import_module(entry.module)
    parsed = doctest.DocTestParser().get_doctest(
        entry.sections["Example"], dict(vars(module)), entry.reference, entry.module, 0
    )
    executed = [item for item in parsed.examples if not item.options.get(doctest.SKIP, False)]
    name = entry.qualname.split(".")[-1]
    assert any(f"{name}(" in item.source for item in executed), (
        f"{entry.reference}: its Example must actually call {name}"
    )
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    runner.run(parsed)
    assert runner.summarize(verbose=False).failed == 0


def test_a_stable_module_outside_the_spine_list_is_checked_as_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(catalog, "SPINE_MODULES", catalog.SPINE_MODULES - {"dsio.eval"})
    entries = catalog.collect()
    evaluate = next(entry for entry in entries if entry.reference.endswith(":evaluate"))

    assert evaluate.kind == "component"
    problems = catalog.validate(entries, EVIDENCE, CANDIDATES)
    assert any(":evaluate: missing evidence entry" in problem for problem in problems)


# --- validator rules on synthetic inputs ----------------------------------------------


def _component(reference: str, *, sections: dict[str, str] | None = None) -> catalog.Entry:
    full = {name: "documented" for name in catalog.SECTIONS}
    return catalog.Entry(reference, "component", "Summary.", full if sections is None else sections)


def _consumer(root: Path, path: str, mentions: str) -> dict[str, object]:
    directory = root / path
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tasks.py").write_text(f"from somewhere import {mentions}\n", encoding="utf-8")
    test = root / f"test_{path.replace('/', '_')}.py"
    test.touch()
    return {"consumer": path, "test": test.name, "kind": "real"}


def test_missing_evidence_and_sections_are_named(tmp_path: Path) -> None:
    entries = [
        _component("dsio.experimental.model.x:Undocumented", sections={}),
        _component("dsio.experimental.model.x:Unregistered"),
    ]
    use = _consumer(tmp_path, "reference_projects/kaggle/a", "Undocumented")
    evidence = {"dsio.experimental.model.x:Undocumented": {"uses": [use]}}

    problems = catalog.validate(entries, evidence, {}, root=tmp_path)

    assert any("Undocumented: missing docstring sections" in problem for problem in problems)
    assert any("Unregistered: missing evidence entry" in problem for problem in problems)


def test_groups_derive_from_consumer_paths(tmp_path: Path) -> None:
    entry = _component("dsio.model.blocks:Encoder")
    same_competition = {
        "uses": [
            _consumer(tmp_path, "reference_projects/kaggle/cmi", "Encoder"),
            _consumer(tmp_path, "reference_projects/kaggle/cmi/sequence", "Encoder"),
        ],
        "approval": "PR #1 maintainer approval",
    }

    problems = catalog.validate([entry], {entry.reference: same_competition}, {}, root=tmp_path)
    assert any("needs two unrelated real uses, has 1" in problem for problem in problems)

    unrelated = {
        "uses": [
            _consumer(tmp_path, "reference_projects/kaggle/cmi", "Encoder"),
            _consumer(tmp_path, "reference_projects/kaggle/fog", "Encoder"),
        ],
        "approval": "PR #1 maintainer approval",
    }
    assert catalog.validate([entry], {entry.reference: unrelated}, {}, root=tmp_path) == []

    no_approval = {"uses": unrelated["uses"]}
    problems = catalog.validate([entry], {entry.reference: no_approval}, {}, root=tmp_path)
    assert any("needs a recorded approval" in problem for problem in problems)


def test_real_uses_must_be_kaggle_consumers_that_reference_the_component(tmp_path: Path) -> None:
    entry = _component("dsio.experimental.model.x:Block")
    synthetic = _consumer(tmp_path, "reference_projects/supervised", "Block")
    silent = _consumer(tmp_path, "reference_projects/kaggle/titanic", "SomethingElse")
    evidence = {entry.reference: {"uses": [synthetic, silent]}}

    problems = catalog.validate([entry], evidence, {}, root=tmp_path)

    assert any("a real use must be a Kaggle consumer" in problem for problem in problems)
    assert any("never references 'Block'" in problem for problem in problems)
    assert any("experimental needs one real downstream use" in problem for problem in problems)


def test_via_names_the_component_the_consumer_references(tmp_path: Path) -> None:
    entry = _component("dsio.experimental.x:inner")
    use = {
        **_consumer(tmp_path, "reference_projects/kaggle/fog", "outer"),
        "via": "dsio.experimental.x:outer",
    }
    assert catalog.validate([entry], {entry.reference: {"uses": [use]}}, {}, root=tmp_path) == []


def test_legacy_is_frozen_and_lost_on_first_real_use(tmp_path: Path) -> None:
    newcomer = _component("dsio.experimental.model.x:Newcomer", sections={})
    promoted = _component("dsio.experimental.model.x:Frozen", sections={})
    stable = _component("dsio.model.blocks:Legacy")
    evidence = {
        newcomer.reference: {"legacy": True, "uses": []},
        promoted.reference: {
            "legacy": True,
            "uses": [_consumer(tmp_path, "reference_projects/kaggle/rogii", "Frozen")],
        },
        stable.reference: {"legacy": True, "uses": []},
    }
    frozen = frozenset({promoted.reference, stable.reference})

    problems = catalog.validate(
        [newcomer, promoted, stable], evidence, {}, root=tmp_path, legacy=frozen
    )

    assert any("Newcomer: only frozen pre-1.0 components may be legacy" in p for p in problems)
    assert any("Frozen: a legacy component with a real use must drop legacy" in p for p in problems)


def test_paths_runs_and_unknown_entries_are_checked(tmp_path: Path) -> None:
    entry = _component("dsio.experimental.model.x:Block")
    bad = {
        "uses": [
            {
                "consumer": "missing/consumer",
                "test": "missing_test.py",
                "kind": "real",
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
    doc = (
        "Summary.\n\nConsumes:\n    x [B, C, T].\n\nExample:\n    >>> 1 + 1\n    2\n\n"
        "    >>> 2 + 2\n    4\nNotes:\n    ignored."
    )
    sections, problems = catalog.parse_sections(doc)
    assert sections["Consumes"] == "x [B, C, T]."
    assert sections["Example"] == ">>> 1 + 1\n2\n\n>>> 2 + 2\n4"
    assert problems == []


def test_repeated_sections_are_reported() -> None:
    _, problems = catalog.parse_sections("Consumes:\n    a\nConsumes:\n    b\n")
    assert problems == ["section 'Consumes' appears more than once"]


def test_public_names_missing_from_all_are_reported(tmp_path: Path) -> None:
    source = tmp_path / "experimental_block.py"
    source.write_text('__all__ = ["Declared"]\n\nclass Declared: ...\n\ndef forgotten(): ...\n')
    spec = importlib.util.spec_from_file_location("dsio.experimental.fake.block", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert catalog._undeclared_public_names(module) == [
        "dsio.experimental.fake.block: public name 'forgotten' is not in __all__"
    ]
