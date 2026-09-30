"""Generate and check the DSio component catalog.

The catalog is how a consumer finds a lego block before writing one. It is generated
from three sources so it cannot drift from the code:

- the public ``__all__`` of the spine modules and the ``dsio.experimental`` packages,
  plus the closed dispatchers (split algorithms, ``METRICS``);
- each component's docstring sections (Consumes, Produces, Parameters, Devices,
  Limitations, Example);
- ``docs/component-warehouse/evidence.yaml``, which records who uses each component and
  whether it is legacy. It lives outside ``src/dsio`` because the admission audit rejects
  consumer names in DSio source;
- ``docs/component-warehouse/candidates.yaml``: consumer-local code awaiting a block.

``python tools/catalog.py`` rewrites ``docs/component-warehouse/catalog.md``;
``python tools/catalog.py --check`` fails on any rule violation or on a stale catalog.
This is repository tooling, not part of the wheel.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT / "docs" / "component-warehouse"
CATALOG = WAREHOUSE / "catalog.md"
EVIDENCE = WAREHOUSE / "evidence.yaml"
CANDIDATES = WAREHOUSE / "candidates.yaml"

# Modules whose public names are the spine: composition roots, loaders, export and
# evidence machinery. They are catalogued for discovery but are not warehouse components.
SPINE_MODULES = (
    "dsio.data.loading",
    "dsio.data.splits",
    "dsio.eval",
    "dsio.inference",
    "dsio.model.module",
    "dsio.train.artifacts",
    "dsio.train.capabilities",
    "dsio.train.trainer",
)
SECTIONS = ("Consumes", "Produces", "Parameters", "Devices", "Limitations", "Example")
USE_KINDS = ("real", "fixture")
_RUN_URI = re.compile(
    r"^(runs:/[0-9a-f]{32}/\S+|https?://\S+/#/experiments/\d+/runs/[0-9a-f]{32})$"
)
_SECTION = re.compile(r"^(?P<name>[A-Z][a-z]+):\s*$")


@dataclass(frozen=True)
class Entry:
    """One catalogued public object."""

    reference: str
    kind: str  # "spine", "component" or "dispatcher"
    summary: str
    sections: Mapping[str, str] = field(default_factory=dict)

    @property
    def module(self) -> str:
        return self.reference.partition(":")[0]

    @property
    def experimental(self) -> bool:
        return self.module.startswith("dsio.experimental.")

    @property
    def package(self) -> str:
        parts = self.module.split(".")
        return ".".join(parts[:3] if self.experimental else parts[:2])


def collect() -> list[Entry]:
    """Every public spine object, experimental component and dispatcher entry."""
    entries: dict[str, Entry] = {}
    for name in SPINE_MODULES:
        for reference, value in _public(importlib.import_module(name)):
            entries.setdefault(reference, Entry(reference, "spine", _summary(value)))
    import dsio.experimental

    for info in pkgutil.walk_packages(dsio.experimental.__path__, prefix="dsio.experimental."):
        if ".admission" in info.name:
            continue  # the auditor, not a component
        for reference, value in _public(importlib.import_module(info.name)):
            doc = inspect.getdoc(value) or ""
            entries.setdefault(
                reference, Entry(reference, "component", _summary(value), parse_sections(doc))
            )
    from dsio.data.splits.models.manifest import STABLE_ALGORITHMS
    from dsio.eval.metrics import METRICS

    for algorithm in sorted(STABLE_ALGORITHMS):
        reference = f"dsio.data.splits:generate[{algorithm}]"
        entries[reference] = Entry(reference, "dispatcher", "Split algorithm (closed dispatcher).")
    for metric in sorted(METRICS.names()):
        reference = f"dsio.eval.metrics:METRICS[{metric}]"
        entries[reference] = Entry(
            reference, "dispatcher", "Evaluation metric (closed dispatcher)."
        )
    return sorted(entries.values(), key=lambda entry: entry.reference)


def parse_sections(doc: str) -> dict[str, str]:
    """Return ``{section: body}`` for the catalog headers found in a docstring."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in doc.splitlines():
        match = _SECTION.match(line.strip()) if not line.startswith(" ") else None
        if match and match.group("name") in SECTIONS:
            current = match.group("name")
            sections[current] = []
        elif current is not None:
            if line and not line.startswith(" "):
                current = None  # a dedented line ends the section
            else:
                sections[current].append(line[4:] if line.startswith("    ") else line.strip())
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    return dict(data or {})


def validate(
    entries: Sequence[Entry],
    evidence: Mapping[str, Any],
    candidates: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> list[str]:
    """Every rule the catalog enforces, as human-readable problems."""
    problems: list[str] = []
    known = {entry.reference: entry for entry in entries}
    for reference in sorted(set(evidence) - set(known)):
        problems.append(f"{reference}: evidence entry names no public component")
    for entry in entries:
        if entry.kind != "component":
            continue
        record = evidence.get(entry.reference)
        if not isinstance(record, Mapping):
            problems.append(f"{entry.reference}: missing evidence entry")
            continue
        legacy = bool(record.get("legacy", False))
        uses = record.get("uses") or []
        real_groups = set()
        for index, use in enumerate(uses):
            where = f"{entry.reference}: use {index}"
            if not isinstance(use, Mapping):
                problems.append(f"{where}: must be a mapping")
                continue
            if use.get("kind") not in USE_KINDS:
                problems.append(f"{where}: kind must be one of {USE_KINDS}")
            for key in ("consumer", "test"):
                path = use.get(key)
                if not isinstance(path, str) or not (root / path).exists():
                    problems.append(f"{where}: {key} path {path!r} does not exist")
            for uri in use.get("runs") or []:
                if not isinstance(uri, str) or not _RUN_URI.match(uri):
                    problems.append(f"{where}: run {uri!r} is not an immutable MLflow run URI")
            if use.get("kind") == "real":
                group = use.get("group")
                if not isinstance(group, str) or not group:
                    problems.append(f"{where}: a real use needs an unrelated-use group")
                else:
                    real_groups.add(group)
        if legacy:
            if not entry.experimental:
                problems.append(
                    f"{entry.reference}: legacy components must live in dsio.experimental"
                )
            continue
        missing = [name for name in SECTIONS if not entry.sections.get(name)]
        if missing:
            problems.append(f"{entry.reference}: missing docstring sections {missing}")
        if entry.experimental and not real_groups:
            problems.append(f"{entry.reference}: experimental needs one real downstream use")
        if not entry.experimental:
            if len(real_groups) < 2:
                problems.append(
                    f"{entry.reference}: a stable component needs two unrelated real uses, "
                    f"has {len(real_groups)}"
                )
            if not record.get("approval"):
                problems.append(f"{entry.reference}: a stable component needs a recorded approval")
    for name, candidate in candidates.items():
        if not isinstance(candidate, Mapping):
            problems.append(f"candidate {name}: must be a mapping")
            continue
        for path in candidate.get("consumers") or []:
            if not (root / path).exists():
                problems.append(f"candidate {name}: consumer path {path!r} does not exist")
        if not candidate.get("reason"):
            problems.append(f"candidate {name}: needs a reason")
    return problems


def render(
    entries: Sequence[Entry], evidence: Mapping[str, Any], candidates: Mapping[str, Any]
) -> str:
    """The catalog document, grouped by maturity and pipeline package."""
    lines = [
        "# DSio component catalog",
        "",
        "<!-- Generated by `uv run python tools/catalog.py`; do not edit by hand. -->",
        "",
        "Find a block here before writing one. **Maturity is location**: `dsio.experimental`",
        "components carry no compatibility promise; *legacy* ones have no real downstream use",
        "yet and are deleted at 1.0 if still unproven (`docs/component-admission.md`).",
        "Conventions every block follows: [conventions.md](conventions.md).",
        "",
    ]
    components = [entry for entry in entries if entry.kind == "component"]
    proven = [e for e in components if not evidence.get(e.reference, {}).get("legacy")]
    legacy = [e for e in components if evidence.get(e.reference, {}).get("legacy")]
    lines += ["## Components", ""]
    if not proven:
        lines += ["_No component has a recorded real use yet._", ""]
    for entry in proven:
        lines += _render_component(entry, evidence.get(entry.reference, {}))
    lines += [
        "## Legacy experimental components",
        "",
        "No real downstream use yet. Warehouse stories reshape them when a consumer first",
        "needs them; full sections arrive with that first use.",
        "",
        "| Component | Summary | Fixture uses |",
        "|---|---|---|",
    ]
    for entry in legacy:
        uses = evidence.get(entry.reference, {}).get("uses") or []
        fixtures = ", ".join(sorted({Path(use["consumer"]).name for use in uses})) or "—"
        lines.append(f"| `{entry.reference}` | {_cell(entry.summary)} | {fixtures} |")
    lines += [
        "",
        "## Spine",
        "",
        "Composition roots and evidence machinery every block plugs into.",
        "",
    ]
    for package in sorted({entry.package for entry in entries if entry.kind == "spine"}):
        lines += [f"### `{package}`", "", "| Name | Summary |", "|---|---|"]
        for entry in entries:
            if entry.kind == "spine" and entry.package == package:
                lines.append(f"| `{entry.reference}` | {_cell(entry.summary)} |")
        lines.append("")
    lines += ["## Closed dispatchers", "", "| Entry | Kind |", "|---|---|"]
    for entry in entries:
        if entry.kind == "dispatcher":
            lines.append(f"| `{entry.reference}` | {entry.summary} |")
    lines += [
        "",
        "## Admission candidates",
        "",
        "Consumer-local code that a warehouse block is planned to replace, or that stays",
        "local on purpose.",
        "",
        "| Candidate | Consumers | Reason |",
        "|---|---|---|",
    ]
    for name, candidate in sorted(candidates.items()):
        consumers = ", ".join(f"`{path}`" for path in candidate.get("consumers") or [])
        lines.append(f"| {name} | {consumers} | {_cell(str(candidate.get('reason', '')))} |")
    return "\n".join(lines) + "\n"


def _render_component(entry: Entry, record: Mapping[str, Any]) -> list[str]:
    uses = record.get("uses") or []
    groups = sorted({use["group"] for use in uses if use.get("kind") == "real"})
    maturity = "experimental" if entry.experimental else "stable"
    lines = [
        f"### `{entry.reference}`",
        "",
        f"{entry.summary} — **{maturity}**; real uses: {len(groups)} "
        f"({', '.join(groups) or 'none'}).",
        "",
    ]
    for name in SECTIONS:
        body = entry.sections.get(name, "")
        if name == "Example":
            lines += [f"**{name}**", "", "```python", body, "```", ""]
        else:
            lines += [f"**{name}**: {body}", ""]
    lines += ["**Evidence**:", ""]
    for use in uses:
        runs = ", ".join(use.get("runs") or []) or "—"
        lines.append(f"- {use['kind']}: `{use['consumer']}` (test `{use['test']}`; runs {runs})")
    return [*lines, ""]


def _public(module: Any) -> list[tuple[str, Any]]:
    found = []
    for name in getattr(module, "__all__", ()):
        value = getattr(module, name)
        if inspect.isclass(value) and issubclass(value, BaseException):
            continue  # errors are part of an API, not blocks
        if not (inspect.isclass(value) or inspect.isfunction(value)):
            continue  # type aliases
        found.append((f"{value.__module__}:{value.__qualname__}", value))
    return found


def _summary(value: Any) -> str:
    doc = inspect.getdoc(value) or ""
    return doc.strip().splitlines()[0] if doc.strip() else ""


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail on problems or drift")
    args = parser.parse_args(argv)
    entries = collect()
    evidence = load_yaml(EVIDENCE)
    candidates = load_yaml(CANDIDATES)
    problems = validate(entries, evidence, candidates)
    rendered = render(entries, evidence, candidates)
    if args.check:
        current = CATALOG.read_text(encoding="utf-8") if CATALOG.exists() else None
        if current != rendered:
            problems.append(f"{CATALOG.relative_to(ROOT)} is stale; run tools/catalog.py")
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0
    CATALOG.write_text(rendered, encoding="utf-8")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
