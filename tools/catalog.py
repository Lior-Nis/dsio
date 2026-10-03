"""Generate and check the DSio component catalog.

The catalog is how a consumer finds a lego block before writing one. It is generated so
it cannot drift from the code:

- every ``dsio`` module's public ``__all__``, classified as **spine** (the explicit
  ``SPINE_MODULES`` below) or **component** (everything else, stable or experimental),
  plus the closed dispatchers (split algorithms, ``METRICS``);
- each component's docstring sections (Consumes, Produces, Parameters, Devices,
  Limitations, Example);
- ``docs/component-warehouse/evidence.yaml``, which records who uses each component and
  whether it is legacy. It lives outside ``src/dsio`` because the admission audit rejects
  consumer names in DSio source;
- ``docs/component-warehouse/candidates.yaml``: consumer-local code awaiting a block.

A new stable module is a component module unless it is deliberately added to
``SPINE_MODULES``, so a promoted block can never skip the evidence rules unnoticed.

``python tools/catalog.py`` rewrites ``docs/component-warehouse/catalog.md``;
``python tools/catalog.py --check`` fails on any rule violation or on a stale catalog.
This is repository tooling, not part of the wheel.
"""

from __future__ import annotations

import argparse
import ast
import difflib
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
KAGGLE = "reference_projects/kaggle/"

# Stable modules whose public names are the spine: composition roots, data infrastructure,
# export and evidence machinery. Catalogued for discovery; not warehouse components. Any
# other stable module is a component module and must satisfy the stable evidence rules.
SPINE_MODULES = frozenset(
    {
        "dsio.batches",
        "dsio.config",
        "dsio.config.components",
        "dsio.config.registry",
        "dsio.contracts",
        "dsio.contracts.base",
        "dsio.contracts.hashing",
        "dsio.contracts.io",
        "dsio.data",
        "dsio.data.adapters",
        "dsio.data.examples",
        "dsio.data.format",
        "dsio.data.loading",
        "dsio.data.loading.collation",
        "dsio.data.loading.datasets",
        "dsio.data.loading.loaders",
        "dsio.data.loading.module",
        "dsio.data.readers",
        "dsio.data.splits",
        "dsio.data.splits.folds",
        "dsio.data.splits.generate",
        "dsio.data.splits.models",
        "dsio.data.splits.models.fold",
        "dsio.data.splits.models.manifest",
        "dsio.data.splits.resolve",
        "dsio.data.splits.temporal",
        "dsio.data.splits.validation",
        "dsio.data.staging",
        "dsio.data.store",
        "dsio.data.store.builder",
        "dsio.data.store.layout",
        "dsio.data.store.reader",
        "dsio.data.views",
        "dsio.eval",
        "dsio.eval.execution",
        "dsio.eval.metrics",
        "dsio.inference",
        "dsio.inference.export",
        "dsio.inference.lineage",
        "dsio.inference.loading",
        "dsio.inference.predictor",
        "dsio.model.module",
        "dsio.testing",
        "dsio.testing.examples_contract",
        "dsio.testing.reader_contract",
        "dsio.tracking",
        "dsio.tracking._lifecycle",
        "dsio.tracking.attempt",
        "dsio.tracking.cache",
        "dsio.tracking.client",
        "dsio.tracking.evidence",
        "dsio.tracking.evidence.references",
        "dsio.tracking.evidence.resolution",
        "dsio.tracking.evidence.splits",
        "dsio.tracking.evidence.validation",
        "dsio.tracking.execution",
        "dsio.tracking.execution.capture",
        "dsio.tracking.execution.environment",
        "dsio.tracking.execution.git",
        "dsio.tracking.experiment",
        "dsio.tracking.provenance",
        "dsio.train.artifacts",
        "dsio.train.capabilities",
        "dsio.train.trainer",
    }
)
# Pre-1.0 legacy experimental components (docs/component-admission.md), frozen: the list
# may only shrink. A new component cannot be declared legacy to skip the rules.
LEGACY = frozenset(
    {
        "dsio.experimental.data.labels:entity_attribute_labels",
        "dsio.experimental.data.samples:StoredSamples",
        "dsio.experimental.data.samples:stored_samples",
        "dsio.experimental.data.windows:WindowDataset",
        "dsio.experimental.eval.ess:autocorrelation",
        "dsio.experimental.eval.ess:effective_sample_size",
        "dsio.experimental.model.chain:ComponentChain",
        "dsio.experimental.model.chain:LossObjective",
        "dsio.experimental.model.chain:export_encoder",
        "dsio.experimental.model.components:Conv1dEncoder",
        "dsio.experimental.model.components:CrossEntropy",
        "dsio.experimental.model.components:EmbeddingEncoder",
        "dsio.experimental.model.components:FixedStandardize",
        "dsio.experimental.model.components:IdentityAugmentation",
        "dsio.experimental.model.components:InstanceStandardize",
        "dsio.experimental.model.components:Jitter",
        "dsio.experimental.model.components:MLP1d",
        "dsio.experimental.model.components:MaskedMSE",
        "dsio.experimental.model.components:NTXent",
        "dsio.experimental.model.components:RandomScale",
        "dsio.experimental.model.components:VICReg",
        "dsio.experimental.model.components:identity_head",
        "dsio.experimental.model.components:identity_transform",
        "dsio.experimental.model.components:linear_head",
        "dsio.experimental.model.components:mae_decoder_head",
        "dsio.experimental.model.components:mlp_head",
        "dsio.experimental.model.components:no_augmentation",
        "dsio.experimental.model.components:simclr_projector_head",
        "dsio.experimental.model.components:vicreg_projector_head",
        "dsio.experimental.model.masking:CausalMask",
        "dsio.experimental.model.masking:PatchMask",
        "dsio.experimental.model.masking:RandomMask",
        "dsio.experimental.model.masking:SpanMask",
        "dsio.experimental.model.masking:apply_mask",
        "dsio.experimental.train.augmentation:MaskedReconstruction",
        "dsio.experimental.train.augmentation:TwoView",
        "dsio.experimental.train.callbacks:Encodable",
        "dsio.experimental.train.callbacks:OnlineProbe",
        "dsio.experimental.train.callbacks:RankMeMonitor",
        "dsio.experimental.train.callbacks:embed",
        "dsio.experimental.train.callbacks:rankme",
    }
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
    problems: tuple[str, ...] = ()

    @property
    def module(self) -> str:
        return self.reference.partition(":")[0]

    @property
    def qualname(self) -> str:
        return self.reference.partition(":")[2]

    @property
    def experimental(self) -> bool:
        return self.module.startswith("dsio.experimental.")

    @property
    def package(self) -> str:
        parts = self.module.split(".")
        return ".".join(parts[:3] if self.experimental else parts[:2])


def collect() -> list[Entry]:
    """Every public object of every dsio module, plus closed-dispatcher entries."""
    import dsio

    entries: dict[str, Entry] = {}
    for info in pkgutil.walk_packages(dsio.__path__, prefix="dsio."):
        if info.name.startswith("dsio.experimental.admission"):
            continue  # the auditor, not a component
        module = importlib.import_module(info.name)
        kind = "spine" if info.name in SPINE_MODULES else "component"
        problems = _undeclared_public_names(module, component_module=kind == "component")
        for reference, value in _public(module):
            if reference.startswith("dsio.experimental.admission"):
                continue  # re-exported auditor
            doc = inspect.getdoc(value) or ""
            sections, section_problems = parse_sections(doc)
            entries.setdefault(
                reference,
                Entry(reference, kind, _summary(doc), sections, tuple(section_problems)),
            )
        if problems:
            key = f"{info.name}:<module>"
            entries[key] = Entry(key, "problem", "", problems=tuple(problems))
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


def parse_sections(doc: str) -> tuple[dict[str, str], list[str]]:
    """Return ``({section: body}, problems)`` for the catalog headers in a docstring."""
    sections: dict[str, list[str]] = {}
    problems: list[str] = []
    current: str | None = None
    for line in doc.splitlines():
        match = _SECTION.match(line) if not line.startswith(" ") else None
        if match and match.group("name") in SECTIONS:
            current = match.group("name")
            if current in sections:
                problems.append(f"section {current!r} appears more than once")
            sections[current] = []
        elif match or (line and not line.startswith(" ")):
            current = None  # another header or a dedented line ends the section
        elif current is not None:
            sections[current].append(line[4:] if line.startswith("    ") else line.strip())
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}, problems


def load_yaml(path: Path) -> dict[str, Any]:
    data = (
        yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        if path.exists()
        else None
    )
    return dict(data or {})


def use_group(consumer: str) -> str | None:
    """The unrelated-use group of a real consumer path: its Kaggle project directory."""
    if not consumer.startswith(KAGGLE):
        return None
    return consumer[len(KAGGLE) :].split("/", 1)[0] or None


def validate(
    entries: Sequence[Entry],
    evidence: Mapping[str, Any],
    candidates: Mapping[str, Any],
    *,
    root: Path = ROOT,
    legacy: frozenset[str] = LEGACY,
) -> list[str]:
    """Every rule the catalog enforces, as human-readable problems."""
    problems = [
        problem for entry in entries if entry.kind == "problem" for problem in entry.problems
    ]
    known = {entry.reference for entry in entries if entry.kind != "problem"}
    for reference in sorted(set(evidence) - known):
        problems.append(f"{reference}: evidence entry names no public component")
    for entry in entries:
        if entry.kind != "component":
            continue
        problems.extend(f"{entry.reference}: {problem}" for problem in entry.problems)
        record = evidence.get(entry.reference)
        if not isinstance(record, Mapping):
            problems.append(f"{entry.reference}: missing evidence entry")
            continue
        use_problems, groups = _check_uses(entry, record.get("uses") or [], root)
        problems.extend(use_problems)
        if record.get("legacy"):
            if entry.reference not in legacy:
                problems.append(f"{entry.reference}: only frozen pre-1.0 components may be legacy")
            if groups:
                problems.append(
                    f"{entry.reference}: a legacy component with a real use must drop legacy "
                    "and gain its docstring sections"
                )
            continue
        missing = [name for name in SECTIONS if not entry.sections.get(name)]
        if missing:
            problems.append(f"{entry.reference}: missing docstring sections {missing}")
        if entry.experimental and not groups:
            problems.append(f"{entry.reference}: experimental needs one real downstream use")
        if not entry.experimental:
            if len(groups) < 2:
                problems.append(
                    f"{entry.reference}: a stable component needs two unrelated real uses, "
                    f"has {len(groups)}"
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


def real_groups(record: Mapping[str, Any]) -> set[str]:
    """Distinct unrelated-use groups among a component's real uses."""
    groups = set()
    for use in record.get("uses") or []:
        if isinstance(use, Mapping) and use.get("kind") == "real":
            group = use_group(str(use.get("consumer", "")))
            if group:
                groups.add(group)
    return groups


def _check_uses(entry: Entry, uses: Sequence[Any], root: Path) -> tuple[list[str], set[str]]:
    """Problems with each recorded use, and the groups of the real uses that are valid."""
    problems: list[str] = []
    groups: set[str] = set()
    for index, use in enumerate(uses):
        where = f"{entry.reference}: use {index}"
        if not isinstance(use, Mapping):
            problems.append(f"{where}: must be a mapping")
            continue
        before = len(problems)
        kind = use.get("kind")
        if kind not in USE_KINDS:
            problems.append(f"{where}: kind must be one of {USE_KINDS}")
        consumer = use.get("consumer")
        for key in ("consumer", "test"):
            path = use.get(key)
            if not isinstance(path, str) or not (root / path).exists():
                problems.append(f"{where}: {key} path {path!r} does not exist")
        for uri in use.get("runs") or []:
            if not isinstance(uri, str) or not _RUN_URI.match(uri):
                problems.append(f"{where}: run {uri!r} is not an immutable MLflow run URI")
        if not isinstance(consumer, str) or not (root / consumer).exists():
            continue
        if kind == "real" and use_group(consumer) is None:
            problems.append(f"{where}: a real use must be a Kaggle consumer under {KAGGLE}")
        via = use.get("via")
        name = str(via).partition(":")[2] if via else entry.qualname
        if not _references(root / consumer, name):
            problems.append(f"{where}: {consumer} never references {name!r}")
        group = use_group(consumer)
        if kind == "real" and group and len(problems) == before:
            groups.add(group)
    return problems, groups


def _references(consumer: Path, name: str) -> bool:
    paths = [consumer] if consumer.is_file() else sorted(consumer.rglob("*.py"))
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    return any(pattern.search(path.read_text(encoding="utf-8")) for path in paths)


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

    def is_legacy(entry: Entry) -> bool:
        record = evidence.get(entry.reference)
        return isinstance(record, Mapping) and bool(record.get("legacy"))

    proven = [entry for entry in components if not is_legacy(entry)]
    legacy = [entry for entry in components if is_legacy(entry)]
    lines += ["## Components", ""]
    if not proven:
        lines += ["_No component has a recorded real use yet._", ""]
    for package in sorted({entry.package for entry in proven}):
        lines += [f"### `{package}`", ""]
        for entry in proven:
            if entry.package == package:
                lines += _render_component(entry, evidence.get(entry.reference) or {})
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
        uses = (evidence.get(entry.reference) or {}).get("uses") or []
        fixtures = ", ".join(sorted({Path(str(use.get("consumer"))).name for use in uses})) or "—"
        lines.append(f"| `{entry.reference}` | {_cell(entry.summary)} | {fixtures} |")
    lines += [
        "",
        "## Spine",
        "",
        "Composition roots, data infrastructure and evidence machinery every block plugs into.",
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
        "| Candidate | Uses | Consumers | Reason |",
        "|---|---|---|---|",
    ]
    for name, candidate in sorted(candidates.items()):
        consumers = candidate.get("consumers") or [] if isinstance(candidate, Mapping) else []
        listed = ", ".join(f"`{path}`" for path in consumers)
        reason = candidate.get("reason", "") if isinstance(candidate, Mapping) else ""
        lines.append(f"| {name} | {len(consumers)} | {listed} | {_cell(str(reason))} |")
    return "\n".join(lines) + "\n"


def _render_component(entry: Entry, record: Mapping[str, Any]) -> list[str]:
    uses = [use for use in record.get("uses") or [] if isinstance(use, Mapping)]
    groups = sorted(real_groups(record))
    maturity = "experimental" if entry.experimental else "stable"
    lines = [
        f"#### `{entry.reference}`",
        "",
        f"{entry.summary.replace('``', '`')} — **{maturity}**; real uses: {len(groups)} "
        f"({', '.join(groups) or 'none'}).",
        "",
    ]
    for name in SECTIONS:
        body = entry.sections.get(name, "")
        if name == "Example":
            lines += [f"**{name}**", "", "```python", body, "```", ""]
        else:
            lines += [f"**{name}**: {body.replace('``', '`')}", ""]
    lines += ["**Evidence**:", ""]
    for use in uses:
        runs = ", ".join(use.get("runs") or []) or "—"
        via = f"; via `{use['via']}`" if use.get("via") else ""
        lines.append(
            f"- {use.get('kind')}: `{use.get('consumer')}` "
            f"(test `{use.get('test')}`{via}; runs {runs})"
        )
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


def _undeclared_public_names(module: Any, *, component_module: bool = False) -> list[str]:
    """Public top-level classes/functions defined in a module but missing from __all__."""
    if not getattr(module, "__file__", None) or module.__name__.endswith("__main__"):
        return []
    declared = set(getattr(module, "__all__", ()))
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    defined = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and not node.name.startswith("_")
    ]
    if component_module and not hasattr(module, "__all__"):
        return [f"{module.__name__}: component module must define __all__"]
    if (
        not component_module
        and not module.__name__.startswith("dsio.experimental.")
        and not declared
    ):
        return []
    return [
        f"{module.__name__}: public name {name!r} is not in __all__"
        for name in defined
        if name not in declared
        and not (
            inspect.isclass(getattr(module, name, None))
            and issubclass(getattr(module, name), BaseException)
        )
    ]


def _summary(doc: str) -> str:
    """The docstring's first paragraph, joined onto one line."""
    paragraph = doc.strip().split("\n\n", 1)[0] if doc.strip() else ""
    return " ".join(line.strip() for line in paragraph.splitlines())


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses governance entries hidden by duplicate keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"YAML key {key!r} appears more than once")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail on problems or drift")
    args = parser.parse_args(argv)
    entries = collect()
    evidence = load_yaml(EVIDENCE)
    candidates = load_yaml(CANDIDATES)
    problems = validate(entries, evidence, candidates)
    rendered = render(entries, evidence, candidates)
    if args.check:
        current = CATALOG.read_text(encoding="utf-8") if CATALOG.exists() else ""
        if current != rendered:
            diff = difflib.unified_diff(
                current.splitlines(), rendered.splitlines(), "committed", "generated", lineterm=""
            )
            problems.append(
                f"{CATALOG.relative_to(ROOT)} is stale; run tools/catalog.py\n"
                + "\n".join(list(diff)[:40])
            )
    else:
        CATALOG.write_text(rendered, encoding="utf-8")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
