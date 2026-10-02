"""Measure how much code each reference consumer still writes (Component Warehouse v1).

Two measures back the warehouse's success signal:

- **Lines per category.** *Ingestion* is ``data.py``, ``tasks/data.py`` and ingestion-only
  packages such as ``parkinsons_fog/scale/``. Everything else is *non-ingestion*: model-side
  ``components.py``, task *wiring* (``tasks/training.py``, ``tasks/export.py``,
  ``tasks/downstream.py``, ``tasks/evaluation.py``) and *flow* glue. Non-ingestion lines
  are what the warehouse is meant to delete.
- **Banned local definitions.** A syntax-tree scan finds consumer-defined ``nn.Module`` or
  ``Dataset`` subclasses, and consumer-defined functions passed as an objective, collate
  function, normalizer, validator or dataset factory. At release, every finding must be a
  catalogued admission candidate (``docs/component-warehouse/candidates.yaml``).

``python tools/consumer_metrics.py`` prints both; ``--json`` emits them for tooling.
Repository tooling, not part of the wheel.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference_projects"
INGESTION_FILES = ("data.py", "tasks/data.py")
INGESTION_PACKAGES = ("scale",)
WIRING_FILES = (
    "tasks/training.py",
    "tasks/export.py",
    "tasks/downstream.py",
    "tasks/evaluation.py",
)
BANNED_BASES = frozenset({"Module", "Dataset", "IterableDataset"})
BANNED_KEYWORDS = frozenset(
    {"objective", "collate_fn", "normalizer", "validator", "dataset_factory"}
)


@dataclass(frozen=True)
class Lines:
    consumer: str
    kaggle: bool
    ingestion: int
    model_side: int
    wiring: int
    flow: int

    @property
    def non_ingestion(self) -> int:
        return self.model_side + self.wiring + self.flow


@dataclass(frozen=True)
class Finding:
    consumer: str
    path: str
    line: int
    name: str
    kind: str  # "nn.Module/Dataset subclass" or "passed as <keyword>"


def consumers(root: Path = REFERENCE) -> list[Path]:
    """Every consumer directory: Kaggle projects (and nested ones) plus the fixtures."""
    found = []
    for flow in sorted(root.rglob("flow.py")):
        if "__pycache__" not in flow.parts:
            found.append(flow.parent)
    return found


def _files(consumer: Path, nested: Iterable[Path]) -> list[Path]:
    skip = [path for path in nested if path != consumer and path.is_relative_to(consumer)]
    return [
        path
        for path in sorted(consumer.rglob("*.py"))
        if "__pycache__" not in path.parts and not any(path.is_relative_to(s) for s in skip)
    ]


def _category(relative: str) -> str:
    if relative in INGESTION_FILES or relative.split("/", 1)[0] in INGESTION_PACKAGES:
        return "ingestion"
    if relative == "components.py":
        return "model_side"
    if relative in WIRING_FILES:
        return "wiring"
    return "flow"


def count_lines(root: Path = REFERENCE) -> list[Lines]:
    """Line counts per consumer and category (``wc -l`` semantics)."""
    all_consumers = consumers(root)
    counts = []
    for consumer in all_consumers:
        totals = {"ingestion": 0, "model_side": 0, "wiring": 0, "flow": 0}
        for path in _files(consumer, all_consumers):
            relative = path.relative_to(consumer).as_posix()
            totals[_category(relative)] += len(path.read_bytes().splitlines())
        counts.append(
            Lines(
                consumer.relative_to(root.parent).as_posix(),
                "kaggle" in consumer.relative_to(root).parts,
                **totals,
            )
        )
    return counts


def banned_definitions(root: Path = REFERENCE) -> list[Finding]:
    """Consumer-defined models, datasets and injected callables the warehouse replaces."""
    all_consumers = consumers(root)
    findings = []
    for consumer in all_consumers:
        name = consumer.relative_to(root.parent).as_posix()
        files = _files(consumer, all_consumers)
        trees = {path: ast.parse(path.read_text(encoding="utf-8")) for path in files}
        local = {
            node.name: (path, node)
            for path, tree in trees.items()
            for node in tree.body
            if isinstance(node, ast.ClassDef | ast.FunctionDef)
        }
        for path, tree in trees.items():
            relative = path.relative_to(root.parent).as_posix()
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and any(
                    _base_name(base) in BANNED_BASES for base in node.bases
                ):
                    findings.append(
                        Finding(
                            name, relative, node.lineno, node.name, "nn.Module/Dataset subclass"
                        )
                    )
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                for keyword in call.keywords:
                    target = _passed_name(keyword.value)
                    if keyword.arg in BANNED_KEYWORDS and target in local:
                        defining_path, definition = local[target]
                        if isinstance(definition, ast.FunctionDef):
                            findings.append(
                                Finding(
                                    name,
                                    defining_path.relative_to(root.parent).as_posix(),
                                    definition.lineno,
                                    target,
                                    f"passed as {keyword.arg}",
                                )
                            )
    unique = {(f.consumer, f.path, f.name): f for f in findings}
    return sorted(unique.values(), key=lambda f: (f.consumer, f.path, f.line))


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return node.id if isinstance(node, ast.Name) else ""


def _passed_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id  # e.g. normalizer=CmiPrediction()
    return None


def report(lines: Sequence[Lines], findings: Sequence[Finding]) -> str:
    """A Markdown summary of both measures."""
    out = [
        "| Consumer | Ingestion | Model-side | Wiring | Flow | Non-ingestion |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in lines:
        out.append(
            f"| `{row.consumer}` | {row.ingestion} | {row.model_side} | {row.wiring} | "
            f"{row.flow} | {row.non_ingestion} |"
        )
    kaggle = [row for row in lines if row.kaggle]
    out.append(
        f"| **Kaggle total** | {sum(r.ingestion for r in kaggle)} | "
        f"{sum(r.model_side for r in kaggle)} | {sum(r.wiring for r in kaggle)} | "
        f"{sum(r.flow for r in kaggle)} | {sum(r.non_ingestion for r in kaggle)} |"
    )
    out += ["", f"Banned local definitions: {len(findings)}", ""]
    out += ["| Consumer | Definition | Kind | Location |", "|---|---|---|---|"]
    for finding in findings:
        out.append(
            f"| `{finding.consumer}` | `{finding.name}` | {finding.kind} | "
            f"`{finding.path}:{finding.line}` |"
        )
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    lines = count_lines()
    findings = banned_definitions()
    if args.json:
        payload = {
            "lines": [{**asdict(row), "non_ingestion": row.non_ingestion} for row in lines],
            "banned_definitions": [asdict(finding) for finding in findings],
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(report(lines, findings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
