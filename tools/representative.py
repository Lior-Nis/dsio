"""Run reference consumers on their real data and record or compare parity evidence.

The representative tier is each Kaggle consumer's own flow on its real competition data,
logged to the live MLflow server. Story 6.6 records the pre-migration baseline; the parity
stories (7.7, 8.7, 9.6, 10.6) rerun the same consumers and compare against it:

    uv run python tools/representative.py titanic essay_scoring --record
    uv run python tools/representative.py titanic --compare --tolerance 0

Each run uses an isolated ``PREFECT_HOME`` and a fresh workspace under ``--runs-root``.
Results (metrics, MLflow run IDs and URIs, commit, hardware, duration) land in
``docs/component-warehouse/parity-baseline.json`` and its rendered ``.md``. Repository
tooling, not part of the wheel.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs" / "component-warehouse" / "parity-baseline.json"
REPORT = BASELINE.with_suffix(".md")
DATASETS = Path.home() / "Datasets"
DEFAULT_RUNS = DATASETS / "dsio-runs" / "representative"


@dataclass(frozen=True)
class Consumer:
    flow: str  # "module:function"
    data: Path
    options: Mapping[str, Any]


CONSUMERS: dict[str, Consumer] = {
    "titanic": Consumer(
        "reference_projects.kaggle.titanic.flow:titanic_flow",
        DATASETS / "dsio-kaggle-portfolio" / "titanic",
        {},
    ),
    "bike_sharing": Consumer(
        "reference_projects.kaggle.bike_sharing.flow:bike_sharing_flow",
        DATASETS / "dsio-kaggle-portfolio" / "bike-sharing",
        {},
    ),
    "digit_recognizer": Consumer(
        "reference_projects.kaggle.digit_recognizer.flow:digit_recognizer_flow",
        DATASETS / "dsio-kaggle-portfolio" / "digit-recognizer",
        {},
    ),
    "essay_scoring": Consumer(
        "reference_projects.kaggle.essay_scoring.flow:essay_scoring_flow",
        DATASETS / "learning-agency-lab-automated-essay-scoring-2",
        {},
    ),
    "rogii": Consumer(
        "reference_projects.kaggle.rogii.flow:rogii_flow",
        DATASETS / "rogii-wellbore-geology-prediction",
        {},
    ),
    "store_sales": Consumer(
        "reference_projects.kaggle.store_sales.flow:store_sales_flow",
        DATASETS / "store-sales-time-series-forecasting",
        {},
    ),
    "child_mind": Consumer(
        "reference_projects.kaggle.child_mind.flow:child_mind_flow",
        DATASETS / "child-mind-institute-problematic-internet-use",
        {},
    ),
    "parkinsons_fog": Consumer(
        "reference_projects.kaggle.parkinsons_fog.flow:parkinsons_fog_flow",
        DATASETS / "tlvmc-parkinsons-freezing-gait-prediction",
        {},
    ),
    "child_mind_sequence": Consumer(
        "reference_projects.kaggle.child_mind.sequence.flow:child_mind_sequence_flow",
        DATASETS / "child-mind-institute-problematic-internet-use",
        {"accelerator": "cuda"},
    ),
}


def run(name: str, runs_root: Path) -> dict[str, Any]:
    """Execute one consumer's representative flow and return its parity record."""
    consumer = CONSUMERS[name]
    if not consumer.data.exists():
        raise FileNotFoundError(f"{name}: data directory {consumer.data} does not exist")
    module, _, function = consumer.flow.partition(":")
    flow: Callable[..., Mapping[str, Any]] = getattr(importlib.import_module(module), function)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    workspace = runs_root / f"{name}-{stamp}"
    workspace.mkdir(parents=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dsio-prefect-") as prefect_home:
        os.environ["PREFECT_HOME"] = prefect_home
        result = flow(str(consumer.data), str(workspace), seed=19, **consumer.options)
    seconds = time.perf_counter() - started
    return {
        "metrics": _metrics(result),
        "runs": _runs(result),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "hardware": _hardware(),
        "tracking_uri": _tracking_uri(),
        "seconds": round(seconds, 1),
        "recorded": stamp,
    }


def compare(
    baseline: Mapping[str, float], observed: Mapping[str, float], tolerance: float
) -> dict[str, tuple[float | None, float | None]]:
    """Metrics that moved beyond an absolute ``tolerance`` (0 means exact)."""
    moved: dict[str, tuple[float | None, float | None]] = {}
    for name in sorted(set(baseline) | set(observed)):
        before, after = baseline.get(name), observed.get(name)
        if (
            before is None
            or after is None
            or not math.isclose(before, after, rel_tol=0.0, abs_tol=tolerance)
        ):
            moved[name] = (before, after)
    return moved


def _metrics(result: Mapping[str, Any]) -> dict[str, float]:
    if "metrics" in result:
        return {name: float(value) for name, value in sorted(result["metrics"].items())}
    return {  # multi-model consumers (child_mind) report per mode
        f"{mode}.{name}": float(value)
        for mode, model in sorted(result["models"].items())
        for name, value in sorted(model["metrics"].items())
    }


def _runs(result: Mapping[str, Any]) -> dict[str, str]:
    runs = {key: value for key, value in result.items() if key.endswith("_run_id")}
    for mode, model in (result.get("models") or {}).items():
        runs.update(
            {f"{mode}.{key}": value for key, value in model.items() if key.endswith("_run_id")}
        )
    return dict(sorted(runs.items()))


def _tracking_uri() -> str:
    import mlflow

    return str(mlflow.get_tracking_uri())


def _hardware() -> dict[str, Any]:
    import torch

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def render(baseline: Mapping[str, Any]) -> str:
    lines = [
        "# Parity baseline (Component Warehouse v1, Story 6.6)",
        "",
        "<!-- Generated by `uv run python tools/representative.py --record`. -->",
        "",
        "Contract-tier goldens live in `tests/golden_metrics.json` and are asserted by every",
        "replay test. This file holds the representative tier: each Kaggle consumer's own flow",
        "on its real data, logged to the live MLflow server. Parity stories rerun a consumer",
        "with `--compare`; runs must match exactly unless the story pre-declared a tolerance.",
        "",
        "| Consumer | Commit | Hardware | Seconds | Metrics | Evaluation run |",
        "|---|---|---|---:|---|---|",
    ]
    for name, record in sorted(baseline.items()):
        hardware = record["hardware"]
        device = hardware["cuda"] or f"CPU x{hardware['threads']}"
        metrics = ", ".join(f"{key}={value:.6g}" for key, value in record["metrics"].items())
        evaluation = [value for key, value in record["runs"].items() if "evaluation" in key]
        dirty = " (dirty)" if record["dirty"] else ""
        lines.append(
            f"| `{name}` | `{record['commit'][:12]}`{dirty} | {device} | {record['seconds']} | "
            f"{metrics} | {', '.join(f'`{run}`' for run in evaluation)} |"
        )
    missing = sorted(set(CONSUMERS) - set(baseline))
    if missing:
        lines += [
            "",
            "Contract tier only (no representative data under `~/Datasets`): "
            + ", ".join(f"`{name}`" for name in missing)
            + ".",
        ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("consumers", nargs="+", choices=sorted(CONSUMERS))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true", help="record as the parity baseline")
    mode.add_argument("--compare", action="store_true", help="compare with the baseline")
    parser.add_argument("--tolerance", type=float, default=0.0, help="absolute, per metric")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))  # consumers import as reference_projects.*
    # Evidence belongs on the live MLflow server, never in a local file store.
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    failed = False
    for name in args.consumers:
        record = run(name, args.runs_root)
        print(f"{name}: {json.dumps(record['metrics'], sort_keys=True)} in {record['seconds']}s")
        if args.record:
            baseline[name] = record
            BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
            REPORT.write_text(render(baseline))
        else:
            if name not in baseline:
                print(f"{name}: no recorded baseline", file=sys.stderr)
                failed = True
                continue
            moved = compare(baseline[name]["metrics"], record["metrics"], args.tolerance)
            print(f"{name}: parity {'FAILED ' + str(moved) if moved else 'holds'}")
            failed = failed or bool(moved)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
