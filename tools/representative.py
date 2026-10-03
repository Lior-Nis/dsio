"""Run reference consumers on their real data and record or compare parity evidence.

The representative tier is each Kaggle consumer's own flow on its real competition data,
logged to the live MLflow server. Story 6.6 records the pre-migration baseline; the parity
stories (7.7, 8.7, 9.6, 10.6) rerun the same consumers and compare against it:

    uv run python tools/representative.py titanic essay_scoring --record
    uv run python tools/representative.py titanic --compare --tolerance 0

Each run starts a fresh process with an isolated ``PREFECT_HOME`` and a fresh workspace
under ``--runs-root``. Results (metrics, immutable MLflow run URIs, commit, hardware,
duration) land in
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
from urllib.parse import urlsplit

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
    tracking_uri = _require_live_tracking()
    ui_uri = _ui_uri(tracking_uri)
    consumer = CONSUMERS[name]
    if not consumer.data.exists():
        raise FileNotFoundError(f"{name}: data directory {consumer.data} does not exist")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    workspace = runs_root / f"{name}-{stamp}"
    workspace.mkdir(parents=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dsio-prefect-") as prefect_home:
        result_path = Path(prefect_home) / "result.json"
        environment = {
            name: value for name, value in os.environ.items() if not name.startswith("PREFECT_")
        }
        environment["PREFECT_HOME"] = prefect_home
        environment["MLFLOW_TRACKING_URI"] = tracking_uri
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "_worker",
                "--flow",
                consumer.flow,
                "--data",
                str(consumer.data),
                "--workspace",
                str(workspace),
                "--options",
                json.dumps(consumer.options),
                "--result",
                str(result_path),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    seconds = time.perf_counter() - started
    result = payload["result"]
    from mlflow.tracking import MlflowClient

    return {
        "metrics": _metrics(result),
        "runs": _runs(result, ui_uri, MlflowClient(tracking_uri)),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "hardware": payload["hardware"],
        "tracking_uri": tracking_uri,
        "ui_uri": ui_uri,
        "seconds": round(seconds, 1),
        "recorded": stamp,
    }


def compare(
    baseline: Mapping[str, float], observed: Mapping[str, float], tolerance: float
) -> dict[str, tuple[float | None, float | None]]:
    """Metrics that moved beyond an absolute ``tolerance`` (0 means exact)."""
    _require_tolerance(tolerance)
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


def record_parity(
    name: str,
    baseline: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    tolerance: float,
    ui_uri: str,
    client: Any | None = None,
) -> dict[str, Any]:
    """Log one immutable comparison run beside the observed evaluation run."""
    ui_uri = _validated_http_uri(ui_uri, "MLflow UI")
    _require_clean_sources(baseline, observed)
    _require_metrics(baseline["metrics"], observed["metrics"])
    baseline_runs = _evaluation_run_ids(baseline)
    observed_runs = _evaluation_run_ids(observed)
    if set(baseline_runs) != set(observed_runs):
        raise ValueError(
            "baseline and observed evaluation runs differ: "
            f"{sorted(baseline_runs)} != {sorted(observed_runs)}"
        )
    if client is None:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(observed["tracking_uri"])

    baseline_metrics = baseline["metrics"]
    observed_metrics = observed["metrics"]
    moved = compare(baseline_metrics, observed_metrics, tolerance)
    missing = sorted(set(baseline_metrics) ^ set(observed_metrics))
    baseline_run_id = _tag_run_ids(baseline_runs)
    observed_run_id = _tag_run_ids(observed_runs)
    for run_id in baseline_runs.values():
        client.get_run(run_id)
    observed_experiments = {
        client.get_run(run_id).info.experiment_id for run_id in observed_runs.values()
    }
    if len(observed_experiments) != 1:
        raise ValueError("observed evaluation runs must belong to one MLflow experiment")
    experiment_id = next(iter(observed_experiments))
    status = "failed" if moved else "passed"
    tags = {
        "mlflow.runName": f"parity-{name}",
        "dsio.parity.consumer": name,
        "dsio.parity.baseline_commit": str(baseline["commit"]),
        "dsio.parity.baseline_run_id": baseline_run_id,
        "dsio.parity.observed_commit": str(observed["commit"]),
        "dsio.parity.observed_run_id": observed_run_id,
        "dsio.parity.hardware": json.dumps(_hardware_class(observed["hardware"]), sort_keys=True),
        "dsio.parity.status": "recording",
        "dsio.parity.tolerance": str(float(tolerance)),
    }
    if missing:
        tags["dsio.parity.missing_metrics"] = json.dumps(missing)
    parity_run = client.create_run(experiment_id, tags=tags)
    parity_run_id = parity_run.info.run_id
    try:
        for metric in sorted(set(baseline_metrics) | set(observed_metrics)):
            before = baseline_metrics.get(metric)
            after = observed_metrics.get(metric)
            if before is not None:
                client.log_metric(parity_run_id, f"baseline.{metric}", float(before))
            if after is not None:
                client.log_metric(parity_run_id, f"observed.{metric}", float(after))
            if before is not None and after is not None:
                client.log_metric(parity_run_id, f"delta.{metric}", float(after - before))
    except Exception:
        client.set_tag(parity_run_id, "dsio.parity.status", "evidence_error")
        client.set_terminated(parity_run_id, status="FAILED")
        raise
    client.set_tag(parity_run_id, "dsio.parity.status", status)
    client.set_terminated(parity_run_id, status="FAILED" if moved else "FINISHED")
    return {
        "moved": moved,
        "run_id": parity_run_id,
        "run_uri": (f"{ui_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{parity_run_id}"),
    }


def _metrics(result: Mapping[str, Any]) -> dict[str, float]:
    if "metrics" in result:
        return {name: float(value) for name, value in sorted(result["metrics"].items())}
    return {  # multi-model consumers (child_mind) report per mode
        f"{mode}.{name}": float(value)
        for mode, model in sorted(result["models"].items())
        for name, value in sorted(model["metrics"].items())
    }


def _run_ids(result: Mapping[str, Any]) -> dict[str, str]:
    runs = {key: value for key, value in result.items() if key.endswith("_run_id")}
    for mode, model in (result.get("models") or {}).items():
        runs.update(
            {f"{mode}.{key}": value for key, value in model.items() if key.endswith("_run_id")}
        )
    return dict(sorted(runs.items()))


def _evaluation_run_ids(record: Mapping[str, Any]) -> dict[str, str]:
    runs = record["runs"]
    return {
        name: str(run_id)
        for name, run_id in sorted(runs.items())
        if "evaluation" in name and name.endswith("_run_id")
    }


def _tag_run_ids(runs: Mapping[str, str]) -> str:
    if not runs:
        raise ValueError("representative record has no evaluation run ID")
    if len(runs) == 1:
        return next(iter(runs.values()))
    return json.dumps(runs, sort_keys=True)


def _require_baseline_runs(records: Sequence[Mapping[str, Any]], client: Any | None = None) -> None:
    """Resolve clean baseline evaluations before starting an expensive consumer."""
    if client is None:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(_require_live_tracking())
    for record in records:
        for run_id in _evaluation_run_ids(record).values():
            client.get_run(run_id)
            with tempfile.TemporaryDirectory(prefix="dsio-baseline-provenance-") as directory:
                artifact = client.download_artifacts(run_id, "provenance.json", directory)
                provenance = json.loads(Path(artifact).read_text(encoding="utf-8"))
            git = provenance.get("execution", {}).get("git", {})
            if git.get("dirty") is not False:
                raise ValueError(f"baseline evaluation run {run_id} has dirty provenance")
            expected_commit = record.get("commit")
            if expected_commit and git.get("sha") != expected_commit:
                raise ValueError(
                    f"baseline evaluation run {run_id} provenance commit "
                    f"{git.get('sha')!r} != {expected_commit!r}"
                )


def _runs(
    result: Mapping[str, Any], tracking_uri: str, client: Any | None = None
) -> dict[str, str]:
    """Keep result Run IDs and add immutable links on their tracking server."""
    if client is None:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri)
    base = tracking_uri.rstrip("/")
    runs: dict[str, str] = {}
    for name, run_id in _run_ids(result).items():
        runs[name] = run_id
        uri_name = f"{name.removesuffix('_run_id')}_run_uri"
        runs[uri_name] = (
            f"{base}/#/experiments/{client.get_run(run_id).info.experiment_id}/runs/{run_id}"
        )
    return runs


def _tracking_uri() -> str:
    import mlflow

    return str(mlflow.get_tracking_uri())


def _require_live_tracking(tracking_uri: str | None = None) -> str:
    """Return a reachable HTTP(S) MLflow URI, rejecting local or offline stores."""
    uri = _tracking_uri() if tracking_uri is None else tracking_uri
    uri = _validated_http_uri(uri, "MLflow tracking")
    from mlflow.tracking import MlflowClient

    try:
        MlflowClient(uri).search_experiments(max_results=1)
    except Exception as error:
        raise RuntimeError(f"MLflow tracking server {uri!r} is not reachable: {error}") from error
    return uri


def _ui_uri(tracking_uri: str) -> str:
    """Return the public MLflow UI base without persisting secrets in evidence."""
    return _validated_http_uri(os.environ.get("MLFLOW_UI_URI", tracking_uri), "MLflow UI")


def _validated_http_uri(uri: str, label: str) -> str:
    parsed = urlsplit(uri)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} requires an HTTP(S) URI, got {uri!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} URI must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{label} URI must not contain a query or fragment")
    return uri.rstrip("/")


def _require_tolerance(tolerance: float) -> None:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("parity tolerance must be finite and non-negative")


def _require_metrics(baseline: Mapping[str, float], observed: Mapping[str, float]) -> None:
    if not baseline or not observed:
        raise ValueError("parity requires non-empty baseline and observed metrics")
    for source, metrics in (("baseline", baseline), ("observed", observed)):
        for name, value in metrics.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{source} parity metric {name!r} must be finite")


def _require_clean_sources(baseline: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    dirty = [
        name
        for name, record in (("baseline", baseline), ("observed", observed))
        if record.get("dirty")
    ]
    if dirty:
        raise ValueError(f"parity cannot certify dirty {' and '.join(dirty)} source trees")


def _hardware_class(hardware: Mapping[str, Any]) -> dict[str, Any]:
    accelerator = hardware.get("accelerator") or ("cuda" if hardware.get("cuda") else "cpu")
    return {
        "accelerator": accelerator,
        "device": hardware.get("device") or hardware.get("cuda") or hardware.get("cpu"),
        "threads": hardware.get("threads"),
    }


def _require_hardware_class(baseline: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    """Reject parity runs made on a different accelerator/device/thread class."""
    expected = _hardware_class(baseline)
    actual = _hardware_class(observed)
    if actual != expected:
        raise ValueError(f"hardware class mismatch: expected {expected}, observed {actual}")


def _hardware(accelerator: str | None = None) -> dict[str, Any]:
    import torch

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    selected = accelerator or ("cuda" if torch.cuda.is_available() else "cpu")
    selected = "cuda" if selected in {"cuda", "gpu"} else "cpu"
    device = torch.cuda.get_device_name(0) if selected == "cuda" else cpu
    return {
        "accelerator": selected,
        "cpu": cpu,
        "device": device,
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "cuda": device if selected == "cuda" else None,
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
        evaluation = [
            value
            for key, value in record["runs"].items()
            if "evaluation" in key and key.endswith("_run_uri")
        ]
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


def _worker_main(argv: Sequence[str]) -> int:
    """Import and execute one consumer after process-level isolation is established."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--options", required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    module, separator, function = args.flow.partition(":")
    if not separator:
        raise ValueError(f"consumer flow must be module:function, got {args.flow!r}")
    flow: Callable[..., Mapping[str, Any]] = getattr(importlib.import_module(module), function)
    options = json.loads(args.options)
    result = flow(args.data, args.workspace, seed=19, **options)
    representative_result = {"metrics": _metrics(result), **_run_ids(result)}
    args.result.write_text(
        json.dumps(
            {
                "result": representative_result,
                "hardware": _hardware(str(options.get("accelerator", "cpu"))),
            }
        ),
        encoding="utf-8",
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments[:1] == ["_worker"]:
        return _worker_main(arguments[1:])
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("consumers", nargs="+", choices=sorted(CONSUMERS))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true", help="record as the parity baseline")
    mode.add_argument("--compare", action="store_true", help="compare with the baseline")
    parser.add_argument("--tolerance", type=float, default=0.0, help="absolute, per metric")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args(arguments)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))  # consumers import as reference_projects.*
    # Evidence belongs on the live MLflow server, never in a local file store.
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    _require_tolerance(args.tolerance)
    if args.compare:
        missing = [name for name in args.consumers if name not in baseline]
        if missing:
            print(f"no recorded baseline: {', '.join(missing)}", file=sys.stderr)
            return 1
        for name in args.consumers:
            _tag_run_ids(_evaluation_run_ids(baseline[name]))
        _require_baseline_runs([baseline[name] for name in args.consumers])
    for name in args.consumers:
        record = run(name, args.runs_root)
        print(f"{name}: {json.dumps(record['metrics'], sort_keys=True)} in {record['seconds']}s")
        if args.record:
            baseline[name] = record
        else:
            _require_hardware_class(baseline[name]["hardware"], record["hardware"])
            parity = record_parity(
                name,
                baseline[name],
                record,
                tolerance=args.tolerance,
                ui_uri=record["ui_uri"],
            )
            moved = parity["moved"]
            outcome = f"FAILED {moved}" if moved else "holds"
            deltas = {
                metric: {
                    "baseline": baseline[name]["metrics"].get(metric),
                    "observed": record["metrics"].get(metric),
                    "delta": (
                        record["metrics"][metric] - baseline[name]["metrics"][metric]
                        if metric in baseline[name]["metrics"] and metric in record["metrics"]
                        else None
                    ),
                }
                for metric in sorted(set(baseline[name]["metrics"]) | set(record["metrics"]))
            }
            print(
                f"{name}: parity {outcome}; run_id={parity['run_id']}; "
                f"evidence {parity['run_uri']}; deltas={json.dumps(deltas, sort_keys=True)}"
            )
            if moved:
                return 1
    if args.record:
        BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        REPORT.write_text(render(baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
