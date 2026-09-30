"""Replay assertions that explain *why* an execution identity changed.

A bare ``first["identities"] == second["identities"]`` only reports two different hashes.
When a replay diverges, the useful evidence is which provenance field moved, so on a
mismatch this helper downloads both runs' ``provenance.json`` and reports a field-level
diff, plus the files named in each run's captured ``git.patch`` when the code hash moved.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

_MISSING = object()


def assert_same_identities(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    """Assert two flow results share execution identities, explaining any divergence."""
    first_identities = first["identities"]
    second_identities = second["identities"]
    if first_identities == second_identities:
        return
    lines = ["replay execution identities differ:"]
    for stage in sorted(set(first_identities) | set(second_identities)):
        left = first_identities.get(stage)
        right = second_identities.get(stage)
        if left == right:
            continue
        lines.append(f"- {stage}: {left} != {right}")
        left_run = first.get(f"{stage}_run_id")
        right_run = second.get(f"{stage}_run_id")
        if isinstance(left_run, str) and isinstance(right_run, str):
            diff = provenance_diff(left_run, right_run)
            lines.extend(f"    {line}" for line in diff)
            if any(line.startswith("configuration.checkpoint_digest") for line in diff):
                lines.extend(f"    {line}" for line in _training_checkpoint_diff(first, second))
        else:
            lines.append(f"    (no {stage}_run_id in both results; provenance not compared)")
    raise AssertionError("\n".join(lines))


def provenance_diff(first_run_id: str, second_run_id: str) -> list[str]:
    """Return one line per provenance field that differs between two runs."""
    with TemporaryDirectory(prefix="dsio-replay-diff-") as directory:
        root = Path(directory)
        first = _provenance(first_run_id, root / "first")
        second = _provenance(second_run_id, root / "second")
        first_fields = _flatten(first)
        second_fields = _flatten(second)
        lines = [
            f"{path}: {first_fields.get(path, '<absent>')!r} != "
            f"{second_fields.get(path, '<absent>')!r}"
            for path in sorted(set(first_fields) | set(second_fields))
            if path != "execution_identity"
            and first_fields.get(path, _MISSING) != second_fields.get(path, _MISSING)
        ]
        if any(line.startswith("execution.git.") for line in lines):
            lines.append(f"git.patch files (first): {_patched_files(first_run_id, root / 'fp')}")
            lines.append(f"git.patch files (second): {_patched_files(second_run_id, root / 'sp')}")
    return lines or ["(provenance documents are identical; identity inputs differ elsewhere)"]


def _training_checkpoint_diff(first: Mapping[str, Any], second: Mapping[str, Any]) -> list[str]:
    """Explain a moved checkpoint digest: numeric weight drift or non-tensor metadata."""
    left_run = first.get("train_run_id")
    right_run = second.get("train_run_id")
    if not (isinstance(left_run, str) and isinstance(right_run, str)):
        return ["(no train_run_id in both results; checkpoints not compared)"]
    try:
        left = _flatten_checkpoint(_checkpoint(left_run))
        right = _flatten_checkpoint(_checkpoint(right_run))
    except Exception as error:  # diagnostics must never mask the original mismatch
        return [f"(checkpoints could not be compared: {type(error).__name__}: {error})"]
    lines = []
    for path in sorted(set(left) | set(right)):
        a, b = left.get(path, _MISSING), right.get(path, _MISSING)
        if a is _MISSING or b is _MISSING:
            lines.append(f"checkpoint {path}: present in only one checkpoint")
        elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                lines.append(
                    f"checkpoint {path}: {a.dtype}{list(a.shape)} != {b.dtype}{list(b.shape)}"
                )
            elif not torch.equal(a, b):
                drift = (a.double() - b.double()).abs().max().item()
                lines.append(f"checkpoint {path}: max abs diff {drift:.3e}")
        elif repr(a) != repr(b):
            lines.append(f"checkpoint {path}: {a!r} != {b!r}")
    return lines or ["(checkpoint contents are identical; only the serialized bytes differ)"]


def _checkpoint(run_id: str) -> Mapping[str, Any]:
    from dsio.train.artifacts import ArtifactRef, load_artifact

    with TemporaryDirectory(prefix="dsio-replay-ckpt-") as directory:
        path = MlflowClient().download_artifacts(run_id, "outputs/checkpoint.json", directory)
        reference = ArtifactRef.model_validate_json(Path(path).read_text(encoding="utf-8"))
    return torch.load(io.BytesIO(load_artifact(reference)), map_location="cpu", weights_only=False)


def _flatten_checkpoint(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        fields: dict[str, Any] = {}
        for key, item in value.items():
            fields.update(_flatten_checkpoint(item, f"{prefix}.{key}" if prefix else str(key)))
        return fields
    if isinstance(value, list | tuple) and not isinstance(value, str):
        fields = {}
        for index, item in enumerate(value):
            fields.update(_flatten_checkpoint(item, f"{prefix}[{index}]"))
        return fields
    return {prefix: value}


def _provenance(run_id: str, destination: Path) -> Mapping[str, Any]:
    destination.mkdir(parents=True)
    path = MlflowClient().download_artifacts(run_id, "provenance.json", str(destination))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _patched_files(run_id: str, destination: Path) -> list[str]:
    destination.mkdir(parents=True)
    try:
        path = MlflowClient().download_artifacts(run_id, "git.patch", str(destination))
    except (MlflowException, OSError):
        return []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return [
        line.split(" b/", 1)[-1] if line.startswith("diff --git") else line[2:].strip()
        for line in text.splitlines()
        if line.startswith("diff --git") or line.startswith("# untracked binary")
    ]


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        fields: dict[str, Any] = {}
        for key, item in value.items():
            fields.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return fields
    return {prefix: value}
