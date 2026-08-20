"""Test-only split-file builders.

`dsio.splits.generate` is gone: offline generation is a project concern now, not the
skeleton's. These tests still need split *files* to exercise the reader (`resolve`,
`folds_from_splits`, the CLI, the training loop), so this module builds them directly with
`SplitFile` — the same shape a project's own generator would produce — instead of keeping a
generator alive just to feed a fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dsio.data.examples import Examples
from dsio.splits import SplitError, SplitFile, TemporalSpec, describe, walk_forward


def _shuffled_buckets(groups: list[str], seed: int, k: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    return [sorted(shuffled[i::k]) for i in range(k)]


def _carve_validation(
    buckets: list[list[str]], fold: int, k: int, val_fraction: float
) -> list[str]:
    """Take validation out of the *training* portion, spread across the other buckets.

    Round-robin rather than a straight slice, so whatever balance the bucketing achieved
    survives into validation instead of draining one bucket first. Mirrors the carving the
    deleted generator used, so a fixture built here behaves the way one it wrote did.
    """
    if val_fraction <= 0.0:
        return []
    remaining = [buckets[(fold + offset) % k] for offset in range(1, k)]
    available = sum(len(bucket) for bucket in remaining)
    wanted = min(available - 1, max(1, round(available * val_fraction)))
    if wanted <= 0:
        return []

    validation: list[str] = []
    cursors = [0] * len(remaining)
    while len(validation) < wanted:
        progressed = False
        for i, bucket in enumerate(remaining):
            if len(validation) >= wanted:
                break
            if cursors[i] < len(bucket):
                validation.append(bucket[cursors[i]])
                cursors[i] += 1
                progressed = True
        if not progressed:  # pragma: no cover - wanted is bounded by available
            break
    return validation


def kfold_split_files(
    examples: Examples,
    k: int,
    *,
    name: str,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> list[SplitFile]:
    """``k`` plain folds, the same shape ``generate(scheme="kfold")`` used to produce."""
    groups = sorted({str(g) for g in examples.groups})
    buckets = _shuffled_buckets(groups, seed, k)
    manifest_sha = examples.digest
    files: list[SplitFile] = []
    for fold in range(k):
        test = buckets[fold]
        val = _carve_validation(buckets, fold, k, val_fraction)
        used = set(test) | set(val)
        train = [g for g in groups if g not in used]
        parts = {"train": sorted(train), "test": sorted(test)}
        if val:
            parts["val"] = sorted(val)
        files.append(
            SplitFile(
                store=examples.name,
                store_manifest_sha256=manifest_sha,
                name=name,
                fold=fold,
                counts={part: len(members) for part, members in parts.items()},
                parts=parts,
            )
        )
    return files


def logo_split_files(examples: Examples, *, name: str) -> list[SplitFile]:
    """One fold per group, leave-one-group-out."""
    groups = sorted({str(g) for g in examples.groups})
    manifest_sha = examples.digest
    files: list[SplitFile] = []
    for i, held in enumerate(groups):
        train = [g for g in groups if g != held]
        files.append(
            SplitFile(
                store=examples.name,
                store_manifest_sha256=manifest_sha,
                name=name,
                fold=i,
                counts={"train": len(train), "test": 1},
                parts={"train": sorted(train), "test": [held]},
            )
        )
    return files


def temporal_split_files(
    examples: Examples,
    spec: TemporalSpec,
    *,
    name: str,
    groups: dict[str, list[str]] | None = None,
) -> list[SplitFile]:
    """One walk-forward fold per :func:`~dsio.splits.walk_forward` bound."""
    times = examples.times()
    if times is None:
        raise SplitError(
            f"{examples.name!r} has no time coordinates, so it cannot be split temporally"
        )
    t_start, t_end = times
    manifest_sha = examples.digest
    files: list[SplitFile] = []
    for fold, bounds in enumerate(walk_forward(t_start, t_end, spec)):
        counts = describe(bounds, t_start, t_end)
        files.append(
            SplitFile(
                store=examples.name,
                store_manifest_sha256=manifest_sha,
                name=name,
                fold=fold,
                counts=counts,
                parts=groups or {},
                temporal=bounds,
                notes=(
                    f"{counts['discarded']} window(s) discarded by purge and embargo"
                    if counts.get("discarded")
                    else None
                ),
            )
        )
    return files


def write_split_files(files: list[SplitFile], *, name: str, root: Path) -> list[Path]:
    """Save each fold under ``root/<name>/``, the same layout ``write_splits`` used."""
    paths: list[Path] = []
    for split in files:
        filename = "split.yaml" if split.fold is None else f"fold{split.fold}.yaml"
        path = root / name / filename
        split.save(path)
        paths.append(path)
    return paths
