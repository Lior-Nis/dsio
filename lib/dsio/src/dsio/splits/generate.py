"""Generate split files from a declarative spec.

The spec is what you write; the YAML files are what it produces, and those get committed.
Generation is reproducible from the spec, so nobody has to re-run it to read a result — but
nobody hand-maintains 145 files either, which is how FORGE's split directory grew.

Stratified assignment is **serpentine**, not random. Sorting groups by their stratified
value and dealing them back and forth across folds keeps the totals close, which random
assignment over a small number of subjects reliably fails to do. FORGE reached the same
approach the hard way.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from dsio.data.store import SignalStore
from dsio.data.views import WindowIndex
from dsio.splits.models import SplitError, SplitFile, SplitSpec
from dsio.splits.stratify import BalanceReport, stratified_partition
from dsio.splits.temporal import TemporalSpec, describe, walk_forward, window_times


def group_values(store: SignalStore, attr: str | None) -> dict[str, float]:
    """Aggregate an entity attribute up to the group level.

    Stratification happens over groups because groups are what get assigned. An attribute
    recorded per recording is summed across the recordings belonging to that group.
    """
    totals: dict[str, float] = defaultdict(float)
    for entity in store.entities:
        totals[entity.group] += 0.0 if attr is None else float(entity.attrs.get(attr, 0.0))
    return dict(totals)


def _shuffled(groups: list[str], seed: int, k: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    return [sorted(shuffled[i::k]) for i in range(k)]


def generate(
    store: SignalStore,
    spec: SplitSpec,
    *,
    name: str,
) -> list[SplitFile]:
    """Produce one :class:`SplitFile` per fold.

    Every group in the store lands in exactly one part of every fold, so a split is total:
    no recording is silently dropped, which is a quiet way to lose data and never notice.
    """
    all_groups = store.groups
    pinned = [g for g in spec.always_train if g in all_groups]
    unknown = set(spec.always_train) - set(all_groups)
    if unknown:
        raise SplitError(
            f"always_train names groups absent from store {store.path.name!r}: "
            f"{', '.join(sorted(unknown))}"
        )
    poolable = [g for g in all_groups if g not in set(pinned)]
    if not poolable:
        raise SplitError("every group is pinned to train; nothing left to split")

    manifest_sha = None
    try:
        manifest_sha = store.manifest().signal_sha256
    except Exception:  # noqa: BLE001 - a store without a manifest is still splittable
        manifest_sha = None

    files: list[SplitFile] = []

    if spec.scheme == "leave_one_group_out":
        for i, held in enumerate(poolable):
            held_val = poolable[(i + 1) % len(poolable)] if len(poolable) > 1 else None
            train = [g for g in poolable if g not in {held, held_val}]
            parts = {"train": sorted(train + pinned), "test": [held]}
            if held_val is not None:
                parts["val"] = [held_val]
            files.append(_build(store, spec, name, i, parts, manifest_sha))
        return files

    if spec.scheme == "holdout":
        if spec.keys:
            buckets, hold_report = stratified_partition(
                poolable, store.entities, spec.keys, 5, seed=spec.seed
            )
        else:
            buckets, hold_report = _shuffled(poolable, spec.seed, 5), None
        flat = [g for bucket in buckets for g in bucket]
        n_test = max(1, round(len(flat) * spec.test_fraction))
        n_val = max(1, round(len(flat) * spec.val_fraction))
        parts = {
            "test": sorted(flat[:n_test]),
            "val": sorted(flat[n_test : n_test + n_val]),
            "train": sorted(flat[n_test + n_val :] + pinned),
        }
        return [_build(store, spec, name, None, parts, manifest_sha, hold_report)]

    k = spec.k
    if k > len(poolable):
        raise SplitError(
            f"cannot make {k} folds from {len(poolable)} splittable groups in "
            f"{store.path.name!r}"
        )
    report: BalanceReport | None = None
    if spec.scheme == "stratified_kfold":
        buckets, report = stratified_partition(
            poolable, store.entities, spec.keys, k, seed=spec.seed
        )
    else:
        buckets = _shuffled(poolable, spec.seed, k)
    for fold in range(k):
        test = buckets[fold]
        val = _carve_validation(buckets, fold, k, spec.val_fraction)
        used = set(test) | set(val)
        train = [g for g in poolable if g not in used]
        parts = {"train": sorted(train + pinned), "test": sorted(test)}
        if val:
            parts["val"] = sorted(val)
        files.append(_build(store, spec, name, fold, parts, manifest_sha, report))
    return files


def _carve_validation(
    buckets: list[list[str]], fold: int, k: int, val_fraction: float
) -> list[str]:
    """Take the validation groups out of the *training* portion, spread across buckets.

    An earlier version handed validation a whole rotation bucket, which made train
    ``(k-2)/k``. At k=3 that is a 33% training set against a 67% test-plus-validation
    — smaller than what it is evaluated on, which is not what anyone means by "3-fold
    cross-validation", and it silently starves the model. Nothing in the split file's shape
    reveals it; you only notice because the model does not learn.

    Groups are taken one at a time round-robin from the non-test buckets, so whatever
    balance stratification achieved across buckets survives into the validation set instead
    of being concentrated in whichever bucket happened to be next.
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


def _build(
    store: SignalStore,
    spec: SplitSpec,
    name: str,
    fold: int | None,
    parts: dict[str, list[str]],
    manifest_sha: str | None,
    balance: BalanceReport | None = None,
) -> SplitFile:
    covered = {g for groups in parts.values() for g in groups}
    missing = set(store.groups) - covered
    if missing:
        raise SplitError(
            f"split {name!r} leaves {len(missing)} group(s) unassigned: "
            f"{', '.join(sorted(missing)[:5])}"
        )
    return SplitFile(
        store=store.path.name,
        store_manifest_sha256=manifest_sha,
        group_key=spec.group_key,
        name=name,
        fold=fold,
        spec=spec,
        counts={part: len(groups) for part, groups in parts.items()},
        balance=balance,
        parts=parts,
    )


def write_splits(
    store: SignalStore,
    spec: SplitSpec,
    *,
    name: str,
    root: Path,
) -> list[Path]:
    """Generate and write split files under ``root/<name>/``."""
    paths: list[Path] = []
    for split in generate(store, spec, name=name):
        filename = "split.yaml" if split.fold is None else f"fold{split.fold}.yaml"
        path = root / name / filename
        split.save(path)
        paths.append(path)
    return paths


def generate_temporal(
    store: SignalStore,
    index: WindowIndex,
    spec: TemporalSpec,
    *,
    name: str,
    groups: dict[str, list[str]] | None = None,
) -> list[SplitFile]:
    """Produce one walk-forward :class:`SplitFile` per fold.

    Needs the index as well as the store, because a temporal split divides *time*, and
    where a window sits in time is a property of the index rather than of the corpus.

    ``groups`` optionally composes a group partition on top, which is how you get a
    purged walk-forward *within* a held-out cohort — the correct design for a strategy
    that must generalise across both symbols and time.
    """
    t_start, t_end = window_times(store, index, unit=spec.time_unit)
    manifest_sha = None
    try:
        manifest_sha = store.manifest().signal_sha256
    except Exception:  # noqa: BLE001 - a store without a manifest is still splittable
        manifest_sha = None

    files: list[SplitFile] = []
    for fold, bounds in enumerate(walk_forward(t_start, t_end, spec)):
        counts = describe(bounds, t_start, t_end)
        files.append(
            SplitFile(
                store=store.path.name,
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


def write_temporal_splits(
    store: SignalStore,
    index: WindowIndex,
    spec: TemporalSpec,
    *,
    name: str,
    root: Path,
    groups: dict[str, list[str]] | None = None,
) -> list[Path]:
    """Generate and write walk-forward split files under ``root/<name>/``."""
    paths: list[Path] = []
    for split in generate_temporal(store, index, spec, name=name, groups=groups):
        path = root / name / f"fold{split.fold}.yaml"
        split.save(path)
        paths.append(path)
    return paths
