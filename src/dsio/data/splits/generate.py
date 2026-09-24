"""The closed DSIO-owned split dispatcher."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any, Literal

import numpy as np
from pydantic import ValidationError
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
)

from dsio.data.examples import Examples, ExamplesError, assert_consistent, check
from dsio.data.splits.models import STABLE_ALGORITHMS, SplitError, SplitFile, SplitFold
from dsio.data.splits.temporal import TemporalBounds, TemporalSpec, apply, walk_forward
from dsio.data.splits.validation import validate

ALGORITHM_VERSION = "1"
TEMPORAL_ALGORITHM_VERSION = "2"
_SKLEARN_ALGORITHMS = {
    "group_kfold",
    "group_shuffle",
    "leave_one_group_out",
    "stratified_group_kfold",
}
_ALGORITHMS = tuple(sorted(STABLE_ALGORITHMS))


def generate(
    examples: Examples,
    algorithm: str,
    *,
    name: str,
    seed: int,
    roles: tuple[str, str] = ("train", "test"),
    parameters: dict[str, Any] | None = None,
) -> SplitFile:
    """Generate one normalized manifest through a known, non-registerable algorithm."""
    if algorithm not in _ALGORITHMS:
        raise SplitError(
            f"unknown split algorithm {algorithm!r}; supported: {', '.join(_ALGORITHMS)}. "
            "Add novel algorithms through DSIO's governed experimental admission path; "
            "runtime registration is not supported"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitError(f"split seed must be an integer, got {seed!r}")
    if len(roles) != 2 or len(set(roles)) != 2 or any(not role for role in roles):
        raise SplitError("roles must contain two distinct, non-empty names")
    if algorithm == "purged_walk_forward" and "discarded" in roles:
        raise SplitError("role name 'discarded' is reserved for temporal coverage evidence")

    try:
        source = check(examples)
        assert_consistent(source)
    except ExamplesError as exc:
        raise SplitError(str(exc)) from exc

    params = dict(parameters or {})
    coverage: Literal["total", "partial"]
    if algorithm == "purged_walk_forward":
        folds, normalized = _temporal_folds(source, roles, params)
        dependencies: dict[str, str] = {}
        coverage = "partial"
        validations = (
            "identity_membership",
            "role_disjointness",
            "temporal_purge_embargo",
            "partial_coverage",
        )
    else:
        folds, normalized = _grouped_folds(source, algorithm, seed, roles, params)
        dependencies = {"scikit-learn": version("scikit-learn")}
        coverage = "total"
        validations = (
            "identity_membership",
            "role_disjointness",
            "group_disjointness",
            "total_coverage",
        )

    try:
        manifest = SplitFile(
            store=source.name,
            store_manifest_sha256=source.digest,
            examples_derivation=source.derivation,
            name=name,
            algorithm=algorithm,
            algorithm_version=(
                TEMPORAL_ALGORITHM_VERSION
                if algorithm == "purged_walk_forward"
                else ALGORITHM_VERSION
            ),
            parameters=normalized,
            seed=seed,
            dependencies=dependencies,
            validations=validations,
            coverage=coverage,
            required_roles=roles,
            folds=folds,
        )
    except ValidationError as exc:
        messages = [str(error["msg"]).removeprefix("Value error, ") for error in exc.errors()]
        raise SplitError("invalid generated split: " + "; ".join(messages)) from exc
    validate(source, manifest)
    return manifest


def _ordered_source(examples: Examples) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_ids = np.asarray(examples.sample_ids, dtype=str)
    order = np.argsort(sample_ids, kind="stable")
    return sample_ids[order], np.asarray(examples.groups, dtype=str)[order], order


def _grouped_folds(
    examples: Examples,
    algorithm: str,
    seed: int,
    roles: tuple[str, str],
    parameters: dict[str, Any],
) -> tuple[list[SplitFold], dict[str, Any]]:
    sample_ids, groups, order = _ordered_source(examples)
    train_role, test_role = roles
    x = np.zeros((len(sample_ids), 1), dtype=np.uint8)
    y: np.ndarray | None = None

    if algorithm == "group_kfold":
        _only(parameters, {"n_splits"}, algorithm)
        n_splits = _positive_int(parameters.get("n_splits", 5), "n_splits")
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        normalized: dict[str, Any] = {"n_splits": n_splits}
    elif algorithm == "stratified_group_kfold":
        _only(parameters, {"n_splits", "target"}, algorithm)
        n_splits = _positive_int(parameters.get("n_splits", 5), "n_splits")
        target = parameters.get("target")
        if not isinstance(target, str) or not target:
            raise SplitError("stratified_group_kfold requires a non-empty 'target' attribute")
        try:
            y = np.asarray(examples.attribute(target))[order]
        except ExamplesError as exc:
            raise SplitError(str(exc)) from exc
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        normalized = {"n_splits": n_splits, "target": target}
    elif algorithm == "group_shuffle":
        _only(parameters, {"n_splits", "test_size"}, algorithm)
        n_splits = _positive_int(parameters.get("n_splits", 1), "n_splits")
        if n_splits != 1:
            raise SplitError(
                "group_shuffle currently requires n_splits=1 because repeated random "
                "test membership is not a replay-safe cross-validation family"
            )
        test_size = _fraction(parameters.get("test_size", 0.2), "test_size")
        splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
        normalized = {"n_splits": n_splits, "test_size": test_size}
    else:
        _only(parameters, set(), algorithm)
        splitter = LeaveOneGroupOut()
        normalized = {}

    try:
        raw_folds = list(splitter.split(x, y, groups))
    except ValueError as exc:
        raise SplitError(f"{algorithm} cannot split dataset {examples.name!r}: {exc}") from exc
    return (
        [
            _fold_from_positions(
                index,
                sample_ids,
                groups,
                train_positions,
                test_positions,
                train_role,
                test_role,
            )
            for index, (train_positions, test_positions) in enumerate(raw_folds)
        ],
        normalized,
    )


def _fold_from_positions(
    index: int,
    sample_ids: np.ndarray,
    groups: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_role: str,
    test_role: str,
) -> SplitFold:
    assignments = {
        train_role: sorted(sample_ids[train_positions].tolist()),
        test_role: sorted(sample_ids[test_positions].tolist()),
    }
    parts = {
        train_role: sorted(set(groups[train_positions].tolist())),
        test_role: sorted(set(groups[test_positions].tolist())),
    }
    return SplitFold(
        index=index,
        counts={role: len(values) for role, values in assignments.items()},
        parts=parts,
        assignments=assignments,
    )


def _temporal_folds(
    examples: Examples,
    roles: tuple[str, str],
    parameters: dict[str, Any],
) -> tuple[list[SplitFold], dict[str, Any]]:
    allowed = set(TemporalSpec.model_fields)
    _only(parameters, allowed, "purged_walk_forward")
    try:
        spec = TemporalSpec.model_validate(parameters)
    except ValidationError as exc:
        raise SplitError(f"invalid purged_walk_forward parameters: {exc}") from exc
    times = examples.times()
    if times is None:
        raise SplitError(
            f"purged_walk_forward requires time coordinates; dataset {examples.name!r} has none"
        )
    starts, ends = times
    train_role, test_role = roles
    sample_ids = np.asarray(examples.sample_ids, dtype=str)
    folds: list[SplitFold] = []
    try:
        bounds_list = walk_forward(starts, ends, spec)
    except ValueError as exc:
        raise SplitError(f"purged_walk_forward cannot split {examples.name!r}: {exc}") from exc
    for index, bounds in enumerate(bounds_list):
        train_mask = apply(bounds, starts, ends, part="train")
        test_mask = apply(bounds, starts, ends, part="test")
        assignments = {
            train_role: sorted(sample_ids[train_mask].tolist()),
            test_role: sorted(sample_ids[test_mask].tolist()),
        }
        renamed_bounds = TemporalBounds(
            time_unit=bounds.time_unit,
            label_horizon=bounds.label_horizon,
            embargo=bounds.embargo,
            spans={train_role: bounds.spans["train"], test_role: bounds.spans["test"]},
        )
        counts = {role: len(values) for role, values in assignments.items()}
        counts["discarded"] = len(sample_ids) - sum(counts.values())
        folds.append(
            SplitFold(
                index=index,
                counts=counts,
                assignments=assignments,
                temporal=renamed_bounds,
            )
        )
    return folds, spec.model_dump(mode="json")


def _only(parameters: dict[str, Any], allowed: set[str], algorithm: str) -> None:
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise SplitError(
            f"{algorithm} does not accept parameter {unknown[0]!r}; allowed: "
            f"{', '.join(sorted(allowed)) or 'none'}"
        )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SplitError(f"{name} must be a positive integer, got {value!r}")
    return value


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SplitError(f"{name} must be a number strictly between 0 and 1, got {value!r}")
    result = float(value)
    if not 0.0 < result < 1.0:
        raise SplitError(f"{name} must be strictly between 0 and 1, got {value!r}")
    return result
