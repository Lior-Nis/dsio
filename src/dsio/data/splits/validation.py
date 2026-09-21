"""Manifest invariants that require the concrete source examples."""

from __future__ import annotations

from collections import Counter

import numpy as np

from dsio.data.examples import Examples, ExamplesError, assert_consistent, check
from dsio.data.splits.models import STABLE_ALGORITHMS, SplitError, SplitFile
from dsio.data.splits.temporal import apply as apply_temporal


def validate(examples: Examples, manifest: SplitFile) -> None:
    """Fail unless ``manifest`` is an exact, leakage-safe division of ``examples``."""
    try:
        source = check(examples)
        assert_consistent(source)
    except ExamplesError as exc:
        raise SplitError(str(exc)) from exc

    if manifest.algorithm not in STABLE_ALGORITHMS:
        raise SplitError(
            f"split {manifest.name!r} uses unknown governed algorithm {manifest.algorithm!r}"
        )
    temporal = manifest.algorithm == "purged_walk_forward"
    expected_coverage = "partial" if temporal else "total"
    if manifest.coverage != expected_coverage:
        raise SplitError(
            f"split {manifest.name!r} algorithm {manifest.algorithm!r} requires "
            f"{expected_coverage} coverage, not {manifest.coverage}"
        )
    expected_validations = (
        (
            "identity_membership",
            "role_disjointness",
            "temporal_purge_embargo",
            "partial_coverage",
        )
        if temporal
        else (
            "identity_membership",
            "role_disjointness",
            "group_disjointness",
            "total_coverage",
        )
    )
    if manifest.validations != expected_validations:
        raise SplitError(
            f"split {manifest.name!r} does not declare the required validation set for "
            f"{manifest.algorithm!r}"
        )
    if not temporal and "scikit-learn" not in manifest.dependencies:
        raise SplitError(f"split {manifest.name!r} has no scikit-learn dependency provenance")

    if manifest.store != source.name:
        raise SplitError(
            f"split {manifest.name!r} was built for dataset {manifest.store!r}, not {source.name!r}"
        )
    if manifest.store_manifest_sha256 != source.digest:
        raise SplitError(
            f"split {manifest.name!r} was built for dataset digest "
            f"{manifest.store_manifest_sha256!r}, not {source.digest!r}"
        )
    if manifest.examples_derivation != source.derivation:
        raise SplitError(
            f"split {manifest.name!r} was built for derivation "
            f"{manifest.examples_derivation!r}, not {source.derivation!r}"
        )

    sample_ids = [str(value) for value in np.asarray(source.sample_ids).tolist()]
    duplicate_source = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    if duplicate_source:
        raise SplitError(
            f"dataset {source.name!r} has duplicate sample identity {duplicate_source[0]!r}"
        )
    source_ids = set(sample_ids)
    group_by_id = {
        sample_id: str(group)
        for sample_id, group in zip(sample_ids, np.asarray(source.groups).tolist(), strict=True)
    }
    times = source.times() if temporal else None
    if temporal and times is None:
        raise SplitError(
            f"split {manifest.name!r} requires time coordinates, but {source.name!r} has none"
        )
    evaluation_owners: dict[str, int] = {}

    for fold in manifest.folds:
        assignment_roles = set(fold.assignments)
        required_roles = set(manifest.required_roles)
        missing_roles = [role for role in manifest.required_roles if role not in fold.assignments]
        if missing_roles:
            raise SplitError(
                f"split {manifest.name!r} fold {fold.index} is missing required role "
                f"{missing_roles[0]!r}"
            )
        if assignment_roles != required_roles:
            raise SplitError(
                f"split {manifest.name!r} fold {fold.index} assignment roles must be "
                f"exactly {sorted(required_roles)}, got {sorted(assignment_roles)}"
            )
        if not temporal and set(fold.parts) != required_roles:
            raise SplitError(
                f"split {manifest.name!r} fold {fold.index} group roles must be exactly "
                f"{sorted(required_roles)}, got {sorted(fold.parts)}"
            )
        owners: dict[str, str] = {}
        role_groups: dict[str, set[str]] = {}
        for role, assignments in fold.assignments.items():
            if not assignments:
                raise SplitError(
                    f"split {manifest.name!r} fold {fold.index} role {role!r} is empty"
                )
            role_groups[role] = set()
            for sample_id in assignments:
                if sample_id not in source_ids:
                    raise SplitError(
                        f"split {manifest.name!r} fold {fold.index} names unknown sample "
                        f"identity {sample_id!r}"
                    )
                previous = owners.get(sample_id)
                if previous is not None:
                    raise SplitError(
                        f"split {manifest.name!r} fold {fold.index} sample {sample_id!r} "
                        f"appears in more than one role: {previous!r} and {role!r}"
                    )
                owners[sample_id] = role
                role_groups[role].add(group_by_id[sample_id])

        if manifest.coverage == "total" and set(owners) != source_ids:
            missing = sorted(source_ids - set(owners))
            raise SplitError(
                f"split {manifest.name!r} fold {fold.index} violates total coverage; "
                f"{len(missing)} sample(s) are unassigned"
            )

        if not temporal:
            roles = sorted(role_groups)
            for index, left in enumerate(roles):
                for right in roles[index + 1 :]:
                    overlap = role_groups[left] & role_groups[right]
                    if overlap:
                        raise SplitError(
                            f"split {manifest.name!r} fold {fold.index} leaks group "
                            f"{sorted(overlap)[0]!r} across roles {left!r} and {right!r}"
                        )

        for role, declared_groups in fold.parts.items():
            actual_groups = role_groups.get(role, set())
            if set(declared_groups) != actual_groups:
                raise SplitError(
                    f"split {manifest.name!r} fold {fold.index} role {role!r} group "
                    "evidence disagrees with its sample assignments"
                )

        assigned_counts = {role: len(values) for role, values in fold.assignments.items()}
        expected_count_names = set(assigned_counts)
        if temporal:
            expected_count_names.add("discarded")
        if set(fold.counts) != expected_count_names:
            raise SplitError(
                f"split {manifest.name!r} fold {fold.index} count names do not match "
                f"its assignments: expected {sorted(expected_count_names)}"
            )
        for role, count in assigned_counts.items():
            if fold.counts.get(role) != count:
                raise SplitError(
                    f"split {manifest.name!r} fold {fold.index} count for {role!r} is "
                    f"{fold.counts.get(role)!r}, expected {count}"
                )

        if temporal:
            if fold.temporal is None:
                raise SplitError(
                    f"split {manifest.name!r} fold {fold.index} has no temporal bounds"
                )
            assert times is not None
            starts, ends = times
            for role in manifest.required_roles:
                expected = {
                    sample_ids[index]
                    for index in np.flatnonzero(
                        apply_temporal(
                            fold.temporal,
                            starts,
                            ends,
                            part=role,
                            test_part=manifest.required_roles[1],
                        )
                    )
                }
                actual = set(fold.assignments[role])
                if actual != expected:
                    raise SplitError(
                        f"split {manifest.name!r} fold {fold.index} role {role!r} "
                        "assignments disagree with its temporal bounds"
                    )
            discarded = len(source_ids) - len(owners)
            if fold.counts["discarded"] != discarded:
                raise SplitError(
                    f"split {manifest.name!r} fold {fold.index} discarded count is "
                    f"{fold.counts['discarded']}, expected {discarded}"
                )

        evaluation_role = manifest.required_roles[1]
        for sample_id in fold.assignments[evaluation_role]:
            previous_fold = evaluation_owners.get(sample_id)
            if previous_fold is not None:
                raise SplitError(
                    f"split {manifest.name!r} evaluation sample {sample_id!r} appears "
                    f"in folds {previous_fold} and {fold.index}"
                )
            evaluation_owners[sample_id] = fold.index
