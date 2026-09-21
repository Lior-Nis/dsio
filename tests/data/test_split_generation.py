from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from dsio.data.adapters import TableExamples
from dsio.data.splits import generate, validate
from dsio.data.splits.folds import folds_from_splits
from dsio.data.splits.models import SplitError, SplitFile
from dsio.data.splits.resolve import resolve_masks


def _examples(order: np.ndarray | None = None) -> TableExamples:
    sample_ids = np.asarray([f"sample-{index:02d}" for index in range(18)])
    groups = np.asarray([f"group-{index // 3}" for index in range(18)])
    labels = np.asarray([index // 3 % 2 for index in range(18)])
    positions = np.arange(18) if order is None else order
    return TableExamples(
        name="cohort",
        sample_ids=sample_ids[positions],
        groups=groups[positions],
        attributes={"label": labels[positions]},
        digest="fixed-dataset-identity",
    )


def test_group_kfold_returns_a_replayable_governed_manifest() -> None:
    examples = _examples()

    manifest = generate(
        examples,
        "group_kfold",
        name="three-fold",
        seed=42,
        parameters={"n_splits": 3},
    )

    assert manifest.store == examples.name
    assert manifest.store_manifest_sha256 == examples.digest
    assert manifest.examples_derivation == examples.derivation
    assert manifest.algorithm == "group_kfold"
    assert manifest.algorithm_version == "1"
    assert manifest.parameters == {"n_splits": 3}
    assert manifest.seed == 42
    assert manifest.dependencies["scikit-learn"]
    assert manifest.coverage == "total"
    assert manifest.validations == (
        "identity_membership",
        "role_disjointness",
        "group_disjointness",
        "total_coverage",
    )
    assert len(manifest.folds) == 3
    assert len(manifest.digest) == 64
    validate(examples, manifest)

    all_ids = set(examples.sample_ids.tolist())
    for fold in manifest.folds:
        assert set(fold.assignments) == {"test", "train"}
        assert set(fold.assignments["test"]).isdisjoint(fold.assignments["train"])
        assert set(fold.assignments["test"]) | set(fold.assignments["train"]) == all_ids
        assert fold.assignments["test"] == sorted(fold.assignments["test"])
        assert fold.parts["test"] == sorted(fold.parts["test"])
        masks = resolve_masks(examples, manifest, fold)
        assert set(examples.sample_ids[masks["test"]].tolist()) == set(fold.assignments["test"])


@pytest.mark.parametrize(
    ("algorithm", "parameters", "folds"),
    [
        ("group_shuffle", {"n_splits": 1, "test_size": 0.34}, 1),
        ("leave_one_group_out", {}, 6),
        ("stratified_group_kfold", {"n_splits": 3, "target": "label"}, 3),
    ],
)
def test_supported_sklearn_algorithms_share_the_same_manifest(
    algorithm: str,
    parameters: dict[str, object],
    folds: int,
) -> None:
    manifest = generate(
        _examples(),
        algorithm,
        name=algorithm,
        seed=7,
        parameters=parameters,
    )

    assert len(manifest.folds) == folds
    assert manifest.algorithm == algorithm
    assert manifest.dependencies.keys() == {"scikit-learn"}
    validate(_examples(), manifest)


def test_generation_is_independent_of_source_iteration_order() -> None:
    ordered = generate(
        _examples(),
        "group_kfold",
        name="stable",
        seed=17,
        parameters={"n_splits": 3},
    )
    shuffled = generate(
        _examples(np.random.default_rng(99).permutation(18)),
        "group_kfold",
        name="stable",
        seed=17,
        parameters={"n_splits": 3},
    )

    assert shuffled.digest == ordered.digest
    assert shuffled.folds == ordered.folds


def test_derived_dataset_identity_is_independent_of_source_iteration_order() -> None:
    ordered = _examples()
    permutation = np.random.default_rng(99).permutation(len(ordered))
    reordered = TableExamples(
        name=ordered.name,
        sample_ids=ordered.sample_ids[permutation],
        groups=ordered.groups[permutation],
        attributes={"label": ordered.attribute("label")[permutation]},
    )
    canonical = TableExamples(
        name=ordered.name,
        sample_ids=ordered.sample_ids,
        groups=ordered.groups,
        attributes={"label": ordered.attribute("label")},
    )

    assert reordered.digest == canonical.digest
    left = generate(canonical, "group_kfold", name="stable", seed=17, parameters={"n_splits": 3})
    right = generate(reordered, "group_kfold", name="stable", seed=17, parameters={"n_splits": 3})
    assert right.digest == left.digest


def test_roles_can_be_named_for_the_task() -> None:
    manifest = generate(
        _examples(),
        "group_shuffle",
        name="calibration",
        seed=3,
        roles=("fit", "calibrate"),
        parameters={"n_splits": 1, "test_size": 0.34},
    )

    assert set(manifest.folds[0].assignments) == {"fit", "calibrate"}
    assert set(resolve_masks(_examples(), manifest, manifest.folds[0])) == {
        "fit",
        "calibrate",
    }


def test_manifest_round_trip_verifies_its_digest(tmp_path: Path) -> None:
    manifest = generate(
        _examples(),
        "group_kfold",
        name="round-trip",
        seed=1,
        parameters={"n_splits": 3},
    )
    path = tmp_path / "split.yaml"
    manifest.save(str(path))

    restored = SplitFile.load(path)
    assert restored == manifest
    assert restored.digest == manifest.digest

    payload = yaml.safe_load(path.read_text())
    payload["folds"][0]["assignments"]["test"].append("forged")
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(SplitError, match="digest"):
        SplitFile.load(path)

    payload.pop("digest")
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(SplitError, match="no content digest"):
        SplitFile.load(path)


def test_manifest_load_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "split.yaml"
    path.write_text("- not\n- a\n- manifest\n")

    with pytest.raises(SplitError, match="YAML mapping"):
        SplitFile.load(path)


def test_manifest_header_escapes_newlines_and_round_trips(tmp_path: Path) -> None:
    examples = TableExamples(
        name="cohort\nname",
        sample_ids=["a", "b", "c", "d"],
        groups=["a", "b", "c", "d"],
        digest="fixed",
    )
    manifest = generate(
        examples,
        "group_shuffle",
        name="split\nname",
        seed=1,
        roles=("fit\nrole", "score"),
        parameters={"test_size": 0.5},
    ).model_copy(update={"notes": "first\nsecond"})
    path = tmp_path / "split.yaml"

    manifest.save(path)

    assert SplitFile.load(path) == manifest
    assert "# dsio split: split\\nname" in path.read_text()


def test_unknown_algorithm_points_to_governed_admission() -> None:
    with pytest.raises(SplitError, match="unknown split algorithm.*experimental admission"):
        generate(_examples(), "project_magic", name="bad", seed=0)


@pytest.mark.parametrize(
    ("algorithm", "parameters", "message"),
    [
        ("group_kfold", {"n_splits": True}, "positive integer"),
        ("group_shuffle", {"test_size": 1.0}, "strictly between"),
        ("stratified_group_kfold", {"n_splits": 3}, "requires.*target"),
        ("leave_one_group_out", {"extra": 1}, "does not accept parameter"),
    ],
)
def test_algorithm_parameters_fail_at_the_dispatch_boundary(
    algorithm: str,
    parameters: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(SplitError, match=message):
        generate(_examples(), algorithm, name="bad", seed=0, parameters=parameters)


@pytest.mark.parametrize("n_splits", [True, 1.0])
def test_temporal_n_splits_is_a_strict_integer(n_splits: object) -> None:
    size = 20
    examples = TableExamples(
        name="timeline",
        sample_ids=[f"tick-{index}" for index in range(size)],
        groups=["market"] * size,
        times=(np.arange(size, dtype=float), np.arange(1, size + 1, dtype=float)),
        digest="strict-temporal",
    )

    with pytest.raises(SplitError, match="invalid purged_walk_forward parameters"):
        generate(
            examples,
            "purged_walk_forward",
            name="bad",
            seed=0,
            parameters={"n_splits": n_splits},
        )


def test_temporal_discarded_role_name_is_reserved() -> None:
    with pytest.raises(SplitError, match="'discarded' is reserved"):
        generate(
            _examples(),
            "purged_walk_forward",
            name="bad",
            seed=0,
            roles=("discarded", "test"),
        )


def test_dispatcher_exposes_no_runtime_registration_hook() -> None:
    import dsio.data.splits as splits

    assert not hasattr(splits, "register")
    assert not hasattr(splits, "register_splitter")


def test_duplicate_source_identity_is_rejected_before_dispatch() -> None:
    examples = _examples()
    examples._sample_ids[1] = examples._sample_ids[0]

    with pytest.raises(SplitError, match="duplicate sample identity"):
        generate(examples, "group_kfold", name="bad", seed=0, parameters={"n_splits": 3})


def test_validation_rejects_unknown_or_overlapping_assignments() -> None:
    manifest = generate(
        _examples(),
        "group_shuffle",
        name="bad",
        seed=0,
        parameters={"n_splits": 1, "test_size": 0.34},
    )
    fold = manifest.folds[0]
    unknown = fold.model_copy(update={"assignments": {**fold.assignments, "test": ["missing"]}})
    with pytest.raises(SplitError, match="unknown sample"):
        validate(_examples(), manifest.model_copy(update={"folds": [unknown]}))

    shared = fold.assignments["train"][0]
    overlap = fold.model_copy(
        update={
            "assignments": {
                **fold.assignments,
                "test": [*fold.assignments["test"], shared],
            }
        }
    )
    with pytest.raises(SplitError, match="more than one role"):
        validate(_examples(), manifest.model_copy(update={"folds": [overlap]}))


def test_validation_rejects_group_leakage_even_if_assignments_cover_every_sample() -> None:
    examples = _examples()
    manifest = generate(
        examples,
        "group_shuffle",
        name="bad",
        seed=0,
        parameters={"n_splits": 1, "test_size": 0.34},
    )
    fold = manifest.folds[0]
    train = list(fold.assignments["train"])
    test = list(fold.assignments["test"])
    train[0], test[0] = test[0], train[0]
    leaking = fold.model_copy(
        update={"assignments": {"train": sorted(train), "test": sorted(test)}}
    )

    with pytest.raises(SplitError, match="leaks group"):
        validate(examples, manifest.model_copy(update={"folds": [leaking]}))


def test_missing_required_role_and_partial_total_coverage_are_rejected() -> None:
    manifest = generate(
        _examples(),
        "group_shuffle",
        name="bad",
        seed=0,
        parameters={"n_splits": 1, "test_size": 0.34},
    )
    fold = manifest.folds[0]
    missing_role = fold.model_copy(update={"assignments": {"train": fold.assignments["train"]}})
    with pytest.raises(SplitError, match="required role 'test'"):
        validate(_examples(), manifest.model_copy(update={"folds": [missing_role]}))

    partial = fold.model_copy(
        update={
            "assignments": {
                "train": fold.assignments["train"][:-1],
                "test": fold.assignments["test"],
            }
        }
    )
    with pytest.raises(SplitError, match="total coverage"):
        validate(_examples(), manifest.model_copy(update={"folds": [partial]}))


def test_purged_walk_forward_uses_the_same_dispatcher_and_records_discards() -> None:
    size = 30
    examples = TableExamples(
        name="timeline",
        sample_ids=[f"tick-{index:02d}" for index in range(size)],
        groups=["market"] * size,
        times=(np.arange(size, dtype=float), np.arange(1, size + 1, dtype=float)),
        digest="timeline-v1",
    )

    manifest = generate(
        examples,
        "purged_walk_forward",
        name="walk-forward",
        seed=11,
        parameters={
            "n_splits": 2,
            "test_fraction": 0.2,
            "label_horizon": 1.0,
            "embargo": 1.0,
        },
    )

    assert manifest.dependencies == {}
    assert manifest.coverage == "partial"
    assert len(manifest.folds) == 2
    assert all(fold.counts["discarded"] > 0 for fold in manifest.folds)
    validate(examples, manifest)
    first = resolve_masks(examples, manifest, manifest.folds[0])
    assert set(examples.sample_ids[first["test"]].tolist()) == set(
        manifest.folds[0].assignments["test"]
    )

    fold = manifest.folds[0]
    assigned = set().union(*map(set, fold.assignments.values()))
    discarded_id = next(iter(set(examples.sample_ids.tolist()) - assigned))
    forged_assignments = {
        **fold.assignments,
        "test": [discarded_id],
    }
    forged = fold.model_copy(
        update={
            "assignments": forged_assignments,
            "counts": {
                "train": len(forged_assignments["train"]),
                "test": 1,
                "discarded": size - len(set().union(*map(set, forged_assignments.values()))),
            },
        }
    )
    with pytest.raises(SplitError, match="assignments disagree with its temporal bounds"):
        validate(examples, manifest.model_copy(update={"folds": [forged]}))

    wrong_discarded = fold.model_copy(
        update={"counts": {**fold.counts, "discarded": fold.counts["discarded"] + 1}}
    )
    with pytest.raises(SplitError, match="discarded count"):
        validate(examples, manifest.model_copy(update={"folds": [wrong_discarded]}))


def test_validation_rechecks_cross_fold_evaluation_identity_for_custom_roles() -> None:
    manifest = generate(
        _examples(),
        "group_kfold",
        name="custom-roles",
        seed=4,
        roles=("fit", "score"),
        parameters={"n_splits": 3},
    )
    first, second, *remaining = manifest.folds
    repeated = first.assignments["score"][0]
    repeated_group = str(_examples().groups[_examples().sample_ids == repeated][0])
    displaced_group = second.parts["score"][0]
    group_by_id = dict(
        zip(_examples().sample_ids.tolist(), _examples().groups.tolist(), strict=True)
    )
    fit_groups = (set(second.parts["fit"]) - {repeated_group}) | {displaced_group}
    score_groups = (set(second.parts["score"]) - {displaced_group}) | {repeated_group}
    assignments = {
        "fit": sorted(
            sample_id
            for sample_id in _examples().sample_ids.tolist()
            if group_by_id[sample_id] in fit_groups
        ),
        "score": sorted(
            sample_id
            for sample_id in _examples().sample_ids.tolist()
            if group_by_id[sample_id] in score_groups
        ),
    }
    parts = {
        "fit": sorted(fit_groups),
        "score": sorted(score_groups),
    }
    forged = second.model_copy(
        update={
            "assignments": assignments,
            "parts": parts,
            "counts": {role: len(values) for role, values in assignments.items()},
        }
    )

    with pytest.raises(SplitError, match="evaluation sample.*appears in folds"):
        validate(
            _examples(),
            manifest.model_copy(update={"folds": [first, forged, *remaining]}),
        )


def test_purged_walk_forward_defaults_are_non_overlapping_and_replayable() -> None:
    size = 100
    examples = TableExamples(
        name="timeline",
        sample_ids=[f"tick-{index:03d}" for index in range(size)],
        groups=["market"] * size,
        times=(np.arange(size, dtype=float), np.arange(1, size + 1, dtype=float)),
        digest="timeline-defaults",
    )

    manifest = generate(examples, "purged_walk_forward", name="defaults", seed=0)

    evaluation_ids = [
        sample_id for fold in manifest.folds for sample_id in fold.assignments["test"]
    ]
    assert len(manifest.folds) == 5
    assert len(evaluation_ids) == len(set(evaluation_ids))
    assert all("discarded" in fold.counts for fold in manifest.folds)

    with pytest.raises(SplitError, match="invalid generated split"):
        generate(
            examples,
            "purged_walk_forward",
            name="overlapping",
            seed=0,
            parameters={"n_splits": 5, "test_fraction": 0.2},
        )


def test_temporal_validation_rejects_extra_assignment_roles() -> None:
    size = 30
    examples = TableExamples(
        name="timeline",
        sample_ids=[f"tick-{index:02d}" for index in range(size)],
        groups=["market"] * size,
        times=(np.arange(size, dtype=float), np.arange(1, size + 1, dtype=float)),
        digest="timeline-extra-role",
    )
    manifest = generate(
        examples,
        "purged_walk_forward",
        name="extra-role",
        seed=0,
        parameters={
            "n_splits": 1,
            "test_fraction": 0.2,
            "label_horizon": 1.0,
            "embargo": 1.0,
        },
    )
    fold = manifest.folds[0]
    assigned = set().union(*map(set, fold.assignments.values()))
    discarded = next(iter(set(examples.sample_ids.tolist()) - assigned))
    forged = fold.model_copy(
        update={
            "assignments": {**fold.assignments, "discarded_bucket": [discarded]},
            "counts": {
                **fold.counts,
                "discarded_bucket": 1,
                "discarded": fold.counts["discarded"] - 1,
            },
        }
    )

    with pytest.raises(SplitError, match="assignment roles must be exactly"):
        validate(examples, manifest.model_copy(update={"folds": [forged]}))

    assert fold.temporal is not None
    extra_bounds = fold.temporal.model_copy(
        update={"spans": {**fold.temporal.spans, "shadow": fold.temporal.spans["test"]}}
    )
    extra_span = fold.model_copy(update={"temporal": extra_bounds})
    with pytest.raises(SplitError, match="temporal span roles must be exactly"):
        validate(examples, manifest.model_copy(update={"folds": [extra_span]}))


def test_resolution_rejects_a_fold_not_stored_in_the_manifest() -> None:
    examples = _examples()
    manifest = generate(
        examples,
        "group_shuffle",
        name="canonical",
        seed=2,
        parameters={"test_size": 0.34},
    )
    canonical = manifest.folds[0]
    forged = canonical.model_copy(update={"counts": {**canonical.counts, "test": 999}})

    with pytest.raises(SplitError, match="does not match the fold stored"):
        resolve_masks(examples, manifest, forged)


def test_fold_conversion_validates_a_manifest_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import dsio.data.splits.folds as fold_module

    examples = _examples()
    manifest = generate(
        examples,
        "group_kfold",
        name="three-fold",
        seed=4,
        parameters={"n_splits": 3},
    )
    real_validate = fold_module.validate
    calls = 0

    def counted_validate(source: TableExamples, candidate: SplitFile) -> None:
        nonlocal calls
        calls += 1
        real_validate(source, candidate)

    monkeypatch.setattr(fold_module, "validate", counted_validate)

    assert len(folds_from_splits(examples, [manifest])) == 3
    assert calls == 1
