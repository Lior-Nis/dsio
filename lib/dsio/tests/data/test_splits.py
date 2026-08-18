"""Split invariants. The leakage tests here are the point of the whole module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsio.data import SignalExamples, SignalStore, WindowSpec, build_index, entity_examples
from dsio.splits import (
    SplitError,
    SplitFile,
    SplitSpec,
    assert_no_row_overlap,
    generate,
    group_values,
    resolve,
    write_splits,
)


@pytest.fixture
def store(tmp_path: Path) -> SignalStore:
    """Nine groups, two recordings each, with an uneven event count."""
    path = tmp_path / "cohort"
    rng = np.random.default_rng(0)
    with SignalStore.builder(path, channels=3) as builder:
        for group in range(9):
            for session in range(2):
                builder.add(
                    f"p{group}_s{session}",
                    rng.standard_normal((1500, 3)).astype("float32"),
                    group=f"p{group}",
                    attrs={"events": group * 10},
                )
    return SignalStore(path)


@pytest.fixture
def index(store: SignalStore):
    return build_index(store, WindowSpec(length=500, stride=200))


# --- the check that matters most ----------------------------------------------------


def test_overlapping_parts_are_rejected() -> None:
    """Validating duplicates *within* a part is the obvious check and the insufficient one.

    A group in both train and test passes it and silently invalidates every number the
    split produces.
    """
    with pytest.raises(ValueError, match="mutually disjoint"):
        SplitFile(
            store="s", name="bad", parts={"train": ["p1", "p2"], "test": ["p2", "p3"]}
        )


def test_duplicates_within_a_part_are_rejected() -> None:
    with pytest.raises(ValueError, match="more than once"):
        SplitFile(store="s", name="bad", parts={"train": ["p1", "p1"]})


def test_split_with_neither_groups_nor_time_is_rejected() -> None:
    """A split must divide something; empty is not a valid partition."""
    with pytest.raises(ValueError, match="group parts, temporal bounds, or both"):
        SplitFile(store="s", name="bad", parts={})


def test_no_raw_row_appears_in_two_parts(store: SignalStore, index) -> None:
    """The structural guarantee, verified directly rather than assumed."""
    split = generate(entity_examples(store), SplitSpec(scheme="kfold", k=3), name="k3")[0]
    assert_no_row_overlap(resolve(SignalExamples(store, index), split))


def test_row_overlap_is_detectable_when_it_exists(store: SignalStore, index) -> None:
    """The detector must actually detect; a check that never fires proves nothing."""
    groups = index.groups
    left = index.subset(np.isin(groups, ["p0", "p1"]))
    overlapping = index.subset(np.isin(groups, ["p1", "p2"]))
    with pytest.raises(SplitError, match="share .* raw row"):
        assert_no_row_overlap({"a": left, "b": overlapping})


# --- generation ---------------------------------------------------------------------


def test_kfold_covers_every_group_exactly_once(store: SignalStore) -> None:
    for split in generate(entity_examples(store), SplitSpec(scheme="kfold", k=3), name="k3"):
        assert split.all_groups == set(store.groups)


def test_every_group_is_tested_exactly_once_across_folds(store: SignalStore) -> None:
    folds = generate(entity_examples(store), SplitSpec(scheme="kfold", k=3), name="k3")
    tested = [g for split in folds for g in split.parts["test"]]
    assert sorted(tested) == sorted(store.groups)


def test_stratification_balances_better_than_shuffling(store: SignalStore) -> None:
    """Serpentine assignment is the reason this is not a hash function."""
    values = group_values(entity_examples(store), "events")

    def spread(spec: SplitSpec) -> float:
        folds = generate(entity_examples(store), spec, name="x")
        totals = [sum(values[g] for g in f.parts["test"]) for f in folds]
        return float(np.std(totals))

    stratified = spread(SplitSpec(scheme="stratified_kfold", k=3, stratify_by="events"))
    shuffled = spread(SplitSpec(scheme="kfold", k=3, seed=1))
    assert stratified <= shuffled


def test_leave_one_group_out_yields_one_fold_per_group(store: SignalStore) -> None:
    """LOOCV is a list, not a hash: "leave group i out" has no hash expression."""
    folds = generate(entity_examples(store), SplitSpec(scheme="leave_one_group_out"), name="lopo")
    assert len(folds) == len(store.groups)
    assert sorted(f.parts["test"][0] for f in folds) == sorted(store.groups)
    for split in folds:
        assert len(split.parts["test"]) == 1


def test_training_is_the_largest_part_even_at_k_equals_three(store: SignalStore) -> None:
    """Validation is carved out of the training portion, never given its own fold.

    An earlier version handed val a whole rotation bucket, making train (k-2)/k. At k=3
    that is a 33% training set against 67% of test-plus-validation — smaller than what it
    is evaluated on, which is not what anyone means by "3-fold cross-validation". Nothing
    in the split file's shape reveals it; the only symptom is a model that will not learn,
    which reads as a modelling problem rather than a splitting one.
    """
    for k in (3, 4, 5):
        for split in generate(entity_examples(store), SplitSpec(scheme="kfold", k=k), name=f"k{k}"):
            train = len(split.parts["train"])
            other = sum(len(v) for p, v in split.parts.items() if p != "train")
            assert train > other, f"k={k} fold {split.fold}: train {train} vs rest {other}"


def test_validation_is_present_and_disjoint(store: SignalStore) -> None:
    for split in generate(entity_examples(store), SplitSpec(scheme="kfold", k=4), name="k4"):
        assert split.parts["val"], "a fold with no validation set cannot early-stop"
        assert not (set(split.parts["val"]) & set(split.parts["train"]))
        assert not (set(split.parts["val"]) & set(split.parts["test"]))


def test_validation_can_be_switched_off(store: SignalStore) -> None:
    """Some protocols want every non-test group training; val_fraction=0 says so."""
    for split in generate(entity_examples(store), SplitSpec(scheme="kfold", k=3,
        val_fraction=0.0), name="k3"):
        assert "val" not in split.parts
        assert len(split.parts["train"]) + len(split.parts["test"]) == len(store.groups)


def test_always_train_groups_are_pinned(store: SignalStore) -> None:
    """Groups carrying no positive events belong in train; they carry no test signal."""
    spec = SplitSpec(scheme="kfold", k=3, always_train=("p0", "p1"))
    for split in generate(entity_examples(store), spec, name="pinned"):
        assert "p0" in split.parts["train"]
        assert "p1" in split.parts["train"]
        assert "p0" not in split.parts.get("test", [])


def test_unknown_always_train_group_fails_loudly(store: SignalStore) -> None:
    with pytest.raises(SplitError, match="absent from"):
        generate(
            entity_examples(store),
            SplitSpec(scheme="kfold", k=3, always_train=("nope",)),
            name="x",
        )


def test_more_folds_than_groups_fails(store: SignalStore) -> None:
    with pytest.raises(SplitError, match="cannot make"):
        generate(entity_examples(store), SplitSpec(scheme="kfold", k=99), name="x")


def test_generation_is_deterministic(store: SignalStore) -> None:
    spec = SplitSpec(scheme="kfold", k=3, seed=7)
    assert [f.parts for f in generate(entity_examples(store), spec, name="a")] == [
        f.parts for f in generate(entity_examples(store), spec, name="a")
    ]


# --- file round trip ----------------------------------------------------------------


def test_split_file_round_trips(store: SignalStore, tmp_path: Path) -> None:
    paths = write_splits(entity_examples(store), SplitSpec(scheme="kfold", k=3), name="k3",
        root=tmp_path / "splits"
    )
    assert len(paths) == 3
    restored = SplitFile.load(paths[0])
    assert restored.parts == generate(entity_examples(store), SplitSpec(scheme="kfold", k=3),
        name="k3")[0].parts


def test_split_file_carries_a_readable_header(store: SignalStore) -> None:
    """A split is scientific provenance; it should be readable without parsing."""
    split = generate(
        entity_examples(store), SplitSpec(scheme="stratified_kfold", k=3,
            stratify_by="events"), name="k3"
    )[0]
    header = split.to_yaml()
    assert "# group key: group  <- the leakage boundary" in header
    assert "stratified by events" in header


def test_foreign_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("schema_version: other/9\nstore: s\nname: n\nparts: {train: [a]}\n")
    with pytest.raises(SplitError, match="expected"):
        SplitFile.load(path)


# --- resolution ---------------------------------------------------------------------


def test_resolve_produces_disjoint_group_sets(store: SignalStore, index) -> None:
    split = generate(entity_examples(store), SplitSpec(scheme="kfold", k=3), name="k3")[0]
    parts = resolve(SignalExamples(store, index), split)
    seen: set[str] = set()
    for sub in parts.values():
        groups = set(sub.groups.tolist())
        assert not (groups & seen)
        seen |= groups


def test_resolve_accounts_for_every_window(store: SignalStore, index) -> None:
    split = generate(entity_examples(store), SplitSpec(scheme="kfold", k=3), name="k3")[0]
    parts = resolve(SignalExamples(store, index), split)
    assert sum(len(sub) for sub in parts.values()) == len(index)


def test_resolve_rejects_a_split_from_another_store(store: SignalStore, index) -> None:
    split = generate(entity_examples(store), SplitSpec(scheme="kfold", k=3), name="k3")[0]
    foreign = split.model_copy(update={"store": "somewhere_else"})
    with pytest.raises(SplitError, match="was built for"):
        resolve(SignalExamples(store, index), foreign)


def test_resolve_rejects_a_stale_store_digest(store: SignalStore, index) -> None:
    """A split computed against different data must not silently apply to new data."""
    split = generate(entity_examples(store), SplitSpec(scheme="kfold", k=3), name="k3")[0]
    stale = split.model_copy(update={"store_manifest_sha256": "0" * 64})
    with pytest.raises(SplitError, match="regenerate the split"):
        resolve(SignalExamples(store, index), stale)


def test_resolve_rejects_unassigned_groups(store: SignalStore, index) -> None:
    """Silently dropping windows is how a fold trains on less data than it claims."""
    partial = SplitFile(
        store=store.path.name, name="partial", parts={"train": ["p0"], "test": ["p1"]}
    )
    with pytest.raises(SplitError, match="does not assign"):
        resolve(SignalExamples(store, index), partial)


def test_resolve_rejects_unknown_groups(store: SignalStore, index) -> None:
    bogus = SplitFile(
        store=store.path.name,
        name="bogus",
        parts={"train": sorted(store.groups), "test": ["ghost"]},
    )
    with pytest.raises(SplitError, match="absent from the index"):
        resolve(SignalExamples(store, index), bogus)
