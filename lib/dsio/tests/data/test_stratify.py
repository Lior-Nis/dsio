"""Multi-key stratification invariants."""

from __future__ import annotations

import numpy as np
import pytest

from dsio.data import TableExamples
from dsio.splits import SplitSpec, generate
from dsio.splits.stratify import StratifyKey, stratified_partition


def build(records: list[dict]) -> TableExamples:
    """One example per record, with the record's fields as attributes.

    No store, no files, no modality: stratification balances groups by their attributes and
    does not care what an example physically is.
    """
    return TableExamples(
        name="cohort",
        groups=[f"g{i:02d}" for i in range(len(records))],
        attributes={
            key: [record[key] for record in records] for key in records[0]
        },
    )


@pytest.fixture
def cohort() -> TableExamples:
    """24 groups: two sites, three devices, a skewed event count."""
    return build(
        [
            {
                "site": "A" if i % 2 == 0 else "B",
                "device": ["x", "y", "z"][i % 3],
                "events": float(i * i),
                "sex": "F" if i % 4 < 2 else "M",
            }
            for i in range(24)
        ]
    )


# --- the invariant the local-search bug violated -------------------------------------


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_result_is_always_a_partition(cohort: TableExamples, k: int) -> None:
    """Every group exactly once, in exactly one fold.

    An earlier local search kept an improving swap and carried on with a stale loop
    variable, writing the old value back and duplicating groups across folds. The result
    still looked like a list of lists.
    """
    keys = [StratifyKey(name="events", kind="numeric"), StratifyKey(name="site")]
    buckets, _ = stratified_partition(sorted(cohort.groups.tolist()), cohort, keys, k)
    flat = [g for bucket in buckets for g in bucket]
    assert sorted(flat) == sorted(cohort.groups)
    assert len(flat) == len(set(flat))


def test_partition_holds_with_many_keys(cohort: TableExamples) -> None:
    keys = [
        StratifyKey(name="events", kind="numeric"),
        StratifyKey(name="site"),
        StratifyKey(name="device"),
        StratifyKey(name="sex"),
    ]
    buckets, _ = stratified_partition(sorted(cohort.groups.tolist()), cohort, keys, 3)
    flat = [g for bucket in buckets for g in bucket]
    assert sorted(flat) == sorted(cohort.groups)


# --- balance ------------------------------------------------------------------------


def test_balances_a_categorical_key(cohort: TableExamples) -> None:
    buckets, report = stratified_partition(
        sorted(cohort.groups.tolist()), cohort, [StratifyKey(name="site")], 3
    )
    site = dict(zip(cohort.groups.tolist(), cohort.attribute("site").tolist(), strict=True))
    per_fold = [sum(1 for g in b if site[g] == "A") for b in buckets]
    assert max(per_fold) - min(per_fold) <= 1
    assert report.keys[0].balanced


def test_balances_numeric_mass_not_only_levels(cohort: TableExamples) -> None:
    """A fold can hold the right count of high-event groups and the wrong total."""
    buckets, report = stratified_partition(
        sorted(cohort.groups.tolist()), cohort, [StratifyKey(name="events", kind="numeric")], 3
    )
    events = dict(zip(cohort.groups.tolist(), cohort.attribute("events").tolist(), strict=True))
    totals = [sum(events[g] for g in b) for b in buckets]
    assert (max(totals) - min(totals)) / np.mean(totals) < 0.15
    assert report.keys[0].max_mass_deviation < 0.15


def test_secondary_key_improves_when_included(cohort: TableExamples) -> None:
    """The whole point: adding a key measurably improves that key's balance."""
    device = dict(zip(cohort.groups.tolist(), cohort.attribute("device").tolist(), strict=True))

    def imbalance(keys: list[StratifyKey]) -> float:
        buckets, _ = stratified_partition(sorted(cohort.groups.tolist()), cohort, keys, 3)
        counts = [sum(1 for g in b if device[g] == "x") for b in buckets]
        return float(max(counts) - min(counts))

    only_events = imbalance([StratifyKey(name="events", kind="numeric")])
    with_device = imbalance(
        [StratifyKey(name="events", kind="numeric"), StratifyKey(name="device")]
    )
    assert with_device <= only_events


def test_weight_shifts_the_tradeoff(cohort: TableExamples) -> None:
    """When keys conflict, the heavier one should win."""
    site = dict(zip(cohort.groups.tolist(), cohort.attribute("site").tolist(), strict=True))

    def site_spread(weight: float) -> float:
        keys = [
            StratifyKey(name="events", kind="numeric", weight=1.0),
            StratifyKey(name="site", weight=weight),
        ]
        buckets, _ = stratified_partition(sorted(cohort.groups.tolist()), cohort, keys, 3)
        counts = [sum(1 for g in b if site[g] == "A") for b in buckets]
        return float(max(counts) - min(counts))

    assert site_spread(50.0) <= site_spread(0.1)


# --- honest reporting ---------------------------------------------------------------


def test_report_flags_a_key_it_cannot_balance() -> None:
    """Silently returning an unbalanced split is how a confound reaches a paper."""
    groups = [{"rare": "only_one" if i == 0 else "common"} for i in range(9)]
    store = build(groups)
    _, report = stratified_partition(
        sorted(store.groups.tolist()), store, [StratifyKey(name="rare")], 3
    )
    assert report.warnings
    assert any("fewer than 3 groups" in w for w in report.warnings)
    assert not report.keys[0].balanced


def test_report_flags_a_constant_key() -> None:
    store = build([{"same": "x"} for _ in range(9)])
    _, report = stratified_partition(
        sorted(store.groups.tolist()), store, [StratifyKey(name="same")], 3
    )
    assert any("only one level" in w for w in report.warnings)


def test_report_records_every_key(cohort: TableExamples) -> None:
    keys = [
        StratifyKey(name="events", kind="numeric"),
        StratifyKey(name="site"),
        StratifyKey(name="device"),
    ]
    _, report = stratified_partition(sorted(cohort.groups.tolist()), cohort, keys, 3)
    assert [k.name for k in report.keys] == ["events", "site", "device"]
    assert report.worst is not None


def test_missing_attribute_does_not_crash() -> None:
    """Real metadata has holes; a missing value must become a level, not an exception."""
    groups = [{"site": "A"} if i % 2 else {"site": None} for i in range(9)]
    store = build(groups)
    buckets, report = stratified_partition(
        sorted(store.groups.tolist()), store, [StratifyKey(name="site")], 3
    )
    assert sorted(g for b in buckets for g in b) == sorted(store.groups)
    assert "__missing__" in report.keys[0].levels


# --- determinism and integration ----------------------------------------------------


def test_stratification_is_deterministic(cohort: TableExamples) -> None:
    keys = [StratifyKey(name="events", kind="numeric"), StratifyKey(name="site")]
    first, _ = stratified_partition(sorted(cohort.groups.tolist()), cohort, keys, 3)
    second, _ = stratified_partition(sorted(cohort.groups.tolist()), cohort, keys, 3)
    assert first == second


def test_spec_expands_the_shorthand(cohort: TableExamples) -> None:
    """One numeric key is the common case, but must not become a second code path."""
    spec = SplitSpec(scheme="stratified_kfold", k=3, stratify_by="events")
    assert [k.name for k in spec.keys] == ["events"]
    assert spec.keys[0].kind == "numeric"


def test_spec_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="more than once"):
        SplitSpec(
            scheme="stratified_kfold",
            k=3,
            stratify=(StratifyKey(name="a"), StratifyKey(name="a")),
        )


def test_generated_split_carries_its_balance(cohort: TableExamples) -> None:
    """A reviewer should see what stratification achieved without rerunning anything."""
    spec = SplitSpec(
        scheme="stratified_kfold",
        k=3,
        stratify=(
            StratifyKey(name="events", kind="numeric"),
            StratifyKey(name="site"),
        ),
    )
    split = generate(cohort, spec, name="multi")[0]
    assert split.balance is not None
    header = split.to_yaml()
    assert "# stratify" in header
    assert "events" in header and "site" in header
