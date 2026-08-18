"""Label-budget selection: arm-invariant, stratified, and nested."""

from __future__ import annotations

import numpy as np
import pytest

from dsio.ssl.budget import (
    BudgetError,
    assert_nested,
    budget_curve,
    group_rates,
    select_groups,
)


@pytest.fixture
def cohort() -> dict[str, float]:
    """A realistic rate distribution: a long tail of near-zero groups.

    About 16% of DeFOG groups have almost no FOG. That tail is what makes a uniform draw
    dangerous, so it is what the fixture reproduces.
    """
    rng = np.random.default_rng(0)
    rates = {f"p{i:02d}": 0.0 for i in range(8)}
    rates.update({f"p{i:02d}": float(rng.uniform(0.05, 0.6)) for i in range(8, 40)})
    return rates


# --- arm invariance ---------------------------------------------------------------------


def test_the_same_seed_and_budget_select_the_same_groups(cohort: dict[str, float]) -> None:
    """The property that makes a budget curve a fair comparison.

    Probe, random-init and scratch arms must train on identical groups at a given budget;
    otherwise the gap between two arms partly measures which groups each of them got.
    """
    first = select_groups(cohort, 8, seed=7)
    second = select_groups(cohort, 8, seed=7)
    assert first.groups == second.groups


def test_a_different_seed_selects_a_different_subset(cohort: dict[str, float]) -> None:
    """Guards against a selection that ignores its seed, which would make every "repeat"
    of a budget curve an exact copy and its error bars meaningless."""
    assert select_groups(cohort, 8, seed=1).groups != select_groups(cohort, 8, seed=2).groups


def test_selection_is_stable_across_processes(cohort: dict[str, float]) -> None:
    """Names are sorted and the rate sort is stable, so nothing depends on dict order."""
    shuffled = dict(reversed(list(cohort.items())))
    assert select_groups(cohort, 6, seed=3).groups == select_groups(shuffled, 6, seed=3).groups


# --- stratification -----------------------------------------------------------------------


def test_a_small_budget_is_not_positive_starved(cohort: dict[str, float]) -> None:
    """The failure a uniform draw produces routinely on this distribution.

    With 8 of 40 groups at zero rate, a uniform draw of 4 lands on all-zero often enough to
    matter — and the resulting budget curve measures the luck of the draw rather than the
    value of a label.
    """
    selection = select_groups(cohort, 4, seed=0)
    assert selection.max_rate > 0.0
    assert selection.selected_rate > 0.0


def test_stratified_selection_beats_uniform_on_rate_fidelity(cohort: dict[str, float]) -> None:
    """Measured rather than asserted: across many seeds, the stratified draw's base rate
    sits closer to the cohort's than a uniform draw's."""
    names = sorted(cohort)
    stratified: list[float] = []
    uniform: list[float] = []
    for seed in range(40):
        stratified.append(abs(select_groups(cohort, 6, seed=seed).rate_shift))
        rng = np.random.default_rng(seed)
        drawn = rng.choice(names, size=6, replace=False)
        uniform.append(
            abs(float(np.mean([cohort[g] for g in drawn])) - float(np.mean(list(cohort.values()))))
        )
    assert np.mean(stratified) < np.mean(uniform)


def test_selection_spans_the_rate_range(cohort: dict[str, float]) -> None:
    selection = select_groups(cohort, 10, seed=0)
    assert selection.min_rate == pytest.approx(0.0, abs=1e-9)
    assert selection.max_rate > 0.4


def test_the_realised_rate_is_recorded_not_assumed(cohort: dict[str, float]) -> None:
    """A budget curve built on a subset whose base rate drifted is measuring two things."""
    selection = select_groups(cohort, 5, seed=0)
    assert selection.cohort_rate == pytest.approx(float(np.mean(list(cohort.values()))))
    assert abs(selection.rate_shift) < 0.2


# --- nesting -------------------------------------------------------------------------------


def test_budgets_are_nested_by_default(cohort: dict[str, float]) -> None:
    """Without nesting, a dip in the curve is ambiguous: it could be the budget, or it
    could be that a different and harder set of groups was drawn. Resolving that
    ambiguity costs a full re-run at every budget with several seeds.
    """
    selections = budget_curve(cohort, [2, 4, 8, 16], seed=0)
    assert_nested(selections)
    assert [s.budget for s in selections] == [2, 4, 8, 16]


def test_nesting_can_be_switched_off(cohort: dict[str, float]) -> None:
    """The unnested draw is a valid, different experiment."""
    selections = budget_curve(cohort, [2, 4, 8, 16], seed=0, nested=False)
    with pytest.raises(BudgetError, match="drops"):
        assert_nested(selections)


def test_assert_nested_accepts_a_genuinely_nested_curve() -> None:
    rates = {f"g{i}": float(i) / 10 for i in range(10)}
    assert_nested(budget_curve(rates, [3, 5, 9], seed=0))


def test_nested_prefixes_still_span_the_rate_range(cohort: dict[str, float]) -> None:
    """Nesting must not cost stratification: every prefix has to stay spread out, or the
    smallest budgets quietly become the least representative."""
    for selection in budget_curve(cohort, [3, 6, 12], seed=0):
        assert selection.max_rate > 0.3, f"budget {selection.budget} missed the high tail"


# --- edges ----------------------------------------------------------------------------------


def test_a_budget_at_or_above_the_cohort_returns_everything(cohort: dict[str, float]) -> None:
    assert len(select_groups(cohort, 40, seed=0).groups) == 40
    assert len(select_groups(cohort, 99, seed=0).groups) == 40


def test_a_budget_selects_exactly_that_many_groups(cohort: dict[str, float]) -> None:
    for budget in (1, 3, 7, 21):
        assert len(select_groups(cohort, budget, seed=0).groups) == budget


def test_degenerate_budgets_are_rejected(cohort: dict[str, float]) -> None:
    with pytest.raises(BudgetError, match="at least 1"):
        select_groups(cohort, 0)
    with pytest.raises(BudgetError, match="no groups"):
        select_groups({}, 1)
    with pytest.raises(BudgetError, match="at least one budget"):
        budget_curve(cohort, [])


# --- rates from windows ----------------------------------------------------------------------


def test_group_rates_are_computed_over_the_windows_each_group_owns() -> None:
    groups = np.array(["a", "a", "a", "b", "b"])
    labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    rates = group_rates(["a", "b"], labels, groups)
    assert rates["a"] == pytest.approx(2 / 3)
    assert rates["b"] == pytest.approx(0.0)


def test_a_group_with_no_windows_rates_zero() -> None:
    rates = group_rates(["a", "ghost"], np.array([1.0]), np.array(["a"]))
    assert rates["ghost"] == 0.0


def test_soft_labels_are_binarised_for_the_rate() -> None:
    """The budget is about how many positives a group carries, not their average ratio."""
    groups = np.array(["a", "a"])
    assert group_rates(["a"], np.array([0.6, 0.4]), groups)["a"] == pytest.approx(0.5)
