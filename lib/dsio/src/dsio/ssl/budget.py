"""Label-budget curves with arm-invariant subsets.

The question SSL exists to answer is not "is the pretrained model better" but "how much
labelled data does it save". Answering it means training the same downstream task at several
budgets, for each arm — pretrained, random-init, scratch — and comparing the curves.

Two things a naive implementation gets wrong:

**The budget is a number of groups, not a number of windows.** Sampling windows would draw
from all subjects at every budget, so "10% of labels" would still mean *every* subject was
labelled. Real labelling effort is per-subject, and a model that has seen a little of
everyone is in a much easier position than one that has seen a few people completely.

**The draw is stratified over the positive rate, not uniform.** Rare-event datasets
usually have a long tail of groups with almost no positives, so a small uniform draw is
routinely positive-starved — and the resulting budget curve measures the luck of the draw
rather than the value of a label. Groups are ranked by rate and drawn so the subset spans the
full range: K contiguous strata when ``nested=False``, and a breadth-first bisection of the
ranking when nested, which gives the same spread while making every budget a prefix of the
next.

**Every arm gets the same groups at the same budget.** That is what makes the comparison
fair; otherwise the gap between two arms partly measures which subjects each of them got.

One change from the original: budgets are **nested** by default, so the groups at K=4 are a
subset of those at K=8. Without nesting, a curve that dips at one budget is ambiguous — it
could be the budget or it could be that a different, harder set of subjects was drawn — and
resolving that ambiguity costs a full re-run at every budget with several seeds.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from dsio.contracts import DsioModel


class BudgetError(ValueError):
    """Raised when a budget cannot be honoured by the groups available."""


class BudgetSelection(DsioModel):
    """Which groups a budget selected, and what it did to the positive rate.

    The realised rate is recorded rather than assumed. A stratified draw keeps the rate
    close to the cohort's, but "close" is a claim, and a budget curve built on a subset
    whose base rate drifted is measuring two things at once.
    """

    budget: int
    seed: int
    groups: tuple[str, ...]
    cohort_rate: float
    selected_rate: float
    min_rate: float
    max_rate: float

    @property
    def rate_shift(self) -> float:
        """How far the subset's base rate moved from the cohort's."""
        return self.selected_rate - self.cohort_rate


def group_rates(
    groups: Sequence[str], labels: np.ndarray, window_groups: np.ndarray
) -> dict[str, float]:
    """Per-group positive rate, over whatever windows each group owns."""
    hard = (np.asarray(labels) > 0.5).astype(np.float64)
    rates: dict[str, float] = {}
    for group in groups:
        mask = window_groups == group
        rates[group] = float(hard[mask].mean()) if mask.any() else 0.0
    return rates


def select_groups(
    rates: dict[str, float],
    budget: int,
    *,
    seed: int = 42,
    nested: bool = True,
) -> BudgetSelection:
    """Choose ``budget`` groups, stratified over the positive rate and seeded.

    Deterministic in ``seed``, so every arm at a given (budget, seed) trains on exactly the
    same groups. With ``nested``, a larger budget is a superset of a smaller one.
    """
    if budget < 1:
        raise BudgetError(f"budget must be at least 1, got {budget}")
    if not rates:
        raise BudgetError("no groups to select from")

    names = np.array(sorted(rates))
    values = np.array([rates[name] for name in names], dtype=np.float64)
    if budget >= names.size:
        return _describe(names.tolist(), rates, budget, seed)

    # Stable sort so equal rates keep a deterministic order across processes.
    order = names[np.argsort(values, kind="stable")]
    chosen = _nested_order(order, seed)[:budget] if nested else _stratified(order, budget, seed)
    return _describe(sorted(chosen), rates, budget, seed)


def _stratified(order: np.ndarray, budget: int, seed: int) -> list[str]:
    """Split the rate-ordered groups into K strata and take one from each."""
    rng = np.random.default_rng(seed)
    strata = np.array_split(np.arange(order.size), budget)
    return [str(order[stratum[rng.integers(len(stratum))]]) for stratum in strata]


def _nested_order(order: np.ndarray, seed: int) -> list[str]:
    """A single priority ordering over the rate ranking that every budget takes a prefix of.

    Built by breadth-first bisection: take one group from the middle of the rate range,
    then one from the middle of each half, then each quarter, and so on. Every prefix
    therefore spans the full range — which is what makes a prefix a valid stratified
    subset, and being a prefix is what makes the budgets nested.

    The pick within each range is drawn from its middle third rather than its exact
    midpoint, so the seed produces genuinely different subsets without letting a draw
    collapse toward one end. An exact midpoint would be deterministic, and a uniform draw
    over the whole range would reintroduce the clumping this exists to avoid.
    """
    from collections import deque

    rng = np.random.default_rng(seed)
    ranked: list[str] = []
    queue: deque[tuple[int, int]] = deque([(0, order.size - 1)])
    while queue:
        low, high = queue.popleft()
        if low > high:
            continue
        span = high - low
        start = low + span // 3
        stop = high - span // 3
        pick = int(rng.integers(start, stop + 1)) if stop > start else low + span // 2
        ranked.append(str(order[pick]))
        queue.append((low, pick - 1))
        queue.append((pick + 1, high))
    return ranked


def _describe(
    chosen: list[str], rates: dict[str, float], budget: int, seed: int
) -> BudgetSelection:
    selected = np.array([rates[name] for name in chosen], dtype=np.float64)
    cohort = np.array(list(rates.values()), dtype=np.float64)
    return BudgetSelection(
        budget=budget,
        seed=seed,
        groups=tuple(chosen),
        cohort_rate=float(cohort.mean()),
        selected_rate=float(selected.mean()),
        min_rate=float(selected.min()),
        max_rate=float(selected.max()),
    )


def budget_curve(
    rates: dict[str, float],
    budgets: Sequence[int],
    *,
    seed: int = 42,
    nested: bool = True,
) -> list[BudgetSelection]:
    """One selection per budget, sorted ascending so nesting is visible in the output."""
    if not budgets:
        raise BudgetError("a budget curve needs at least one budget")
    return [
        select_groups(rates, budget, seed=seed, nested=nested)
        for budget in sorted(set(budgets))
    ]


def assert_nested(selections: Sequence[BudgetSelection]) -> None:
    """Verify each budget's groups contain the previous budget's.

    Worth checking rather than trusting: a non-nested curve is still a valid experiment,
    but it is a *different* experiment, and the difference is invisible in the plotted
    result — which is precisely when a silent regression does the most damage.
    """
    for smaller, larger in pairwise(selections):
        missing = set(smaller.groups) - set(larger.groups)
        if missing:
            raise BudgetError(
                f"budget {larger.budget} drops {len(missing)} group(s) present at budget "
                f"{smaller.budget}: {', '.join(sorted(missing)[:5])}"
            )
