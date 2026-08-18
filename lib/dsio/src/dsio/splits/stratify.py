"""Multi-key stratified assignment of groups to folds.

Balancing one label across folds is easy. Balancing a label *and* protocol *and* site *and*
device simultaneously, over a hundred subjects, is a combinatorial problem with no exact
solution — and it is the case that actually shows up in research, where you know several
metadata features matter and cannot afford to have any of them confounded with the fold.

Three things this module does that a one-key stratifier does not:

**It balances several keys at once**, categorical and numeric, with per-key weights, using
greedy assignment ordered by rarity followed by pairwise local search.

**It balances numeric keys on both distribution and mass.** A fold can hold the right
*number* of high-event subjects while holding the wrong *total* number of events. Both
matter, so both are in the objective.

**It reports what it failed to balance.** With twelve subjects and five keys, perfect
balance is impossible. Silently returning an unbalanced split is how a confound reaches a
paper; :class:`BalanceReport` states the achieved deviation per key so the split can be
judged rather than trusted.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from dsio.contracts import DsioModel

Aggregate = Literal["sum", "mean", "first", "mode", "nunique"]
Kind = Literal["categorical", "numeric"]


class StratifyKey(DsioModel):
    """One feature to balance across folds.

    ``aggregate`` reduces an entity-level attribute to a group-level value, because groups
    are what get assigned. Summing an event count across a subject's recordings is right;
    summing their age is not, which is why the reduction is explicit.
    """

    name: str
    kind: Kind = "categorical"
    weight: float = Field(default=1.0, gt=0.0)
    bins: int = Field(default=4, ge=2, description="Quantile bins for a numeric key.")
    aggregate: Aggregate | None = None
    balance_mass: bool = Field(
        default=True,
        description="For numeric keys, also balance the total, not just the distribution.",
    )

    @model_validator(mode="after")
    def _defaults(self) -> StratifyKey:
        if self.aggregate is None:
            object.__setattr__(self, "aggregate", "sum" if self.kind == "numeric" else "mode")
        return self


class KeyBalance(DsioModel):
    """Achieved balance for one key."""

    name: str
    kind: Kind
    levels: dict[str, list[int]] = Field(
        default_factory=dict, description="Per level, the count in each fold."
    )
    max_level_deviation: float = 0.0
    mass_per_fold: list[float] = Field(default_factory=list)
    max_mass_deviation: float = 0.0

    @property
    def balanced(self) -> bool:
        return self.max_level_deviation <= 0.15 and self.max_mass_deviation <= 0.15


class BalanceReport(DsioModel):
    """What the stratifier actually achieved, and what it could not."""

    keys: list[KeyBalance] = Field(default_factory=list)
    fold_sizes: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def worst(self) -> KeyBalance | None:
        if not self.keys:
            return None
        return max(
            self.keys,
            key=lambda k: max(k.max_level_deviation, k.max_mass_deviation),
        )

    def summary_lines(self) -> list[str]:
        """Human-readable lines for a split file header."""
        lines = [f"# fold sizes: {', '.join(str(n) for n in self.fold_sizes)}"]
        for key in self.keys:
            mark = "ok  " if key.balanced else "POOR"
            detail = f"levels dev {key.max_level_deviation:.0%}"
            if key.kind == "numeric" and key.mass_per_fold:
                detail += f", mass dev {key.max_mass_deviation:.0%}"
            lines.append(f"# stratify {mark} {key.name}: {detail}")
        lines.extend(f"# WARNING {w}" for w in self.warnings)
        return lines


def aggregate_to_groups(
    groups: Any, values: Any, key: StratifyKey
) -> dict[str, float | str]:
    """Reduce a per-example attribute to one value per group.

    Takes two parallel arrays rather than a dataset object, so it works for any
    :class:`~dsio.data.examples.Examples` — a table's rows, a corpus's windows, a set of
    agent episodes — without knowing which it was given.
    """
    collected: dict[str, list[Any]] = defaultdict(list)
    for group, value in zip(list(groups), list(values), strict=True):
        # Touch every group so one with no values still appears, and lands in the
        # __missing__ level rather than vanishing from the partition.
        values_for_group = collected[str(group)]
        if value is not None and value == value:
            values_for_group.append(value)

    out: dict[str, float | str] = {}
    for group, values in collected.items():
        if not values:
            out[group] = 0.0 if key.kind == "numeric" else "__missing__"
            continue
        if key.kind == "numeric":
            numbers = [float(v) for v in values]
            if key.aggregate == "mean":
                out[group] = sum(numbers) / len(numbers)
            elif key.aggregate == "first":
                out[group] = numbers[0]
            elif key.aggregate == "nunique":
                out[group] = float(len(set(numbers)))
            else:
                out[group] = sum(numbers)
        else:
            if key.aggregate == "first":
                out[group] = str(values[0])
            elif key.aggregate == "nunique":
                out[group] = str(len(set(map(str, values))))
            else:
                out[group] = str(Counter(map(str, values)).most_common(1)[0][0])
    return out


def _levels_for_key(
    values: dict[str, float | str], key: StratifyKey
) -> tuple[dict[str, str], list[str]]:
    """Map each group to a level label, binning numeric keys by quantile.

    Quantile bins rather than equal-width: an event count with a long tail puts almost every
    subject in one equal-width bin, which balances nothing.
    """
    if key.kind == "categorical":
        levels = {g: str(v) for g, v in values.items()}
        return levels, sorted(set(levels.values()))

    numbers = np.array([float(values[g]) for g in values], dtype=float)
    distinct = np.unique(numbers)
    n_bins = int(min(key.bins, max(1, distinct.size)))
    if n_bins <= 1:
        levels = dict.fromkeys(values, "all")
        return levels, ["all"]

    edges = np.unique(np.quantile(numbers, np.linspace(0, 1, n_bins + 1)[1:-1]))
    assigned = {
        g: f"q{int(np.searchsorted(edges, float(values[g]), side='right'))}" for g in values
    }
    return assigned, sorted(set(assigned.values()))


def _cost(
    buckets: list[list[str]],
    level_maps: dict[str, dict[str, str]],
    mass_maps: dict[str, dict[str, float]],
    keys: list[StratifyKey],
    n_groups: int,
) -> float:
    """Weighted imbalance across every key. Lower is better."""
    k = len(buckets)
    total = 0.0

    for key in keys:
        levels = level_maps[key.name]
        counts = Counter(levels.values())
        for level, n in counts.items():
            target = n / k
            if target <= 0:
                continue
            for bucket in buckets:
                actual = sum(1 for g in bucket if levels[g] == level)
                total += key.weight * ((actual - target) ** 2) / max(target, 1e-9)

        if key.kind == "numeric" and key.balance_mass:
            mass = mass_maps[key.name]
            grand = sum(mass.values())
            if grand > 0:
                target_mass = grand / k
                for bucket in buckets:
                    actual_mass = sum(mass[g] for g in bucket)
                    total += (
                        key.weight * ((actual_mass - target_mass) ** 2)
                        / (target_mass**2)
                    )

    # Keep folds similar in size; without this the optimiser will happily empty a fold.
    size_target = n_groups / k
    for bucket in buckets:
        total += ((len(bucket) - size_target) ** 2) / max(size_target, 1e-9)
    return total


def stratified_partition(
    groups: list[str],
    examples: Any,
    keys: list[StratifyKey],
    k: int,
    *,
    seed: int = 42,
    refine_rounds: int = 200,
) -> tuple[list[list[str]], BalanceReport]:
    """Partition ``groups`` into ``k`` folds balanced across every key.

    Greedy assignment ordered by rarity — the most constrained groups placed first, while
    there is still freedom to place them — followed by deterministic pairwise local search.
    Exact multi-key balance is NP-hard; this gets close and then reports how close.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not groups:
        raise ValueError("no groups to partition")

    level_maps: dict[str, dict[str, str]] = {}
    mass_maps: dict[str, dict[str, float]] = {}
    warnings: list[str] = []

    for key in keys:
        values = aggregate_to_groups(examples.groups, examples.attribute(key.name), key)
        missing = [g for g in groups if g not in values]
        for g in missing:
            values[g] = 0.0 if key.kind == "numeric" else "__missing__"
        levels, labels = _levels_for_key({g: values[g] for g in groups}, key)
        level_maps[key.name] = levels
        mass_maps[key.name] = {
            g: float(values[g]) if key.kind == "numeric" else 0.0 for g in groups
        }

        counts = Counter(levels.values())
        rare = [lv for lv, n in counts.items() if n < k]
        if rare:
            warnings.append(
                f"{key.name}: level(s) {', '.join(sorted(rare)[:3])} have fewer than {k} "
                f"groups, so they cannot appear in every fold"
            )
        if len(labels) == 1:
            warnings.append(f"{key.name}: only one level present, nothing to balance")

    # Rarity ordering: a group carrying a rare level has the fewest viable placements, so
    # it goes first. Ties broken by name for determinism rather than by hash order.
    def rarity(group: str) -> tuple[int, str]:
        scores = [
            Counter(level_maps[key.name].values())[level_maps[key.name][group]]
            for key in keys
        ]
        return (min(scores) if scores else 0, group)

    ordered = sorted(groups, key=rarity)
    if not keys:
        rng = np.random.default_rng(seed)
        shuffled = list(groups)
        rng.shuffle(shuffled)
        ordered = sorted(shuffled, key=lambda g: shuffled.index(g))

    buckets: list[list[str]] = [[] for _ in range(k)]
    for group in ordered:
        best_i, best_cost = 0, math.inf
        for i in range(k):
            buckets[i].append(group)
            cost = _cost(buckets, level_maps, mass_maps, keys, len(groups))
            buckets[i].pop()
            if cost < best_cost:
                best_i, best_cost = i, cost
        buckets[best_i].append(group)

    if k > 1 and keys:
        buckets = _refine(buckets, level_maps, mass_maps, keys, len(groups), refine_rounds)

    report = _report(buckets, level_maps, mass_maps, keys, warnings)
    return [sorted(b) for b in buckets], report


def _refine(
    buckets: list[list[str]],
    level_maps: dict[str, dict[str, str]],
    mass_maps: dict[str, dict[str, float]],
    keys: list[StratifyKey],
    n_groups: int,
    rounds: int,
) -> list[list[str]]:
    """First-improvement pairwise swaps until no swap helps.

    Every trial swap is reverted before the next is considered, and the chosen swap is
    applied once, at the end of the round. An earlier version kept an improving swap and
    carried on iterating with a now-stale loop variable, which wrote the old value back and
    silently duplicated groups across folds — a partition that was no longer a partition.
    Trial-then-revert makes that class of bug unrepresentable.
    """
    current = _cost(buckets, level_maps, mass_maps, keys, n_groups)

    for _ in range(rounds):
        chosen: tuple[int, int, int, int, float] | None = None

        for a in range(len(buckets)):
            for b in range(a + 1, len(buckets)):
                for i in range(len(buckets[a])):
                    for j in range(len(buckets[b])):
                        buckets[a][i], buckets[b][j] = buckets[b][j], buckets[a][i]
                        candidate = _cost(buckets, level_maps, mass_maps, keys, n_groups)
                        buckets[a][i], buckets[b][j] = buckets[b][j], buckets[a][i]
                        if candidate < current - 1e-12:
                            chosen = (a, b, i, j, candidate)
                            break
                    if chosen:
                        break
                if chosen:
                    break
            if chosen:
                break

        if chosen is None:
            break
        a, b, i, j, candidate = chosen
        buckets[a][i], buckets[b][j] = buckets[b][j], buckets[a][i]
        current = candidate

    return buckets


def _report(
    buckets: list[list[str]],
    level_maps: dict[str, dict[str, str]],
    mass_maps: dict[str, dict[str, float]],
    keys: list[StratifyKey],
    warnings: list[str],
) -> BalanceReport:
    """Measure what the partition achieved, per key.

    Values are computed before construction rather than assigned after: KeyBalance is
    frozen, like every dsio model, so a report cannot drift from the numbers it was built
    from.
    """
    k = len(buckets)
    balances: list[KeyBalance] = []
    all_warnings = list(warnings)

    for key in keys:
        levels = level_maps[key.name]
        counts = Counter(levels.values())

        per_level: dict[str, list[int]] = {}
        max_level_dev = 0.0
        for level, n in sorted(counts.items()):
            per_fold = [sum(1 for g in b if levels[g] == level) for b in buckets]
            per_level[level] = per_fold
            target = n / k
            if target > 0:
                max_level_dev = max(
                    max_level_dev, max(abs(c - target) for c in per_fold) / target
                )

        mass_per_fold: list[float] = []
        max_mass_dev = 0.0
        if key.kind == "numeric" and key.balance_mass:
            mass = mass_maps[key.name]
            mass_per_fold = [float(sum(mass[g] for g in b)) for b in buckets]
            grand = sum(mass_per_fold)
            if grand > 0:
                target = grand / k
                max_mass_dev = max(abs(m - target) for m in mass_per_fold) / target

        balance = KeyBalance(
            name=key.name,
            kind=key.kind,
            levels=per_level,
            max_level_deviation=max_level_dev,
            mass_per_fold=mass_per_fold,
            max_mass_deviation=max_mass_dev,
        )
        balances.append(balance)
        if not balance.balanced:
            all_warnings.append(
                f"{key.name}: could not be balanced within 15% "
                f"(levels {max_level_dev:.0%}, mass {max_mass_dev:.0%}); "
                "too few groups, too many keys, or a genuinely skewed feature"
            )

    return BalanceReport(
        keys=balances,
        fold_sizes=[len(b) for b in buckets],
        warnings=all_warnings,
    )
