"""Derivation: which *result* this is, as distinct from which corpus it came from.

``digest`` answers "what data is this?" and is deliberately corpus-level, so that a split
survives re-indexing at a different window length (adapters.py, ``SignalExamples.digest``).
That is the right answer for a *view*, and the wrong one for a *subset*: filtering to a
subpopulation, computing a split on the remainder and committing it produces a split whose
digest still matches the full corpus, so resolving it against the whole thing is accepted
and silently scores a split computed on a different population.

One field cannot answer both questions. ``derivation`` answers the second.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsio.data.adapters import TableExamples
from dsio.data.examples import ExamplesError, check
from dsio.splits.models import SplitError, SplitFile, SplitFold
from dsio.splits.resolve import resolve_masks


@pytest.fixture
def table() -> TableExamples:
    return TableExamples(
        name="rows",
        groups=[f"g{i // 3}" for i in range(12)],
        attributes={"score": np.arange(12, dtype=float)},
    )


# --- the two questions are answered separately ------------------------------------------


def test_a_whole_corpus_is_a_root_derivation(table: TableExamples) -> None:
    assert table.derivation == "root"


def test_a_subset_is_not_the_corpus_it_came_from(table: TableExamples) -> None:
    """The hole this closes: subset() preserves digest (correctly), so digest alone
    cannot tell a subpopulation apart from the corpus."""
    part = table.subset(np.array([i < 6 for i in range(12)]))
    assert part.digest == table.digest, "corpus identity is unchanged by subsetting"
    assert part.derivation != table.derivation


def test_different_masks_derive_differently(table: TableExamples) -> None:
    first = table.subset(np.array([i < 6 for i in range(12)]))
    second = table.subset(np.array([i >= 6 for i in range(12)]))
    assert first.derivation != second.derivation


def test_the_same_mask_derives_identically(table: TableExamples) -> None:
    """Deterministic, or a split committed in one session cannot be resolved in the next."""
    mask = np.array([i % 2 == 0 for i in range(12)])
    assert table.subset(mask).derivation == table.subset(mask).derivation


def test_derivation_composes_through_repeated_subsetting(table: TableExamples) -> None:
    """Two paths to the same six rows are two different derivations, because the
    population each was computed against differs."""
    direct = table.subset(np.array([i < 6 for i in range(12)]))
    stepwise = table.subset(np.array([i < 9 for i in range(12)])).subset(
        np.array([i < 6 for i in range(9)])
    )
    assert direct.groups.tolist() == stepwise.groups.tolist()
    assert direct.derivation != stepwise.derivation


def test_the_protocol_requires_a_derivation() -> None:
    class Incomplete:
        name = "x"
        digest = "abc"
        groups = np.array(["a"])

        def __len__(self) -> int:
            return 1

        def attribute_names(self) -> tuple[str, ...]:
            return ()

        def attribute(self, name: str) -> np.ndarray:
            raise ExamplesError(name)

        def times(self) -> None:
            return None

        def subset(self, mask: np.ndarray) -> Incomplete:
            return self

    with pytest.raises(ExamplesError, match="derivation"):
        check(Incomplete())


# --- and the split layer refuses on it --------------------------------------------------


def _split(table: TableExamples, *, derivation: str | None) -> SplitFile:
    return SplitFile(
        store=table.name,
        store_manifest_sha256=table.digest,
        examples_derivation=derivation,
        name="k1",
        folds=[SplitFold(index=0, parts={"train": ["g2", "g3"], "test": ["g0", "g1"]})],
    )


def test_a_split_computed_on_a_subset_is_refused_against_the_corpus(
    table: TableExamples,
) -> None:
    """The headline. Digest matches — it is the same corpus — and it must still refuse."""
    part = table.subset(np.array([i < 9 for i in range(12)]))
    split = _split(table, derivation=part.derivation)

    assert split.store_manifest_sha256 == table.digest
    with pytest.raises(SplitError, match="derivation"):
        resolve_masks(table, split, split.fold(0))


def test_a_split_resolves_against_the_examples_it_was_computed_on(
    table: TableExamples,
) -> None:
    split = _split(table, derivation=table.derivation)
    masks = resolve_masks(table, split, split.fold(0))
    assert set(masks) == {"train", "test"}


def test_a_split_that_names_no_derivation_still_resolves(table: TableExamples) -> None:
    """Split files committed before this field existed must keep working."""
    split = _split(table, derivation=None)
    assert set(resolve_masks(table, split, split.fold(0))) == {"train", "test"}


def test_a_mask_that_drops_nothing_is_the_same_population(table: TableExamples) -> None:
    """Derivation identifies the *population*, not the number of calls made to reach it.

    A no-op subset -- an all-True mask, which `resolve` produces whenever a part covers
    every group -- must not move off the parent's derivation, or a split computed on the
    parent would be refused against rows identical to it.
    """
    whole = table.subset(np.ones(len(table), dtype=bool))
    assert whole.groups.tolist() == table.groups.tolist()
    assert whole.derivation == table.derivation
