"""Every structural obligation an :class:`~dsio.data.examples.Examples` carries.

``check()`` verifies the members exist and ``assert_consistent()`` verifies the parallel
arrays line up. Neither can see the obligations that only show up under ``subset``, which
are the ones that go wrong quietly:

* a subset that **re-derives its digest** binds splits to a population rather than to a
  corpus, so every committed split stops resolving the first time anyone filters;
* a subset that **keeps its parent's derivation** does the reverse — a split computed on a
  filtered subpopulation resolves against the whole corpus and silently scores a
  population it was never computed for.

The two errors are opposite and both are invisible: nothing raises, the numbers simply
describe different data than the split file claims.
"""

from __future__ import annotations

import numpy as np

from dsio.data.examples import Examples, ExamplesError, assert_consistent, check


class ContractViolation(AssertionError):
    """Raised when an implementation breaks an obligation dsio depends on."""


def check_examples_contract(examples: Examples) -> None:
    """Assert ``examples`` satisfies every obligation the split layer relies on.

    Call it from your own test suite with a small, real instance of your adapter::

        def test_my_adapter_is_a_dsio_dataset():
            check_examples_contract(MyExamples(...))
    """
    try:
        check(examples)
        assert_consistent(examples)
    except ExamplesError as exc:
        raise ContractViolation(str(exc)) from exc

    size = len(examples)
    if size < 2:
        raise ContractViolation(
            f"{examples.name}: the contract needs at least 2 examples to exercise "
            f"subset(), got {size}"
        )
    if not isinstance(examples.derivation, str) or not examples.derivation:
        raise ContractViolation(f"{examples.name}: derivation must be a non-empty str")

    _check_dropping_subset(examples, size)
    _check_no_op_subset(examples, size)


def _check_dropping_subset(examples: Examples, size: int) -> None:
    """A mask that actually removes rows: the case every obligation is about."""
    mask = np.zeros(size, dtype=bool)
    mask[: max(size // 2, 1)] = True
    part = examples.subset(mask)

    expected = int(mask.sum())
    if len(part) != expected:
        raise ContractViolation(
            f"{examples.name}: subset() of a {expected}-row mask has length {len(part)}; "
            "a subset must contain exactly the masked rows"
        )
    if part.digest != examples.digest:
        raise ContractViolation(
            f"{examples.name}: subset() changed the digest from {examples.digest} to "
            f"{part.digest}. A digest names the corpus, and filtering does not produce a "
            "different corpus — re-deriving it stops every committed split resolving. "
            "Pass the parent's digest through."
        )
    if part.derivation == examples.derivation:
        raise ContractViolation(
            f"{examples.name}: subset() kept the derivation {examples.derivation!r} after "
            "dropping rows. Since the digest is (correctly) unchanged, nothing then "
            "distinguishes this subpopulation from the whole corpus, and a split computed "
            "on it will resolve against the corpus. Use dsio.data.examples.derive()."
        )

    try:
        assert_consistent(part)
    except ExamplesError as exc:
        raise ContractViolation(f"{examples.name}: subset() is inconsistent: {exc}") from exc

    if part.attribute_names() != examples.attribute_names():
        raise ContractViolation(
            f"{examples.name}: subset() changed the attributes from "
            f"{examples.attribute_names()} to {part.attribute_names()}; stratification "
            "keys must survive a subset or a fold cannot be balanced the same way twice"
        )


def _check_no_op_subset(examples: Examples, size: int) -> None:
    """An all-True mask is the same population, and must derive as the same population.

    ``resolve`` produces one whenever a part covers every group, so an implementation that
    treats it as a new derivation refuses splits against rows identical to the ones they
    were computed on.
    """
    whole = examples.subset(np.ones(size, dtype=bool))
    if whole.derivation != examples.derivation:
        raise ContractViolation(
            f"{examples.name}: subset() of an all-True mask moved the derivation from "
            f"{examples.derivation!r} to {whole.derivation!r}. It drops no rows, so it is "
            "the same population; deriving it as a new one refuses valid splits."
        )
