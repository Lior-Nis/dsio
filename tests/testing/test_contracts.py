"""The shipped contract suites must actually catch the bugs they exist to catch.

A contract helper that passes everything is worse than no helper: it converts "I did not
check" into "I checked and it was fine". So each of these feeds the suite a deliberately
broken implementation and asserts it fails -- the same reason `check()` has its own test.
"""

from __future__ import annotations

import functools
import os
import time
from pathlib import Path

import numpy as np
import pytest

from dsio.data.adapters import TableExamples
from dsio.data.examples import ROOT_DERIVATION
from dsio.data.readers import MmapReader
from dsio.testing import ContractViolation, check_examples_contract, check_reader_contract


@pytest.fixture
def table() -> TableExamples:
    return TableExamples(
        name="rows",
        groups=[f"g{i // 3}" for i in range(12)],
        attributes={"score": np.arange(12, dtype=float)},
    )


# --- the Examples contract ---------------------------------------------------------------


def test_a_real_adapter_satisfies_the_contract(table: TableExamples) -> None:
    check_examples_contract(table)


def test_the_contract_catches_misaligned_parallel_arrays() -> None:
    broken = TableExamples(
        name="bad", groups=["a", "b"], attributes={"x": np.arange(5, dtype=float)}
    )
    with pytest.raises(ContractViolation, match="rows for 2 examples"):
        check_examples_contract(broken)


def test_the_contract_catches_a_subset_that_forgets_its_derivation(
    table: TableExamples,
) -> None:
    """The exact bug this whole change closes, now enforced for a project's own adapter.

    An adapter whose `subset` returns the parent's derivation unchanged lets a split
    computed on a subpopulation resolve against the full corpus.
    """

    class ForgetfulSubset(TableExamples):
        def subset(self, mask: np.ndarray) -> ForgetfulSubset:
            part = super().subset(mask)
            return ForgetfulSubset(
                name=part.name,
                groups=part.groups,
                digest=part.digest,
                derivation=ROOT_DERIVATION,
            )

    with pytest.raises(ContractViolation, match="derivation"):
        check_examples_contract(ForgetfulSubset(name="f", groups=["a", "a", "b", "b"]))


def test_the_contract_catches_a_subset_that_changes_the_corpus_digest() -> None:
    """The opposite error: a subset that re-derives its digest binds a split to a
    population rather than to a corpus, so every committed split stops resolving."""

    class RederivingSubset(TableExamples):
        def subset(self, mask: np.ndarray) -> RederivingSubset:
            mask = np.asarray(mask, dtype=bool)
            return RederivingSubset(name=self.name, groups=self.groups[mask])

    with pytest.raises(ContractViolation, match="digest"):
        check_examples_contract(RederivingSubset(name="r", groups=["a", "a", "b", "b"]))


def test_the_contract_catches_a_subset_of_the_wrong_length() -> None:
    class ShortSubset(TableExamples):
        def subset(self, mask: np.ndarray) -> ShortSubset:
            mask = np.asarray(mask, dtype=bool)
            return ShortSubset(
                name=self.name, groups=self.groups[mask][:-1], digest=self.digest
            )

    with pytest.raises(ContractViolation, match="length"):
        check_examples_contract(ShortSubset(name="s", groups=["a", "a", "b", "b"]))


# --- the reader contract ------------------------------------------------------------------

CHANNELS = 4
N_ROWS = 65536


@pytest.fixture
def payload(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """1 MB of float32 on disk — big enough that pickling it by value is unmistakable."""
    rows = np.arange(N_ROWS * CHANNELS, dtype=np.float32).reshape(N_ROWS, CHANNELS)
    path = tmp_path / "signal.bin"
    path.write_bytes(rows.tobytes())
    return path, rows


class EagerFactory:
    """The ADR 0005 bug, as a factory: it captures a live reader instead of reopening.

    Module-level (not a closure) so it pickles at all — which is the point. It pickles
    *successfully*, carrying the whole mapping by value, which is exactly why the failure
    stays invisible on Linux until someone runs under spawn.
    """

    def __init__(self, reader: MmapReader) -> None:
        self.reader = reader

    def __call__(self) -> MmapReader:
        return self.reader


def test_a_per_process_reader_satisfies_the_contract(
    payload: tuple[Path, np.ndarray],
) -> None:
    path, rows = payload
    open_reader = functools.partial(
        MmapReader, path, np.dtype(np.float32), CHANNELS, N_ROWS
    )
    check_reader_contract(open_reader, expected=rows)


def test_the_contract_catches_a_factory_that_carries_the_payload(
    payload: tuple[Path, np.ndarray],
) -> None:
    """The bug readers.py warns about in prose, now caught mechanically."""
    path, rows = payload
    eager = EagerFactory(MmapReader(path, np.dtype(np.float32), CHANNELS, N_ROWS))
    with pytest.raises(ContractViolation, match="by value"):
        check_reader_contract(eager, expected=rows)


class DiesInChild:
    """Opens fine in the parent, dies in the spawned process without reporting anything.

    Not contrived: it is what a child looks like when spawn cannot re-import ``__main__``,
    when a native library aborts on import, or when the OOM killer arrives.
    """

    def __init__(self, parent_pid: int, real: functools.partial) -> None:
        self.parent_pid = parent_pid
        self.real = real

    def __call__(self) -> MmapReader:
        if os.getpid() != self.parent_pid:
            os._exit(3)
        return self.real()


def test_a_worker_that_dies_silently_fails_fast_with_its_exit_code(
    payload: tuple[Path, np.ndarray],
) -> None:
    """A dead child must not be reported as a timeout.

    Waiting out the full timeout to say "produced nothing" hides both the fact that the
    process died and the code it died with — and costs a minute per run to do it.
    """
    path, rows = payload
    factory = DiesInChild(
        os.getpid(), functools.partial(MmapReader, path, np.dtype(np.float32), CHANNELS, N_ROWS)
    )

    started = time.monotonic()
    with pytest.raises(ContractViolation, match="exit code 3"):
        check_reader_contract(factory, expected=rows)
    assert time.monotonic() - started < 20.0, "a dead child was waited out, not detected"
