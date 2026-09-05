"""The obligation :mod:`dsio.data.readers` states in prose, as an executable check.

    Every reader is opened **per process**. This is not an optimisation. A ``np.memmap``
    created in a parent process and handed to a ``spawn``-based DataLoader worker is
    pickled *by value*: it serialises the whole array instead of the mapping.

That bug does not raise. On Linux, ``fork`` inherits the mapping and everything works, so
it survives review, survives CI, and surfaces on macOS or the first time someone sets
``multiprocessing_context="spawn"`` to dodge a CUDA fork issue — by which point it looks
like a memory problem, not a pickling one.

The unit that crosses a process boundary is therefore not a reader but a **factory** that
opens one. This suite checks a factory: that it reads correctly, that pickling it does not
drag the payload along, and that a genuinely spawned process can open and read through it.
"""

from __future__ import annotations

import multiprocessing
import pickle
import queue as queue_mod
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from dsio.data.readers import SignalReader
from dsio.testing.examples_contract import ContractViolation

DEFAULT_MAX_PICKLED_BYTES = 64 * 1024
"""A factory is configuration — a path, a dtype, a shape. Anything this size is payload."""

SPAWN_TIMEOUT_S = 60.0
_POLL_S = 0.05
_DRAIN_GRACE_S = 1.0


def check_reader_contract(
    open_reader: Callable[[], SignalReader],
    *,
    expected: np.ndarray,
    max_pickled_bytes: int = DEFAULT_MAX_PICKLED_BYTES,
    spawn: bool = True,
) -> None:
    """Assert ``open_reader`` is safe to hand to a DataLoader worker.

    ``open_reader`` must be picklable *by reference* — a module-level function, a
    ``functools.partial`` of one, or a small object — because that is what a spawned
    worker receives. ``expected`` is the full payload the reader serves, used to check
    reads both in-process and in the spawned one::

        def test_my_reader_reopens_per_process(tmp_path):
            check_reader_contract(
                functools.partial(MyReader, path, dtype, channels, n_rows),
                expected=rows,
            )

    Set ``spawn=False`` only to skip the process round-trip on a platform that cannot
    spawn; the pickle-size check still runs and catches the same bug more cheaply.
    """
    expected = np.asarray(expected)
    # Size first, and deliberately so: `_check_reads` closes the reader it opens, which
    # for an *eager* factory (the bug being hunted) empties the very handle the size
    # check is looking for. Measure the factory as it was handed over, before anything
    # else has had a chance to mutate what it captured.
    _check_pickles_by_reference(open_reader, max_pickled_bytes)
    _check_reads(open_reader, expected)
    if spawn:
        _check_reads_after_spawn(open_reader, expected)


def _check_reads(open_reader: Callable[[], SignalReader], expected: np.ndarray) -> None:
    reader = open_reader()
    try:
        head = np.asarray(reader.read_rows(0, min(8, len(expected))))
        if not np.array_equal(head, expected[: len(head)]):
            raise ContractViolation("read_rows(0, n) did not return the first n rows")
        if len(expected) > 4:
            start = len(expected) // 2
            middle = np.asarray(reader.read_rows(start, 4))
            if not np.array_equal(middle, expected[start : start + 4]):
                raise ContractViolation(
                    f"read_rows({start}, 4) did not return rows {start}..{start + 4}; "
                    "an offset is being applied somewhere it should not be"
                )
    finally:
        reader.close()


def _check_pickles_by_reference(
    open_reader: Callable[[], SignalReader], max_pickled_bytes: int
) -> None:
    """The cheap check, and the one that names the bug.

    A factory holding a live ``np.memmap`` pickles *successfully* — that is precisely why
    the failure is invisible — but it serialises the mapped array by value. Size is what
    tells the two apart.
    """
    try:
        blob = pickle.dumps(open_reader)
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        raise ContractViolation(
            f"the reader factory cannot be pickled ({exc}), so a spawned DataLoader "
            "worker cannot receive it. Use a module-level function or a "
            "functools.partial of one, not a closure or a lambda."
        ) from exc

    if len(blob) > max_pickled_bytes:
        raise ContractViolation(
            f"the reader factory pickles to {len(blob):,} bytes, over the "
            f"{max_pickled_bytes:,} allowed. It is carrying the payload by value rather "
            "than reopening it: a live np.memmap (or an open handle) is reachable from "
            "the factory. Capture the path and the shape, and open in the worker."
        )


def _read_in_worker(
    open_reader: Callable[[], SignalReader],
    start: int,
    n_rows: int,
    out: Any,
) -> None:
    """Runs in the spawned process; module-level so it pickles by reference."""
    try:
        reader = open_reader()
    except Exception as exc:  # noqa: BLE001 - reported to the parent, not swallowed
        out.put(("error", f"{type(exc).__name__}: {exc}"))
        return
    try:
        out.put(("ok", np.asarray(reader.read_rows(start, n_rows))))
    except Exception as exc:  # noqa: BLE001 - reported to the parent, not swallowed
        out.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        reader.close()


def _await_result(process: Any, out: Any) -> tuple[str, Any]:
    """Wait for the child's message, but notice if the child dies instead of sending one.

    A plain blocking ``get(timeout=...)`` cannot tell "still working" from "already dead",
    so a worker that dies on startup — spawn unable to re-import ``__main__``, a native
    library aborting, the OOM killer — is reported a full timeout later as having
    "produced nothing", which names neither the death nor its cause. Polling lets the exit
    code be the error message, immediately.
    """
    deadline = time.monotonic() + SPAWN_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            result: tuple[str, Any] = out.get(timeout=_POLL_S)
            return result
        except queue_mod.Empty:
            if process.is_alive():
                continue
            # The child is gone. It may still have sent something the queue's feeder
            # thread had not flushed when it exited, so look once more before blaming it.
            try:
                late: tuple[str, Any] = out.get(timeout=_DRAIN_GRACE_S)
                return late
            except queue_mod.Empty:
                raise ContractViolation(
                    f"the spawned worker exited with exit code {process.exitcode} without "
                    "producing a result. It never got as far as opening the reader — the "
                    "factory, or something it imports, does not survive a spawn."
                ) from None
    raise ContractViolation(
        f"the spawned worker produced nothing in {SPAWN_TIMEOUT_S:g}s and is still running"
    )


def _check_reads_after_spawn(
    open_reader: Callable[[], SignalReader], expected: np.ndarray
) -> None:
    """The real thing: a process that inherits nothing must still read the right rows."""
    n_rows = min(128, len(expected))
    context = multiprocessing.get_context("spawn")
    out = context.Queue()
    process = context.Process(target=_read_in_worker, args=(open_reader, 0, n_rows, out))
    process.start()
    try:
        # Drain before join: a child blocks writing to a full pipe, and joining first
        # would deadlock rather than fail.
        status, payload = _await_result(process, out)
        if status == "error":
            raise ContractViolation(
                f"the reader failed in a spawned process: {payload}. It works in-process, "
                "so something it needs was inherited rather than reopened."
            )
        if not np.array_equal(np.asarray(payload), expected[:n_rows]):
            raise ContractViolation(
                "the spawned worker read different bytes than the parent did; the reader "
                "is not reopening the same payload"
            )
    finally:
        process.join(timeout=SPAWN_TIMEOUT_S)
        if process.is_alive():
            process.terminate()
            process.join()
