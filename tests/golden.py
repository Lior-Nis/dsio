"""Committed golden contract-tier metrics: the parity baseline for warehouse migrations.

Every replay test asserts its flow's evaluation metrics against ``golden_metrics.json``.
A migration that changes a consumer's numbers therefore fails loudly unless its story
declared the computation change up front and the goldens were deliberately regenerated
(``DSIO_UPDATE_GOLDEN=1 uv run pytest tests/kaggle_portfolio tests/reference_flows``).

The comparison allows a relative difference of 1e-6. Contract-tier flows are exactly
replayable on one machine, but CPU float reductions can differ in the last bits across
CPU generations (thread count, SIMD width), and CI runners are not this workstation. A
real behavior change moves metrics by orders of magnitude more than that.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent / "golden_metrics.json"
RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_TOLERANCE = 1e-9


def assert_golden_metrics(key: str, metrics: Mapping[str, float]) -> None:
    """Assert contract-tier metrics equal the committed goldens (or record them)."""
    store = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
    observed = {name: float(value) for name, value in sorted(metrics.items())}
    if os.environ.get("DSIO_UPDATE_GOLDEN") == "1":
        store[key] = observed
        GOLDEN.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    assert key in store, f"no golden metrics for {key!r}; regenerate with DSIO_UPDATE_GOLDEN=1"
    expected = store[key]
    assert set(observed) == set(expected), (
        f"{key}: metric names changed: {sorted(observed)} != {sorted(expected)}"
    )
    moved = {
        name: (expected[name], value)
        for name, value in observed.items()
        if not math.isclose(
            value, expected[name], rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE
        )
    }
    assert not moved, f"{key}: contract metrics moved from the golden baseline: {moved}"
