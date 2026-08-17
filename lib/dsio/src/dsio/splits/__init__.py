"""Leakage-safe splits: committed YAML lists of group IDs, resolved on the fly.

A split names *groups*, never windows. The group is the leakage boundary — the coarsest,
leakiest key in the data — so it is the smallest unit that may land on one side of a split.
See docs/adr/0006.
"""

from dsio.splits.generate import (
    generate,
    generate_temporal,
    group_values,
    write_splits,
    write_temporal_splits,
)
from dsio.splits.models import SCHEMA, Scheme, SplitError, SplitFile, SplitSpec
from dsio.splits.resolve import assert_no_row_overlap, resolve, summarise
from dsio.splits.stratify import BalanceReport, KeyBalance, StratifyKey
from dsio.splits.temporal import (
    TemporalBounds,
    TemporalError,
    TemporalSpec,
    TimeSpan,
    describe,
    walk_forward,
    window_times,
)

__all__ = [
    "SCHEMA",
    "BalanceReport",
    "KeyBalance",
    "Scheme",
    "SplitError",
    "SplitFile",
    "SplitSpec",
    "StratifyKey",
    "TemporalBounds",
    "TemporalError",
    "TemporalSpec",
    "TimeSpan",
    "assert_no_row_overlap",
    "describe",
    "generate",
    "generate_temporal",
    "group_values",
    "resolve",
    "summarise",
    "walk_forward",
    "window_times",
    "write_splits",
    "write_temporal_splits",
]
