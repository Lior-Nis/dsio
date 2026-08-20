"""Leakage-safe splits: committed YAML lists of group IDs, resolved on the fly.

A split names *groups*, never windows. The group is the leakage boundary — the coarsest,
leakiest key in the data — so it is the smallest unit that may land on one side of a split.
See docs/adr/0006.
"""

from dsio.splits.folds import fold_paths, folds_from_splits, load_folds
from dsio.splits.models import SCHEMA, SplitError, SplitFile
from dsio.splits.resolve import assert_no_row_overlap, resolve, resolve_masks, summarise
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
    "SplitError",
    "SplitFile",
    "TemporalBounds",
    "TemporalError",
    "TemporalSpec",
    "TimeSpan",
    "assert_no_row_overlap",
    "describe",
    "fold_paths",
    "folds_from_splits",
    "load_folds",
    "resolve",
    "resolve_masks",
    "summarise",
    "walk_forward",
    "window_times",
]
