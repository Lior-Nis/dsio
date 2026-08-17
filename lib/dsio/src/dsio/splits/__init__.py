"""Leakage-safe splits: committed YAML lists of group IDs, resolved on the fly.

A split names *groups*, never windows. The group is the leakage boundary — the coarsest,
leakiest key in the data — so it is the smallest unit that may land on one side of a split.
See docs/adr/0006.
"""

from dsio.splits.generate import generate, group_values, write_splits
from dsio.splits.models import SCHEMA, Scheme, SplitError, SplitFile, SplitSpec
from dsio.splits.resolve import assert_no_row_overlap, resolve, summarise

__all__ = [
    "SCHEMA",
    "Scheme",
    "SplitError",
    "SplitFile",
    "SplitSpec",
    "assert_no_row_overlap",
    "generate",
    "group_values",
    "resolve",
    "summarise",
    "write_splits",
]
