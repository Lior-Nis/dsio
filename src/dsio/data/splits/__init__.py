"""Governed split generation, manifests, and leakage-safe replay.

A split names *groups*, never windows. The group is the leakage boundary — the coarsest,
leakiest key in the data — so it is the smallest unit that may land on one side of a split.
See docs/adr/0006.
"""

from dsio.data.splits.generate import generate
from dsio.data.splits.validation import validate

__all__ = ["generate", "validate"]
