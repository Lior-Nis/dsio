"""Leakage-safe splits: committed YAML lists of group IDs, resolved on the fly.

A split names *groups*, never windows. The group is the leakage boundary — the coarsest,
leakiest key in the data — so it is the smallest unit that may land on one side of a split.
See docs/adr/0006.
"""
