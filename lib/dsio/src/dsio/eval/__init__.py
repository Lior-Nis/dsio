"""Evaluation: the fold loop, the fixed artifact contract, and metrics.

This package is a leaf. It imports nothing from dsio except :mod:`dsio.contracts` and the
registry, which is what lets the fold loop drive an sklearn pipeline, a Lightning module or
a forecast model without any of them appearing in its type signatures.
"""
