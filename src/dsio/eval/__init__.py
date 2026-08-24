"""Evaluation: the fixed artifact contract, pooling, verdicts, and metrics.

Decision 6 (fold-as-process, ADR 0017) removed the in-process fold loop this package used
to own: one run trains and predicts one fold and writes one ``predictions.npz``
(:mod:`dsio.eval.contract`). :func:`dsio.eval.pool.pool_folds` reads N of those files back
and pools them, and :mod:`dsio.eval.verdict` compares two pooled sets.

This package is still a leaf. It imports nothing from dsio except :mod:`dsio.contracts`,
which is what lets a runner in :mod:`dsio.train` depend on this package's artifact
contract without this package ever depending back on a runner, a store, or a split.
"""
