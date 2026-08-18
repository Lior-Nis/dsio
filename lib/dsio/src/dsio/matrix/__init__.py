"""Resumable job matrices and adaptive search, both emitting ordinary Runs.

Both commands are content-addressed: a cell's identity is the sha256 of its resolved
config, and the run ledger is the resume state. That replaces the usual shape for this —
shell scripts plus a polling daemon with its own progress file.
"""

from dsio.matrix.cells import (
    Axis,
    Cell,
    MatrixError,
    expand,
    parse_axes,
    parse_axis,
    size,
    validate,
)
from dsio.matrix.execute import CellResult, MatrixReport, completed_hashes, run_matrix
from dsio.matrix.search import SearchReport, SearchSpace, TrialResult, parse_space, run_search

__all__ = [
    "Axis",
    "Cell",
    "CellResult",
    "MatrixError",
    "MatrixReport",
    "SearchReport",
    "SearchSpace",
    "TrialResult",
    "completed_hashes",
    "expand",
    "parse_axes",
    "parse_axis",
    "parse_space",
    "run_matrix",
    "run_search",
    "size",
    "validate",
]
