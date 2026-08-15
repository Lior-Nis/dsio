"""Optional sinks that mirror the authoritative run ledger elsewhere."""

from dsio.tracking.base import (
    ExperimentTracker,
    MultiTracker,
    NullTracker,
    flatten_params,
)

__all__ = [
    "ExperimentTracker",
    "MultiTracker",
    "NullTracker",
    "build_tracker",
    "flatten_params",
]


def build_tracker(spec: str | None, *, experiment: str = "dsio") -> ExperimentTracker:
    """Build a sink from a short name. Unknown names fail rather than silently no-op.

    A typo in ``DSIO_TRACKER`` that quietly disabled tracking would be indistinguishable
    from tracking working, which is precisely the class of silent failure this codebase
    refuses.
    """
    if not spec or spec == "none":
        return NullTracker()
    if spec == "mlflow":
        from dsio.tracking.mlflow_sink import MlflowTracker

        return MlflowTracker(experiment=experiment)
    raise ValueError(f"unknown tracker {spec!r}; known trackers: none, mlflow")
