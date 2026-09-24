"""Cohesive task modules for the Bike Sharing consumer flow."""

from reference_projects.kaggle.bike_sharing.tasks.data import (
    SPLIT_PARAMETERS,
    ingest,
    split_data,
)
from reference_projects.kaggle.bike_sharing.tasks.downstream import (
    evaluate_model,
    export,
    infer_and_submit,
)
from reference_projects.kaggle.bike_sharing.tasks.training import train

__all__ = [
    "SPLIT_PARAMETERS",
    "evaluate_model",
    "export",
    "infer_and_submit",
    "ingest",
    "split_data",
    "train",
]
