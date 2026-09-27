"""Prefect nodes for the Child Mind experiment."""

from reference_projects.kaggle.child_mind.tasks.data import (
    SPLIT_PARAMETERS,
    ingest,
    split_data,
)
from reference_projects.kaggle.child_mind.tasks.downstream import (
    evaluate_model,
    export,
    infer_and_submit,
)
from reference_projects.kaggle.child_mind.tasks.training import train

__all__ = [
    "SPLIT_PARAMETERS",
    "evaluate_model",
    "export",
    "infer_and_submit",
    "ingest",
    "split_data",
    "train",
]
