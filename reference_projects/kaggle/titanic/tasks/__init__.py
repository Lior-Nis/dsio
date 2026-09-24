"""Cohesive task modules for the Titanic consumer flow."""

from reference_projects.kaggle.titanic.tasks.data import ingest, split_data
from reference_projects.kaggle.titanic.tasks.downstream import (
    evaluate_model,
    export,
    infer_and_submit,
)
from reference_projects.kaggle.titanic.tasks.training import train

__all__ = [
    "evaluate_model",
    "export",
    "infer_and_submit",
    "ingest",
    "split_data",
    "train",
]
