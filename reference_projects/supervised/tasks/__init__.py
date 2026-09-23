"""Prefect tasks grouped by their pipeline responsibility."""

from reference_projects.supervised.tasks.data import build_data, split_data
from reference_projects.supervised.tasks.downstream import evaluate_model, infer
from reference_projects.supervised.tasks.export import export_model
from reference_projects.supervised.tasks.training import train_model

__all__ = [
    "build_data",
    "evaluate_model",
    "export_model",
    "infer",
    "split_data",
    "train_model",
]
