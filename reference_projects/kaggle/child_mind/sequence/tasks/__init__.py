"""Prefect nodes for the raw-sequence CMI experiment."""

from reference_projects.kaggle.child_mind.sequence.tasks.data import ingest, split_data
from reference_projects.kaggle.child_mind.sequence.tasks.evaluation import evaluate_model, export
from reference_projects.kaggle.child_mind.sequence.tasks.training import train

__all__ = ["evaluate_model", "export", "ingest", "split_data", "train"]
