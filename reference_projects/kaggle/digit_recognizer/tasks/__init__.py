"""Cohesive task modules for the Digit Recognizer consumer flow."""

from reference_projects.kaggle.digit_recognizer.tasks.data import ingest, split_data
from reference_projects.kaggle.digit_recognizer.tasks.downstream import (
    evaluate_model,
    export,
    infer_and_submit,
)
from reference_projects.kaggle.digit_recognizer.tasks.training import (
    pretrain_encoder,
    train_classifier,
)

__all__ = [
    "evaluate_model",
    "export",
    "infer_and_submit",
    "ingest",
    "pretrain_encoder",
    "split_data",
    "train_classifier",
]
