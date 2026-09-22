"""SSL-specific tasks exposed to the project-owned flow."""

from reference_projects.self_supervised.tasks.export import export_model
from reference_projects.self_supervised.tasks.training import train_model

__all__ = ["export_model", "train_model"]
