"""Consumer-project identity checks for static admission."""

from dsio.experimental.admission.syntax.projects.consumers import (
    references_consumer_names,
    string_values,
)
from dsio.experimental.admission.syntax.projects.contexts import project_contexts
from dsio.experimental.admission.syntax.projects.identity import references_project_identity

__all__ = [
    "project_contexts",
    "references_consumer_names",
    "references_project_identity",
    "string_values",
]
