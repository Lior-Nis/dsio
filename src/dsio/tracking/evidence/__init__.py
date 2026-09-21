"""Verified native MLflow evidence reuse."""

from dsio.tracking.evidence.references import evidence_uri
from dsio.tracking.evidence.resolution import require_evidence, resolve_evidence

__all__ = ["evidence_uri", "require_evidence", "resolve_evidence"]
