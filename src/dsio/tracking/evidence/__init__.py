"""Verified native MLflow evidence reuse."""

from dsio.tracking.evidence.references import evidence_uri
from dsio.tracking.evidence.resolution import require_evidence, resolve_evidence
from dsio.tracking.evidence.splits import (
    canonical_dataset_digest,
    load_split_evidence,
    record_split_evidence,
)

__all__ = [
    "canonical_dataset_digest",
    "evidence_uri",
    "load_split_evidence",
    "record_split_evidence",
    "require_evidence",
    "resolve_evidence",
]
