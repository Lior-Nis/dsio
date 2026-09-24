"""Bind a checkpoint to the training inputs that produced it."""

from __future__ import annotations

from dsio.inference.predictor import PredictorError
from dsio.tracking import TrackingError, require_evidence
from dsio.train.artifacts import ArtifactRef


def require_checkpoint_lineage(
    checkpoint: ArtifactRef,
    *,
    training_run_id: str,
    training_identity: str,
    dataset_digest: str,
    split_digest: str,
) -> None:
    """Reject a checkpoint that did not train on the declared dataset and split."""
    if checkpoint.run_id != training_run_id:
        raise PredictorError("checkpoint evidence belongs to a different training run")
    try:
        require_evidence(
            training_run_id,
            identity=training_identity,
            required_artifacts=(checkpoint.path,),
            expected_configuration={
                "dataset_digest": dataset_digest,
                "split_digest": split_digest,
            },
        )
    except TrackingError as error:
        raise PredictorError(f"checkpoint lineage is invalid: {error}") from error
