"""Inference-side components not yet proven by real downstream use.

Legacy experimental under the pre-1.0 clause in docs/component-admission.md: these
components have no real downstream use yet, carry no compatibility promise, and are
deleted at 1.0 if still unproven.
"""

from dsio.experimental.inference.outputs import (
    TensorOutput,
    validate_tensor_prediction,
)

__all__ = [
    "TensorOutput",
    "validate_tensor_prediction",
]
