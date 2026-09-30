"""Data-side components not yet proven by real downstream use.

Legacy experimental under the pre-1.0 clause in docs/component-admission.md: these
components have no real downstream use yet, carry no compatibility promise, and are
deleted at 1.0 if still unproven.
"""

from dsio.experimental.data.labels import (
    entity_attribute_labels,
)
from dsio.experimental.data.samples import (
    StoredSamples,
    stored_samples,
)
from dsio.experimental.data.windows import (
    WindowDataset,
)

__all__ = [
    "StoredSamples",
    "WindowDataset",
    "entity_attribute_labels",
    "stored_samples",
]
