"""Experimental data-side components: reviewed, without a compatibility promise.

Which ones are proven by real consumers and which are pre-1.0 legacy (no real use yet,
deleted at 1.0 if still unproven) is recorded in docs/component-warehouse/catalog.md.
"""

from dsio.experimental.data.arrays import collate_arrays
from dsio.experimental.data.fitting import fit_standardization, record_fitted
from dsio.experimental.data.items import StoredItems
from dsio.experimental.data.labels import (
    entity_attribute_labels,
)
from dsio.experimental.data.padding import PadCollator
from dsio.experimental.data.samples import (
    StoredSamples,
    stored_samples,
)
from dsio.experimental.data.windows import (
    WindowDataset,
)

__all__ = [
    "PadCollator",
    "StoredItems",
    "StoredSamples",
    "WindowDataset",
    "collate_arrays",
    "entity_attribute_labels",
    "fit_standardization",
    "record_fitted",
    "stored_samples",
]
