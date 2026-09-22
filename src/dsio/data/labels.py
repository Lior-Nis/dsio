"""Reusable label providers over canonical stores."""

from __future__ import annotations

import numpy as np

from dsio.data.store import SignalStore


def entity_attribute_labels(store: SignalStore, attribute: str) -> np.ndarray:
    """Expand one numeric entity attribute to every row owned by that entity."""
    labels = np.empty(store.n_rows, dtype=np.float32)
    for entity in store.entities:
        try:
            value = float(entity.attrs[attribute])
        except KeyError:
            raise ValueError(
                f"entity {entity.entity_id!r} has no label attribute {attribute!r}"
            ) from None
        labels[entity.start_row : entity.end_row] = value
    return labels
