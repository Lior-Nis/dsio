"""Base model every dsio config and value object derives from.

The two settings here are load-bearing, not stylistic.

``extra="forbid"`` — FORGE declared every leaf schema ``extra="allow"`` with a bare
``_target_: str``, so a typo like ``deph: 12`` was silently discarded and training
proceeded with the default. Forbidding extras turns that into an error at construction.

``frozen=True`` — a config that mutates after it has been hashed and recorded makes the
run record a lie. Freezing makes the recorded hash true for the run's whole lifetime.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DsioModel(BaseModel):
    """Immutable, strictly-validated base for configs and value objects."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        # Reject silent lossy coercion such as 1.7 -> 1 for an int field.
        strict=False,
        arbitrary_types_allowed=False,
    )
