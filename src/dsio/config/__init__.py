"""Typed configuration: registries, presets, overrides, and the run config root.

Structure lives in Python. YAML is a recorded output, never an authored input.
"""

from dsio.config.presets import preset
from dsio.config.schema import RunConfig

__all__ = ["RunConfig", "preset"]
