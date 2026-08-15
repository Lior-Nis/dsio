"""Typed configuration: registries, presets, overrides, and the run config root.

Structure lives in Python. YAML is a recorded output, never an authored input.
"""

from dsio.config.overrides import (
    OverrideError,
    apply_override,
    apply_overrides,
    parse_scalar,
    split_override,
)
from dsio.config.presets import (
    PRESETS,
    PresetError,
    load_preset_modules,
    preset,
    preset_parameters,
    resolve,
)
from dsio.config.registry import (
    DuplicateComponentError,
    Registry,
    UnknownComponentError,
    all_registries,
)
from dsio.config.schema import TASKS, RunConfig, TaskConfig

__all__ = [
    "PRESETS",
    "TASKS",
    "DuplicateComponentError",
    "OverrideError",
    "PresetError",
    "Registry",
    "RunConfig",
    "TaskConfig",
    "UnknownComponentError",
    "all_registries",
    "apply_override",
    "apply_overrides",
    "load_preset_modules",
    "parse_scalar",
    "preset",
    "preset_parameters",
    "resolve",
    "split_override",
]
