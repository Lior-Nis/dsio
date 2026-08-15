"""Presets: functions that build a validated :class:`RunConfig`.

A preset is the unit a user launches. Variants are *arguments*, never files — there must
be no path by which ``lr=1e-6`` becomes something you check in. FORGE reached 340 YAMLs
despite a written policy against exactly that, because the system made a new file the
easy way to express a new value.

Override resolution follows one rule: a dotless token matching a preset parameter sets
that argument; anything else is a dotted path into the resulting config.
"""

from __future__ import annotations

import importlib
import inspect
import os
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from dsio.config.overrides import apply_override, split_override
from dsio.config.registry import Registry
from dsio.config.schema import RunConfig

PresetFn = Callable[..., RunConfig]

PRESETS: Registry[PresetFn] = Registry("preset")

PRESET_MODULES_ENV = "DSIO_PRESET_MODULES"
"""Comma-separated module paths to import before resolving a preset."""

PRESET_ENTRY_POINT_GROUP = "dsio.presets"
"""Entry-point group projects use to advertise their preset modules."""


class PresetError(ValueError):
    """Raised when a preset is malformed or produces something other than a RunConfig."""


def preset(fn: PresetFn) -> PresetFn:
    """Register ``fn`` as a preset under its own name."""
    PRESETS.add(fn.__name__, fn)
    return fn


def load_preset_modules() -> list[str]:
    """Import modules that define presets, returning the ones that loaded.

    Discovery is by entry point, so a project advertises its presets in its own
    pyproject.toml and dsio never has to know the project's package name::

        [project.entry-points."dsio.presets"]
        pdm = "pdm.presets"

    A declared entry point that fails to import is an error, not a skip: a silently
    empty preset list is indistinguishable from a project with no presets, and the
    difference matters at 2am.
    """
    loaded: list[str] = []
    for entry in entry_points(group=PRESET_ENTRY_POINT_GROUP):
        importlib.import_module(entry.value)
        loaded.append(entry.value)

    configured = os.environ.get(PRESET_MODULES_ENV, "").strip()
    if configured:
        for module in (part.strip() for part in configured.split(",") if part.strip()):
            importlib.import_module(module)
            loaded.append(module)
    return loaded


def preset_parameters(name: str) -> dict[str, inspect.Parameter]:
    """Return the named parameters a preset accepts."""
    return dict(inspect.signature(PRESETS.get(name)).parameters)


def resolve(name: str, tokens: list[str] | None = None) -> RunConfig:
    """Build the :class:`RunConfig` for ``name``, applying ``tokens`` as overrides."""
    fn = PRESETS.get(name)
    parameters = dict(inspect.signature(fn).parameters)

    preset_kwargs: dict[str, Any] = {}
    config_overrides: list[tuple[str, Any]] = []

    for token in tokens or []:
        path, value = split_override(token)
        if "." not in path and path in parameters:
            preset_kwargs[path] = value
        else:
            config_overrides.append((path, value))

    config = fn(**preset_kwargs)
    if not isinstance(config, RunConfig):
        raise PresetError(
            f"preset {name!r} returned {type(config).__name__}, expected RunConfig"
        )

    if not config_overrides:
        return config

    data = config.to_dict()
    for path, value in config_overrides:
        data = apply_override(data, path, value)
    return RunConfig.model_validate(data)
