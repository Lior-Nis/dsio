"""The remaining registry is private metric dispatch, not training orchestration."""

from __future__ import annotations

import pytest

import dsio.config
from dsio.config.registry import DuplicateComponentError, Registry, UnknownComponentError


def test_config_exports_no_orchestration_decorator() -> None:
    assert dsio.config.__all__ == ["ComponentConfig", "ComponentError", "resolve_component"]
    assert not hasattr(dsio.config, "preset")


def test_unknown_component_suggests_a_near_match() -> None:
    registry: Registry[int] = Registry("widget-suggest")
    registry.add("random_forest", 1)
    with pytest.raises(UnknownComponentError, match="did you mean 'random_forest'"):
        registry.get("randomforest")


def test_unknown_component_lists_options_when_nothing_is_close() -> None:
    registry: Registry[int] = Registry("widget-list")
    registry.add("alpha", 1)
    with pytest.raises(UnknownComponentError, match="known widget-lists: alpha"):
        registry.get("zzzzzz")


def test_duplicate_registration_fails_loudly() -> None:
    """Silent overwrite makes one entry unreachable depending on import order."""
    registry: Registry[int] = Registry("widget-dupe")
    registry.add("thing", 1)
    with pytest.raises(DuplicateComponentError):
        registry.add("thing", 2)
