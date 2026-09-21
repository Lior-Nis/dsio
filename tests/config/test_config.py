"""Config invariants. Each test is named for the guarantee it protects."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

import dsio.config
from dsio.config import RunConfig
from dsio.config.registry import DuplicateComponentError, Registry, UnknownComponentError


def test_config_is_frozen(config: RunConfig) -> None:
    """A config that mutates after being hashed makes its own run record a lie."""
    with pytest.raises(ValidationError):
        config.seed = 7  # type: ignore[misc]


def test_config_exports_no_orchestration_decorator() -> None:
    assert dsio.config.__all__ == ["RunConfig"]
    assert not hasattr(dsio.config, "preset")


def test_unknown_field_is_rejected() -> None:
    """A schema that accepts unknown keys turns a typo into a silently ignored setting."""
    from dsio.train.torch_task import TrainerConfig

    with pytest.raises(ValidationError):
        TrainerConfig(max_epochsz=1)  # type: ignore[call-arg]


def test_config_round_trips_through_yaml(config: RunConfig) -> None:
    """The recorded YAML must rebuild the exact same typed object, subclass included."""
    restored = RunConfig.model_validate(yaml.safe_load(yaml.safe_dump(config.to_dict())))
    assert restored == config
    assert restored.config_hash == config.config_hash
    assert type(restored.task) is type(config.task)


def test_config_hash_is_order_independent(config: RunConfig) -> None:
    """Two configs equal in content must hash identically regardless of key order."""
    data = config.to_dict()
    reordered = dict(reversed(list(data.items())))
    assert RunConfig.model_validate(reordered).config_hash == config.config_hash


def test_config_hash_changes_with_any_value(config: RunConfig) -> None:
    assert config.model_copy(update={"seed": 43}).config_hash != config.config_hash


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
