"""Config invariants. Each test is named for the guarantee it protects."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from dsio.config import (
    PRESETS,
    RunConfig,
    apply_override,
    load_preset_modules,
    parse_scalar,
    preset_parameters,
    resolve,
)
from dsio.config.overrides import OverrideError
from dsio.config.registry import DuplicateComponentError, Registry, UnknownComponentError
from dsio.train import check, load_runners


def test_config_is_frozen(config: RunConfig) -> None:
    """A config that mutates after being hashed makes its own run record a lie."""
    with pytest.raises(ValidationError):
        config.seed = 7  # type: ignore[misc]


def test_unknown_field_is_rejected() -> None:
    """FORGE's extra='allow' leaves silently discarded typos. This is that fix."""
    from dsio.train.tabular import TabularTask

    load_runners()
    with pytest.raises(ValidationError):
        TabularTask(dataset="iris", estimatorr="logreg")  # type: ignore[call-arg]


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


def test_override_rejects_paths_that_do_not_exist(config: RunConfig) -> None:
    with pytest.raises(OverrideError, match=r"no such config path 'task\.estimatr'"):
        apply_override(config.to_dict(), "task.estimatr", "logreg")


def test_override_error_names_the_valid_siblings(config: RunConfig) -> None:
    with pytest.raises(OverrideError, match="estimator"):
        apply_override(config.to_dict(), "task.estimatr", "logreg")


def test_override_does_not_mutate_the_input(config: RunConfig) -> None:
    original = config.to_dict()
    apply_override(original, "task.estimator", "random_forest")
    assert original["task"]["estimator"] == "logreg"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3e-4", 3e-4),
        ("1e-6", 1e-6),
        ("42", 42),
        ("0.25", 0.25),
        ("true", True),
        ("false", False),
        ("null", None),
        ("logreg", "logreg"),
        ("[1, 2]", [1, 2]),
    ],
)
def test_scalar_parsing(text: str, expected: object) -> None:
    """`3e-4` must be a float. PyYAML 1.1 reads it as a string, which is why we parse."""
    assert parse_scalar(text) == expected


def test_preset_argument_beats_config_path() -> None:
    """A dotless token matching a preset parameter sets the argument, not the config."""
    load_runners()
    load_preset_modules()
    config = resolve("spine_baseline", ["estimator=random_forest"])
    assert config.task.estimator == "random_forest"  # type: ignore[attr-defined]
    assert config.name.endswith("random_forest")


def test_dotted_token_reaches_the_config() -> None:
    load_runners()
    load_preset_modules()
    config = resolve("spine_baseline", ["task.test_fraction=0.4"])
    assert config.task.test_fraction == 0.4  # type: ignore[attr-defined]


def test_every_preset_composes_validates_and_preflights() -> None:
    """The cheapest guard against config rot. FORGE proved its value at 340 YAMLs."""
    load_runners()
    load_preset_modules()
    assert PRESETS.names(), "no presets registered; discovery is broken"
    for name in PRESETS.names():
        config = resolve(name)
        assert isinstance(config, RunConfig)
        RunConfig.model_validate(config.to_dict())
        check(config)
        assert preset_parameters(name) is not None
