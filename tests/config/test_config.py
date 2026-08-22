"""Config invariants. Each test is named for the guarantee it protects."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dsio.config import RunConfig
from dsio.config.overrides import OverrideError, apply_override, parse_scalar
from dsio.config.presets import PRESETS, load_preset_modules, preset_parameters, resolve
from dsio.config.registry import DuplicateComponentError, Registry, UnknownComponentError
from dsio.train import load_runners
from dsio.train.runner import check


def test_config_is_frozen(config: RunConfig) -> None:
    """A config that mutates after being hashed makes its own run record a lie."""
    with pytest.raises(ValidationError):
        config.seed = 7  # type: ignore[misc]


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


def test_override_rejects_paths_that_do_not_exist(config: RunConfig) -> None:
    with pytest.raises(OverrideError, match=r"no such config path 'task\.labelz'"):
        apply_override(config.to_dict(), "task.labelz", "tone")


def test_override_error_names_the_valid_siblings(config: RunConfig) -> None:
    with pytest.raises(OverrideError, match="labels"):
        apply_override(config.to_dict(), "task.labelz", "tone")


def test_override_does_not_mutate_the_input(config: RunConfig) -> None:
    original = config.to_dict()
    apply_override(original, "task.labels", "other")
    assert original["task"]["labels"] == "tone"


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
    config = resolve("spine_baseline", ["lr=0.01"])
    assert config.task.lr == 0.01  # type: ignore[attr-defined]
    assert config.name.endswith("lr0.01")


def test_dotted_token_reaches_the_config() -> None:
    load_runners()
    load_preset_modules()
    config = resolve("spine_baseline", ["task.batch_size=64"])
    assert config.task.batch_size == 64  # type: ignore[attr-defined]


def test_resolving_a_preset_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resolve()` only composes a `RunConfig` — `--dry-run`'s entire contract
    ("resolve and validate only") depends on this (see `dsio.cli.run_cmd`).
    `spine_baseline` stages a synthetic corpus via a registered stage hook, but that
    hook must never run here, only on the path that is actually about to execute."""
    monkeypatch.chdir(tmp_path)
    load_runners()
    load_preset_modules()
    resolve("spine_baseline")
    assert list(tmp_path.iterdir()) == []


def test_every_preset_composes_validates_and_preflights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cheapest guard against config rot: every registered preset must still build.

    Mirrors the CLI's real-execute path (`dsio.cli.run_cmd`) exactly: resolve (pure),
    run any registered `STAGE_HOOKS` entry, then preflight. `spine_baseline`'s stage
    hook writes real files, so this chdirs to a scratch dir — not because `resolve()`
    writes anything (it does not), but because staging does.
    """
    from dsio.config.presets import STAGE_HOOKS

    monkeypatch.chdir(tmp_path)
    load_runners()
    load_preset_modules()

    assert PRESETS.names(), "no presets registered; discovery is broken"
    for name in PRESETS.names():
        config = resolve(name)
        assert isinstance(config, RunConfig)
        RunConfig.model_validate(config.to_dict())
        if name in STAGE_HOOKS:
            STAGE_HOOKS.get(name)(config)
        check(config)
        assert preset_parameters(name) is not None


def test_at_most_one_preset_has_a_stage_hook() -> None:
    """Parked, not fixed: ``run_cmd.py``'s ``check_torch``/preflight step raises
    ``UnknownComponentError`` for every registry a typo could hit, so on a hooked preset a
    genuine ``backbone.name="nope"`` is reported as "just needs staging" (dry-run's
    ``pending_stage`` hint) rather than as the typo it is — misleading, though a real run
    still fails loudly on the same error, so nothing is silently wrong, only misdirected.

    Exposure today is exactly one preset (``spine_baseline``), which is why this was
    parked rather than fixed with the ~10-15 line change of having a stage hook declare
    what it provides. The park's own trigger is "before a second preset gains a stage
    hook, because exposure grows with hooks, not with time" — this assertion is that
    trigger made mechanical, so crossing it fails a test instead of depending on someone
    re-reading a ledger entry. If this test starts failing: land the declare-what-you-
    provide fix (or an equivalent) before adding the second hook, don't just raise this
    bound.
    """
    from dsio.config.presets import STAGE_HOOKS

    load_preset_modules()
    assert len(STAGE_HOOKS) <= 1
