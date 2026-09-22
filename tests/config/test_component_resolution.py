"""One importable component format for every native ecosystem object."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn


def spawn_factory() -> None:
    return None


def test_resolve_component_builds_and_validates_a_named_native_object() -> None:
    from dsio.config.components import resolve_component

    configured = {
        "reference": "torch.nn:Linear",
        "parameters": {"in_features": 3, "out_features": 2, "bias": False},
    }

    layer = resolve_component(configured, expected=nn.Module)

    assert isinstance(layer, nn.Linear)
    assert layer.in_features == 3
    assert layer.out_features == 2
    assert layer.bias is None


def test_runtime_arguments_and_declared_parameters_share_one_constructor_call() -> None:
    from dsio.config.components import resolve_component

    parameter = nn.Parameter(torch.ones(1))
    optimizer = resolve_component(
        {
            "reference": "torch.optim:AdamW",
            "parameters": {"lr": 0.025, "weight_decay": 0.0},
        },
        [parameter],
        expected=torch.optim.Optimizer,
    )

    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.param_groups[0]["lr"] == 0.025


@pytest.mark.parametrize(
    ("configured", "message"),
    [
        ({"reference": "torch.nn.Linear"}, "module:qualname"),
        ({"reference": "__main__:Linear"}, "absolutely importable"),
        ({"reference": "__mp_main__:Linear"}, "absolutely importable"),
        ({"reference": ".relative:Linear"}, "absolutely importable"),
        ({"reference": "missing_package:Thing"}, "could not import"),
        ({"reference": "torch.nn:Missing"}, "does not resolve"),
        ({"reference": "torch.nn:Linear", "parameters": {"bad": object()}}, "canonical"),
        ({"reference": "torch.nn:Linear", "parameters": {"bad": {1, 2}}}, "canonical"),
        ({"reference": "torch.nn:Linear", "parameters": {}, "extra": True}, "only"),
    ],
)
def test_invalid_component_configuration_fails_before_construction(
    configured: dict[str, Any],
    message: str,
) -> None:
    from dsio.config.components import ComponentError, resolve_component

    with pytest.raises(ComponentError, match=message):
        resolve_component(configured)


def test_cyclic_component_parameters_fail_at_the_configuration_boundary() -> None:
    from dsio.config.components import ComponentError, validate_component_config

    parameters: dict[str, Any] = {}
    parameters["cycle"] = parameters

    with pytest.raises(ComponentError, match="canonical"):
        validate_component_config(
            {"reference": "torch.nn:Linear", "parameters": parameters}
        )


def test_resolved_component_must_match_the_requested_native_contract() -> None:
    from dsio.config.components import ComponentError, resolve_component

    with pytest.raises(ComponentError, match="torch.optim.Optimizer"):
        resolve_component(
            {
                "reference": "torch.nn:Linear",
                "parameters": {"in_features": 1, "out_features": 1},
            },
            expected=torch.optim.Optimizer,
        )


@pytest.mark.parametrize("value", [lambda: None, object()])
def test_anonymous_or_unimportable_components_are_rejected(value: object) -> None:
    from dsio.config.components import ComponentError, importable_reference

    with pytest.raises(ComponentError, match="named importable|module:qualname"):
        importable_reference(value)


def test_local_components_are_rejected() -> None:
    from dsio.config.components import ComponentError, importable_reference

    def local_factory() -> None:
        return None

    with pytest.raises(ComponentError, match="local|named importable"):
        importable_reference(local_factory)


def test_spawn_process_main_components_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dsio.config.components import ComponentError, importable_reference

    monkeypatch.setattr(spawn_factory, "__module__", "__mp_main__")

    with pytest.raises(ComponentError, match="named importable"):
        importable_reference(spawn_factory)
