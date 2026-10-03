from __future__ import annotations

import pytest
import torch
from torch import nn

from dsio.config.components import resolve_component
from dsio.experimental.model.residual import BaselineResidual


def _input() -> torch.Tensor:
    x = torch.randn(2, 5, 4)
    x[:, :, 1] = torch.tensor([[2.0] * 5, [3.0] * 5])
    x[:, :, 3] = torch.tensor([[1.0] * 5, [1.0, 1.0, 1.0, 0.0, 0.0]])
    return x


def test_initial_output_is_exactly_the_valid_baseline() -> None:
    x = _input()
    model = BaselineResidual(features=4, baseline_channel=1, validity_channel=3)

    expected = torch.where(x[:, :, 3].bool(), x[:, :, 1], 0.0)
    assert torch.equal(model(x), expected)


def test_initial_output_preserves_a_higher_precision_baseline_exactly() -> None:
    x = _input().double()
    x[0, 0, 1] = 1.0000000000000002
    model = BaselineResidual(features=4, baseline_channel=1, validity_channel=3)

    expected = torch.where(x[:, :, 3].bool(), x[:, :, 1], 0.0)
    result = model(x)

    assert result.dtype == torch.float64
    assert torch.equal(result, expected)


def test_correction_is_bounded_and_residual_receives_gradients() -> None:
    x = _input()
    model = BaselineResidual(
        features=4,
        baseline_channel=1,
        validity_channel=3,
        bound=0.25,
    )
    with torch.no_grad():
        final = model.residual[-1]
        assert isinstance(final, nn.Linear)
        final.weight.fill_(0.1)
        final.bias.fill_(0.1)

    output = model(x)
    valid = x[:, :, 3].bool()
    correction = output[valid] - x[:, :, 1][valid]
    assert torch.all(correction.abs() <= 0.25)

    output.sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.residual.parameters()
    )


def test_nonfinite_padding_is_sanitized_before_the_residual() -> None:
    x = _input()
    x[1, 3:, :] = float("nan")
    x[1, 3:, 3] = 0.0
    model = BaselineResidual(features=4, baseline_channel=1, validity_channel=3)

    result = model(x)

    assert torch.isfinite(result).all()
    assert result[1, 3:].eq(0).all()


def test_finite_values_must_remain_representable_in_the_model_dtype() -> None:
    x = _input().double()
    x[0, 0, 0] = torch.finfo(torch.float64).max
    model = BaselineResidual(features=4, baseline_channel=1, validity_channel=3)

    with pytest.raises(RuntimeError, match="not representable in the model dtype"):
        model(x)


def test_nonfinite_residual_logits_raise_a_named_error() -> None:
    x = _input()
    x[:, :, :3] = torch.finfo(torch.float32).max
    model = BaselineResidual(features=4, baseline_channel=1, validity_channel=3)
    with torch.no_grad():
        first = model.residual[0]
        assert isinstance(first, nn.Linear)
        first.weight.fill_(2)
        first.bias.zero_()

    with pytest.raises(RuntimeError, match="residual network produced non-finite values"):
        model(x)


@pytest.mark.parametrize(
    ("x", "message"),
    [
        (torch.ones(2, 4), "rank 3"),
        (torch.ones(2, 3, 3), "feature axis 2 expected 4"),
        (torch.ones(0, 3, 4), "batch axis 0 must be non-empty"),
        (torch.ones(2, 0, 4), "point axis 1 must be non-empty"),
        (torch.ones(2, 3, 4, dtype=torch.int64), "floating tensor"),
    ],
)
def test_input_contract_errors_are_named(x: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BaselineResidual(features=4, baseline_channel=1, validity_channel=3)(x)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda x: x[:, :, 3].fill_(2), "validity channel must contain only zero or one"),
        (lambda x: x[:, :, 1].fill_(float("nan")), "observed features must be finite"),
    ],
)
def test_value_contract_errors_are_named(mutate: object, message: str) -> None:
    x = _input()
    assert callable(mutate)
    mutate(x)
    with pytest.raises(RuntimeError, match=message):
        BaselineResidual(features=4, baseline_channel=1, validity_channel=3)(x)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"features": 0, "baseline_channel": 0, "validity_channel": 1}, "features"),
        ({"features": 4, "baseline_channel": 1, "validity_channel": 1}, "distinct"),
        ({"features": 4, "baseline_channel": 4, "validity_channel": 3}, "baseline_channel"),
        ({"features": 4, "baseline_channel": 1.0, "validity_channel": 3}, "integer"),
        ({"features": 4, "baseline_channel": 1, "validity_channel": 3, "bound": 0}, "bound"),
        (
            {"features": 4, "baseline_channel": 1, "validity_channel": 3, "bound": 1e39},
            "representable",
        ),
    ],
)
def test_configuration_errors_are_named(parameters: dict[str, int | float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BaselineResidual(**parameters)


def test_component_config_is_deterministic_and_fullgraph_compilable() -> None:
    config = {
        "reference": "dsio.experimental.model.residual:BaselineResidual",
        "parameters": {"features": 4, "baseline_channel": 1, "validity_channel": 3},
    }
    torch.manual_seed(5)
    first = resolve_component(config, expected=BaselineResidual)
    torch.manual_seed(5)
    second = resolve_component(config, expected=BaselineResidual)
    x = _input()

    assert torch.equal(first(x), second(x))
    compiled = torch.compile(first, backend="eager", fullgraph=True)
    assert torch.equal(compiled(x), first(x))


def test_bound_must_remain_representable_after_dtype_conversion() -> None:
    model = BaselineResidual(
        features=4,
        baseline_channel=1,
        validity_channel=3,
        bound=1e-8,
    ).half()

    with pytest.raises(RuntimeError, match="bound is not representable"):
        model(_input().half())


def test_input_and_parameters_must_share_a_device() -> None:
    model = BaselineResidual(features=4, baseline_channel=1, validity_channel=3).to("meta")

    with pytest.raises(ValueError, match="must share a device"):
        model(_input())


def test_seeded_model_matches_the_former_architecture_exactly() -> None:
    class Former(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.residual = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 1))
            final = self.residual[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            valid = x[:, :, 3]
            correction = 0.01 * torch.tanh(self.residual(x).squeeze(-1))
            return (x[:, :, 1] + correction) * valid

    x = _input()
    torch.manual_seed(8)
    former = Former()
    torch.manual_seed(8)
    current = BaselineResidual(
        features=4,
        hidden=6,
        baseline_channel=1,
        validity_channel=3,
    )

    assert torch.equal(former(x), current(x))
