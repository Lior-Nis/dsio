"""Signal blocks preserve explicit layouts, extents, and padding semantics."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dsio.config.components import resolve_component
from dsio.experimental.model import (
    ChannelFirstToTimeMajor,
    DenseConv1d,
    InstanceStandardize,
    TimeMajorToChannelFirst,
)


def test_layout_adapters_round_trip_non_square_contiguous_signals() -> None:
    source = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
    channel_first = TimeMajorToChannelFirst(channels=3, time=5)(source)
    assert channel_first.shape == (2, 3, 5)
    assert channel_first.is_contiguous()
    restored = ChannelFirstToTimeMajor(channels=3, time=5)(channel_first)
    assert restored.shape == source.shape
    assert restored.is_contiguous()
    assert torch.equal(restored, source)


def test_layout_adapters_name_rank_channel_and_time_mismatches() -> None:
    with pytest.raises(ValueError, match=r"time-major.*rank 3.*\(2, 3\)"):
        TimeMajorToChannelFirst(channels=3)(torch.ones(2, 3))
    with pytest.raises(ValueError, match=r"channel axis 2.*expected 3.*got 4"):
        TimeMajorToChannelFirst(channels=3)(torch.ones(2, 5, 4))
    with pytest.raises(ValueError, match=r"time axis 1.*expected 5.*got 6"):
        TimeMajorToChannelFirst(channels=3, time=5)(torch.ones(2, 6, 3))
    with pytest.raises(ValueError, match=r"channel axis 1.*expected 3.*got 4"):
        ChannelFirstToTimeMajor(channels=3)(torch.ones(2, 4, 5))
    with pytest.raises(ValueError, match="positive integer"):
        TimeMajorToChannelFirst(channels=0)


def test_dense_conv_matches_the_seeded_consumer_architecture_exactly() -> None:
    torch.manual_seed(7)
    reference = nn.Sequential(
        nn.Conv1d(3, 16, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv1d(16, 3, kernel_size=5, padding=2),
    )
    torch.manual_seed(7)
    dense = DenseConv1d(channels=3, hidden=16, output=3, kernel_size=5)
    x = torch.randn(2, 3, 11)
    for expected, actual in zip(reference.parameters(), dense.parameters(), strict=True):
        assert torch.equal(actual, expected)
    assert torch.equal(dense(x), reference(x))
    assert dense(torch.ones(2, 3, 7, dtype=torch.float64)).shape == (2, 3, 7)
    with pytest.raises(ValueError, match=r"channel axis 1.*expected 3.*got 4"):
        dense(torch.ones(2, 4, 7))
    with pytest.raises(ValueError, match="odd positive integer"):
        DenseConv1d(channels=3, hidden=16, output=3, kernel_size=4)
    with pytest.raises(ValueError, match="time axis 2 must be non-empty"):
        dense(torch.ones(2, 3, 0))


def test_instance_standardization_uses_population_statistics_and_validity_channel() -> None:
    source = torch.tensor(
        [
            [[1.0, 3.0, 999.0], [2.0, 6.0, float("nan")], [1.0, 1.0, 0.0]],
            [[4.0, 4.0, -999.0], [5.0, 7.0, float("inf")], [1.0, 1.0, 0.0]],
        ]
    )
    original = source.clone()
    standardize = InstanceStandardize(observed_channel=2)
    result = standardize(source)
    assert result.shape == source.shape
    assert torch.allclose(result[0, :2, :2], torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]))
    assert torch.equal(result[1, 0], torch.zeros(3))
    assert torch.allclose(result[1, 1, :2], torch.tensor([-1.0, 1.0]))
    assert torch.equal(result[:, :2, 2], torch.zeros(2, 2))
    assert torch.equal(result[:, 2], source[:, 2])
    assert torch.allclose(source, original, equal_nan=True)

    changed = source.clone()
    changed[:, :2, 2] = torch.tensor([[-1e20], [1e20]])
    assert torch.equal(standardize(changed), result)


def test_dense_conv_real_outputs_do_not_depend_on_cobatch_padding() -> None:
    torch.manual_seed(19)
    model = nn.Sequential(
        InstanceStandardize(observed_channel=3),
        DenseConv1d(channels=3, hidden=16, output=3, observed_channel=3),
    )
    short = torch.tensor([[[1.0, 3.0, 2.0], [2.0, 5.0, 4.0], [4.0, 7.0, 8.0]]]).transpose(1, 2)
    alone = torch.cat((short, torch.ones(1, 1, 3)), dim=1)
    padded = torch.zeros(2, 4, 7)
    padded[0, :3, :3] = short
    padded[0, 3, :3] = 1
    padded[1, :3] = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    padded[1, 3] = 1

    assert torch.allclose(model(alone), model(padded)[:1, :, :3], rtol=1e-6, atol=1e-6)
    assert torch.equal(model(padded)[0, :, 3:], torch.zeros(3, 4))


def test_instance_standardization_is_fullgraph_compilable() -> None:
    compiled = torch.compile(
        InstanceStandardize(observed_channel=1),
        backend="eager",
        fullgraph=True,
    )
    source = torch.tensor([[[1.0, 3.0, 0.0], [1.0, 1.0, 0.0]]])
    assert torch.equal(compiled(source), InstanceStandardize(observed_channel=1)(source))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_instance_standardization_keeps_tiny_eps_zero_variance_finite(
    dtype: torch.dtype,
) -> None:
    source = torch.full((2, 3, 5), 65_000.0, dtype=dtype)
    result = InstanceStandardize(eps=1e-100)(source)
    assert result.dtype == dtype
    assert torch.equal(result, torch.zeros_like(source))


def test_instance_standardization_casts_integer_inputs_and_preserves_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.tensor([[[1, 2, 3]]], device=device)
    result = InstanceStandardize()(source)
    assert result.dtype == torch.float32
    assert result.device == source.device
    assert torch.allclose(result.mean(dim=-1), torch.zeros(1, 1, device=device))


def test_instance_standardization_validates_observations_and_configuration() -> None:
    stage = InstanceStandardize(observed_channel=-1)
    with pytest.raises(ValueError, match="rank 3"):
        stage(torch.ones(2, 3))
    with pytest.raises(ValueError, match="observed_channel"):
        stage(torch.ones(2, 1, 3))
    with pytest.raises(RuntimeError, match="zero or one"):
        stage(torch.tensor([[[1.0, 2.0], [1.0, 0.5]]]))
    with pytest.raises(RuntimeError, match="at least one observed timestep"):
        stage(torch.tensor([[[1.0, 2.0], [0.0, 0.0]]]))
    with pytest.raises(RuntimeError, match="observed values must be finite"):
        stage(torch.tensor([[[float("nan"), 2.0], [1.0, 1.0]]]))
    with pytest.raises(ValueError, match="finite and positive"):
        InstanceStandardize(eps=0)


def test_signal_blocks_resolve_as_canonical_components() -> None:
    config = {
        "reference": "dsio.experimental.model.compositions:Stages",
        "parameters": {
            "stages": [
                {
                    "reference": "dsio.experimental.model.layout:TimeMajorToChannelFirst",
                    "parameters": {"channels": 4},
                },
                {
                    "reference": "dsio.experimental.model.standardization:InstanceStandardize",
                    "parameters": {"observed_channel": 3},
                },
                {
                    "reference": "dsio.experimental.model.convolution:DenseConv1d",
                    "parameters": {
                        "channels": 3,
                        "hidden": 5,
                        "output": 2,
                        "observed_channel": 3,
                    },
                },
                {
                    "reference": "dsio.experimental.model.layout:ChannelFirstToTimeMajor",
                    "parameters": {"channels": 2},
                },
            ]
        },
    }
    model = resolve_component(config, expected=nn.Module)
    x = torch.tensor([[[1.0, 2.0, 3.0, 1.0], [2.0, 4.0, 6.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])
    assert model(x).shape == (1, 3, 2)


@pytest.mark.parametrize("time", [1, 4, 11])
def test_layout_adapters_accept_variable_time_extents(time: int) -> None:
    source = torch.arange(2 * time * 3).reshape(2, time, 3)
    channel_first = TimeMajorToChannelFirst(channels=3)(source)
    assert torch.equal(ChannelFirstToTimeMajor(channels=3)(channel_first), source)


def test_inverse_layout_adapter_names_rank_and_time_failures() -> None:
    adapter = ChannelFirstToTimeMajor(channels=3, time=5)
    with pytest.raises(ValueError, match=r"channel-first.*rank 3"):
        adapter(torch.ones(3, 5))
    with pytest.raises(ValueError, match=r"time axis 2.*expected 5.*got 4"):
        adapter(torch.ones(2, 3, 4))
