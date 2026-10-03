from __future__ import annotations

import pytest
import torch
from torch import nn

from dsio.config.components import resolve_component
from dsio.experimental.model.compositions import Chain
from dsio.experimental.model.tokens import EmbeddingEncoder


def _input() -> torch.Tensor:
    return torch.tensor(
        [
            [[1, 1], [2, 1], [3, 1], [0, 0]],
            [[4, 1], [0, 1], [99, 0], [-7, 0]],
        ],
        dtype=torch.int64,
    )


def test_masked_mean_uses_only_declared_valid_tokens() -> None:
    encoder = EmbeddingEncoder(vocab_size=8, embed_dim=2)
    with torch.no_grad():
        encoder.embedding.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(8, 2))
    result = encoder(_input())

    expected = torch.stack(
        (
            encoder.embedding.weight[torch.tensor([1, 2, 3])].mean(dim=0),
            encoder.embedding.weight[torch.tensor([4, 0])].mean(dim=0),
        )
    )
    assert torch.equal(result, expected)


def test_arbitrary_padding_values_cannot_change_representation() -> None:
    encoder = EmbeddingEncoder(vocab_size=8, embed_dim=3)
    original = _input()
    changed = original.clone()
    changed[:, :, 0][~changed[:, :, 1].bool()] = torch.tensor([-999, 999, -7])

    assert torch.equal(encoder(original), encoder(changed))


def test_valid_token_zero_is_not_treated_as_padding() -> None:
    encoder = EmbeddingEncoder(vocab_size=8, embed_dim=1)
    with torch.no_grad():
        encoder.embedding.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(8, 1))
    x = torch.tensor([[[0, 1], [4, 1], [7, 0]]])

    assert torch.equal(encoder(x), torch.tensor([[2.0]]))


@pytest.mark.parametrize(
    ("x", "message"),
    [
        (torch.ones(2, 3, dtype=torch.int64), "rank 3"),
        (torch.ones(2, 3, 1, dtype=torch.int64), "channel axis 2 expected 2"),
        (torch.ones(2, 0, 2, dtype=torch.int64), "token axis 1 must be non-empty"),
        (torch.ones(0, 3, 2, dtype=torch.int64), "batch axis 0 must be non-empty"),
        (torch.ones(2, 3, 2), "integer tensor"),
        (torch.ones(2, 3, 2, dtype=torch.bool), "integer tensor"),
    ],
)
def test_input_contract_errors_are_named(x: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EmbeddingEncoder(vocab_size=8)(x)


@pytest.mark.parametrize(
    ("x", "message"),
    [
        (torch.tensor([[[1, 2], [2, 1]]]), "validity channel must contain only zero or one"),
        (torch.tensor([[[1, 0], [2, 0]]]), "at least one valid token"),
        (torch.tensor([[[8, 1], [2, 1]]]), "valid token ids outside configured vocabulary"),
        (torch.tensor([[[-1, 1], [2, 1]]]), "valid token ids outside configured vocabulary"),
    ],
)
def test_value_contract_errors_are_named(x: torch.Tensor, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        EmbeddingEncoder(vocab_size=8)(x)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"vocab_size": 0}, "vocab_size"),
        ({"vocab_size": 8, "embed_dim": 0}, "embed_dim"),
        ({"vocab_size": 8, "token_channel": 1, "validity_channel": 1}, "distinct"),
        ({"vocab_size": 8, "token_channel": 2}, "channel indices"),
        ({"vocab_size": 8, "token_channel": 0.0}, "channel indices"),
        ({"vocab_size": 8, "safe_token_id": 8}, "safe_token_id"),
    ],
)
def test_configuration_errors_are_named(parameters: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EmbeddingEncoder(**parameters)


def test_component_config_composes_with_a_native_head_deterministically() -> None:
    config = {
        "reference": "dsio.experimental.model.compositions:Chain",
        "parameters": {
            "backbone": {
                "reference": "dsio.experimental.model.tokens:EmbeddingEncoder",
                "parameters": {"vocab_size": 8, "embed_dim": 3},
            },
            "head": {
                "reference": "torch.nn:Linear",
                "parameters": {"in_features": 3, "out_features": 2},
            },
        },
    }
    torch.manual_seed(4)
    first = resolve_component(config, expected=Chain)
    torch.manual_seed(4)
    second = resolve_component(config, expected=Chain)

    assert torch.equal(first(_input()), second(_input()))
    assert first(_input()).shape == (2, 2)


def test_composed_model_is_padding_invariant_and_fullgraph_compilable() -> None:
    model = Chain(
        backbone={
            "reference": "dsio.experimental.model.tokens:EmbeddingEncoder",
            "parameters": {"vocab_size": 8, "embed_dim": 3},
        },
        head={
            "reference": "torch.nn:Linear",
            "parameters": {"in_features": 3, "out_features": 2},
        },
    )
    original = _input()
    changed = original.clone()
    changed[:, :, 0][~changed[:, :, 1].bool()] = torch.tensor([-999, 999, -7])
    compiled = torch.compile(model, backend="eager", fullgraph=True)

    assert torch.equal(compiled(original), compiled(changed))


def test_seeded_composition_matches_the_former_model_exactly() -> None:
    class Former(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(8, 3, padding_idx=0)
            self.output = nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tokens = x[:, :, 0]
            valid = x[:, :, 1].bool()
            embedded = self.embedding(tokens)
            pooled = (embedded * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
            return self.output(pooled)

    clean = _input().clone()
    clean[~clean[:, :, 1].bool(), 0] = 0
    torch.manual_seed(9)
    former = Former()
    torch.manual_seed(9)
    current = Chain(
        backbone={
            "reference": "dsio.experimental.model.tokens:EmbeddingEncoder",
            "parameters": {"vocab_size": 8, "embed_dim": 3},
        },
        head={
            "reference": "torch.nn:Linear",
            "parameters": {"in_features": 3, "out_features": 2},
        },
    )

    assert torch.equal(former(clean), current(clean))


def test_gradient_reaches_only_valid_token_rows() -> None:
    encoder = EmbeddingEncoder(vocab_size=8, embed_dim=3)
    output = encoder(torch.tensor([[[1, 1], [2, 1], [3, 0]]])).sum()
    output.backward()

    assert encoder.embedding.weight.grad is not None
    assert encoder.embedding.weight.grad[1].abs().sum() > 0
    assert encoder.embedding.weight.grad[2].abs().sum() > 0
    assert encoder.embedding.weight.grad[3].abs().sum() == 0


def test_safe_row_can_learn_when_it_is_a_valid_token() -> None:
    encoder = EmbeddingEncoder(vocab_size=8, embed_dim=3, safe_token_id=0)
    encoder(torch.tensor([[[0, 1], [7, 0]]])).sum().backward()

    assert encoder.embedding.weight.grad is not None
    assert encoder.embedding.weight.grad[0].abs().sum() > 0
    assert encoder.embedding.weight.grad[7].abs().sum() == 0


@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
def test_integer_input_dtypes_reach_the_lookup(dtype: torch.dtype) -> None:
    encoder = EmbeddingEncoder(vocab_size=8, embed_dim=3)
    x = torch.tensor([[[1, 1], [2, 1], [3, 0]]], dtype=dtype)

    assert encoder(x).shape == (1, 3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_pooling_accumulates_without_overflow(dtype: torch.dtype) -> None:
    encoder = EmbeddingEncoder(vocab_size=8, embed_dim=2).to(dtype=dtype)
    with torch.no_grad():
        encoder.embedding.weight.fill_(1000)
    x = torch.tensor([[[1, 1]]] * 100, dtype=torch.int64).transpose(0, 1)

    result = encoder(x)

    assert result.dtype == dtype
    assert torch.isfinite(result).all()
    assert torch.equal(result, torch.full_like(result, 1000))


def test_validation_supports_fullgraph_compilation() -> None:
    encoder = EmbeddingEncoder(vocab_size=8, embed_dim=3)
    compiled = torch.compile(encoder, backend="eager", fullgraph=True)
    assert torch.equal(compiled(_input()), encoder(_input()))
