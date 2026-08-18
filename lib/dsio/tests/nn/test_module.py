"""Component chain invariants, including the ones FORGE documented but did not enforce."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from torch import nn  # noqa: E402

from dsio.nn import (  # noqa: E402
    AUGMENTORS,
    BACKBONES,
    HEADS,
    LOSSES,
    ComponentError,
    Conv1dEncoder,
    CrossEntropy,
    DsioModule,
    InstanceStandardize,
    Jitter,
    MLP1d,
    RandomScale,
)


def tiny_module(**overrides) -> DsioModule:  # type: ignore[no-untyped-def]
    defaults = dict(
        backbone=Conv1dEncoder(channels=2, hidden=4, out_dim=8, depth=1),
        head=nn.Linear(8, 2),
        loss=CrossEntropy(),
    )
    return DsioModule(**{**defaults, **overrides})


@pytest.fixture
def batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 2, 64)


# --- required and optional slots ----------------------------------------------------


def test_required_components_may_not_be_none() -> None:
    with pytest.raises(ComponentError, match="backbone is required"):
        DsioModule(backbone=None, head=nn.Linear(8, 2), loss=CrossEntropy())  # type: ignore[arg-type]


def test_transform_defaults_to_identity_rather_than_none() -> None:
    """One chain shape means forward needs no branch, and no slot can be forgotten."""
    module = tiny_module()
    assert isinstance(module.transform, nn.Identity)
    assert module.preprocessor is None and module.augmentor is None


def test_forward_is_encode_plus_head(batch: torch.Tensor) -> None:
    module = tiny_module().eval()
    with torch.no_grad():
        assert torch.allclose(module(batch), module.head(module.encode(batch)))


def test_encode_survives_without_a_head(batch: torch.Tensor) -> None:
    """SSL pretraining hands features to a probe; a head is a task's opinion about them."""
    module = tiny_module(head=nn.Identity()).eval()
    with torch.no_grad():
        assert module.encode(batch).shape == (4, 8)


# --- the enforced improvement over FORGE --------------------------------------------


def test_augmentation_is_skipped_outside_training(batch: torch.Tensor) -> None:
    """The bug this exists to prevent: augmenting a validation batch.

    It makes the metric noisier and irreproducible while looking entirely normal, and
    nothing in a config file or a loss curve reveals it. FORGE's chain applies whatever is
    configured whenever it is called.
    """
    module = tiny_module(augmentor=Jitter(sigma=1.0)).eval()
    with torch.no_grad():
        first, second = module.encode(batch), module.encode(batch)
    assert torch.allclose(first, second), "eval mode must be deterministic"


def test_augmentation_does_apply_in_training(batch: torch.Tensor) -> None:
    """The other half — a skip that always skips would pass the test above trivially."""
    module = tiny_module(augmentor=Jitter(sigma=1.0)).train()
    torch.manual_seed(0)
    first = module.encode(batch)
    second = module.encode(batch)
    assert not torch.allclose(first, second)


def test_spectral_augmentation_obeys_the_same_rule(batch: torch.Tensor) -> None:
    module = tiny_module(spectral_augmentor=Jitter(sigma=1.0)).eval()
    with torch.no_grad():
        assert torch.allclose(module.encode(batch), module.encode(batch))


def test_chain_order_puts_the_transform_after_augmentation(batch: torch.Tensor) -> None:
    """Augmenting after normalising would undo the normalisation it was measured against."""
    seen: list[str] = []

    class Recorder(nn.Module):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.label = label

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seen.append(self.label)
            return x

    module = tiny_module(
        preprocessor=Recorder("preprocessor"),
        augmentor=Recorder("augmentor"),
        transform=Recorder("transform"),
        spectral_augmentor=Recorder("spectral"),
    ).train()
    module.encode(batch)
    assert seen == ["preprocessor", "augmentor", "transform", "spectral"]


# --- one step, three stages ---------------------------------------------------------


def test_every_stage_shares_one_step_implementation(batch: torch.Tensor) -> None:
    """FORGE's _common_step, kept: three near-identical methods are three that drift."""
    module = tiny_module()
    payload = {"x": batch, "y": torch.tensor([0, 1, 0, 1])}
    for method in (module.training_step, module.validation_step, module.test_step):
        value = method(payload, 0)
        assert value.ndim == 0 and torch.isfinite(value)


def test_predict_step_reports_the_rows_it_predicted(batch: torch.Tensor) -> None:
    """Alignment by identity, not by trusting loader ordering."""
    module = tiny_module().eval()
    rows = torch.tensor([7, 3, 11, 5])
    out = module.predict_step({"x": batch, "y": torch.zeros(4).long(), "row": rows}, 0)
    assert torch.equal(out["row"], rows)
    assert out["prediction"].shape == (4, 2)


# --- components ---------------------------------------------------------------------


def test_components_reject_a_missing_channel_axis() -> None:
    """A silent broadcast between channels-first and channels-last trains and is wrong."""
    flat = torch.randn(4, 64)
    for component in (Conv1dEncoder(channels=2, depth=1), InstanceStandardize(), Jitter()):
        with pytest.raises(ValueError, match=r"\[batch, channels, time\]"):
            component(flat)


def test_conv_backbone_accepts_any_window_length() -> None:
    """Pooling over time is what keeps a backbone independent of the view it was built for.

    A backbone whose parameter count depends on window length forces a retrain for every
    change to an index — exactly the coupling the view layer removed.
    """
    encoder = Conv1dEncoder(channels=2, hidden=4, out_dim=8, depth=1)
    assert encoder(torch.randn(2, 2, 64)).shape == (2, 8)
    assert encoder(torch.randn(2, 2, 250)).shape == (2, 8)


def test_mlp_backbone_is_tied_to_its_length() -> None:
    """The honest contrast: MLP1d takes length because it genuinely depends on it."""
    encoder = MLP1d(channels=2, length=64, hidden=8, out_dim=8)
    assert encoder(torch.randn(2, 2, 64)).shape == (2, 8)
    with pytest.raises(RuntimeError):
        encoder(torch.randn(2, 2, 128))


def test_instance_standardize_cannot_leak_across_a_split() -> None:
    """Its statistics come from the window itself, so there is nothing to leak."""
    standardize = InstanceStandardize()
    x = torch.randn(4, 2, 128) * 5 + 3
    out = standardize(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(4, 2), atol=1e-5)
    assert torch.allclose(out.std(dim=-1), torch.ones(4, 2), atol=1e-2)


def test_cross_entropy_refuses_soft_targets_without_a_threshold() -> None:
    """Silently binarising at 0.5 would make a data decision inside a loss function."""
    with pytest.raises(ValueError, match="no threshold"):
        CrossEntropy()(torch.randn(4, 2), torch.tensor([0.2, 0.7, 0.9, 0.1]))


def test_cross_entropy_accepts_soft_targets_with_one() -> None:
    value = CrossEntropy(threshold=0.5)(torch.randn(4, 2), torch.tensor([0.2, 0.7, 0.9, 0.1]))
    assert torch.isfinite(value)


def test_random_scale_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        RandomScale(low=1.5, high=0.5)


def test_registries_expose_the_builtins() -> None:
    assert {"mlp1d", "conv1d"} <= set(BACKBONES.names())
    assert {"linear", "mlp", "identity"} <= set(HEADS.names())
    assert {"cross_entropy", "bce", "mse"} <= set(LOSSES.names())
    assert {"jitter", "random_scale", "none"} <= set(AUGMENTORS.names())


def test_an_unknown_component_suggests_a_close_name() -> None:
    with pytest.raises(KeyError, match="did you mean"):
        BACKBONES.get("conv1D")
