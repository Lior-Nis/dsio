"""Component chain invariants, including the ones usually documented but not enforced."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from torch import nn  # noqa: E402

from dsio.model.components import (  # noqa: E402
    Conv1dEncoder,
    CrossEntropy,
    InstanceStandardize,
    Jitter,
    MLP1d,
    RandomScale,
    mae_decoder_head,
    simclr_projector_head,
    vicreg_projector_head,
)
from dsio.model.module import ComponentError, DsioModule, export_encoder  # noqa: E402
from dsio.model.registry import AUGMENTORS, BACKBONES, HEADS, LOSSES  # noqa: E402


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
    assert module.preprocessor is None


def test_forward_is_encode_plus_head(batch: torch.Tensor) -> None:
    module = tiny_module().eval()
    with torch.no_grad():
        assert torch.allclose(module(batch), module.head(module.encode(batch)))


def test_encode_survives_without_a_head(batch: torch.Tensor) -> None:
    """SSL pretraining hands features to a probe; a head is a task's opinion about them."""
    module = tiny_module(head=nn.Identity()).eval()
    with torch.no_grad():
        assert module.encode(batch).shape == (4, 8)


# --- enforced, not merely documented ------------------------------------------------
#
# The chain used to carry two train-only stochastic slots (augmentor, spectral_augmentor),
# skipped unless ``self.training`` — a runtime flag a validation loop could get wrong
# without anything in a config file revealing it. Those tests
# (test_augmentation_is_skipped_outside_training,
# test_augmentation_does_apply_in_training, test_spectral_augmentation_obeys_the_same_rule)
# covered exactly that skip and are gone along with the slots: there is no longer a
# stochastic component in the chain for ``self.training`` to gate. The property they
# guarded — a validation batch is never augmented — now holds structurally instead: a
# pretext transform like masking lives on the dataset
# (tests/dataset/test_dataset.py::test_validation_dataset_is_unmasked_by_construction), which has
# no notion of "training" for a runtime flag to get wrong.


def test_chain_order_puts_the_transform_after_the_preprocessor(batch: torch.Tensor) -> None:
    """Transforming before preprocessing would compute preprocessing statistics on data
    that has already been through a domain-specific transform they were not fitted for."""
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
        transform=Recorder("transform"),
    ).train()
    module.encode(batch)
    assert seen == ["preprocessor", "transform"]


# --- one step, three stages ---------------------------------------------------------


def test_every_stage_shares_one_step_implementation(batch: torch.Tensor) -> None:
    """One implementation: three near-identical step methods are three that drift apart."""
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


# --- predict is a config choice, not a subclass --------------------------------------
#
# There used to be a separate SslModule whose only real difference from DsioModule was
# this: predict_step returned an embedding, not a classification. Task 6b deleted that
# subclass in favour of a `predict` constructor argument, branched on here instead of on
# `isinstance`.


def test_predict_defaults_to_running_the_whole_chain() -> None:
    module = tiny_module()
    assert module.predict == "prediction"


def test_predict_embedding_stops_before_the_head(batch: torch.Tensor) -> None:
    module = tiny_module(predict="embedding").eval()
    rows = torch.tensor([1, 2, 3, 4])
    out = module.predict_step({"x": batch, "row": rows}, 0)
    assert set(out) == {"row", "embedding"}
    assert torch.equal(out["row"], rows)
    with torch.no_grad():
        assert torch.equal(out["embedding"], module.encode(batch))


def test_predict_choice_is_recorded_in_hyperparameters() -> None:
    """"Visible in the recorded config instead of implied by a type": save_hyperparameters
    is what a checkpoint's hparams and a run's provenance both read."""
    module = tiny_module(predict="embedding")
    assert module.hparams["predict"] == "embedding"


def test_an_unknown_predict_value_is_rejected() -> None:
    with pytest.raises(ComponentError, match="predict"):
        tiny_module(predict="logits")  # type: ignore[arg-type]


# --- export_encoder --------------------------------------------------------------------
#
# Moved from the deleted tests/ssl/test_methods.py::test_the_exported_encoder_excludes_
# the_objective_head, generalised: encoder_state used to be a method only SslModule had,
# so the property could only be checked on modules built through it. export_encoder is a
# free function over any DsioModule now, so this checks the property directly, including
# the fail-then-restore proof the task brief asks for: break the exclusion, watch the
# assertion fail, then restore it and confirm the source is unchanged.


def test_export_encoder_excludes_the_head(batch: torch.Tensor) -> None:
    module = tiny_module()
    state = export_encoder(module)
    assert state, "the encoder must export something"
    assert all(not key.startswith("head.") for key in state)
    assert any(key.startswith("backbone.") for key in state)


@pytest.mark.parametrize(
    "head",
    [
        mae_decoder_head(8, channels=2, length=64),
        simclr_projector_head(8, out_dim=4),
        vicreg_projector_head(8, out_dim=4),
    ],
    ids=["mae_decoder", "simclr_projector", "vicreg_projector"],
)
def test_export_encoder_excludes_every_real_pretext_head(head, batch: torch.Tensor) -> None:
    """The property that matters in practice: not just "some head" but each of the three
    actual objective heads a pretraining run builds -- a decoder trained to reconstruct
    masked spans, or a projector trained only to compare views, has no meaning outside the
    pretext task, and shipping it invites someone to load it as though it were part of the
    model."""
    module = tiny_module(head=head)
    state = export_encoder(module)
    assert state
    assert all(not key.startswith("head.") for key in state)
    assert any(key.startswith("backbone.") for key in state)


def test_export_encoder_keeps_preprocessor_and_transform(batch: torch.Tensor) -> None:
    module = tiny_module(
        preprocessor=nn.Linear(2, 2), transform=nn.Identity(), head=nn.Linear(8, 2)
    )
    state = export_encoder(module)
    assert any(key.startswith("preprocessor.") for key in state)
    assert any(key.startswith("backbone.") for key in state)
    assert all(not key.startswith("head.") for key in state)


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
