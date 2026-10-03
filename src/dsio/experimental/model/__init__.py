"""Model-side components not yet proven by real downstream use.

Legacy experimental under the pre-1.0 clause in docs/component-admission.md: these
components have no real downstream use yet, carry no compatibility promise, and are
deleted at 1.0 if still unproven.
"""

from dsio.experimental.model.chain import (
    ComponentChain,
    LossObjective,
    export_encoder,
)
from dsio.experimental.model.components import (
    Conv1dEncoder,
    CrossEntropy,
    EmbeddingEncoder,
    FixedStandardize,
    IdentityAugmentation,
    InstanceStandardize,
    Jitter,
    MaskedMSE,
    MLP1d,
    NTXent,
    RandomScale,
    VICReg,
    bce_loss,
    identity_head,
    identity_transform,
    linear_head,
    mae_decoder_head,
    mlp_head,
    mse_loss,
    no_augmentation,
    simclr_projector_head,
    vicreg_projector_head,
)
from dsio.experimental.model.compositions import MLP, Chain, Stages
from dsio.experimental.model.masking import (
    CausalMask,
    PatchMask,
    RandomMask,
    SpanMask,
    apply_mask,
)
from dsio.experimental.model.standardization import Standardize

__all__ = [
    "MLP",
    "CausalMask",
    "Chain",
    "ComponentChain",
    "Conv1dEncoder",
    "CrossEntropy",
    "EmbeddingEncoder",
    "FixedStandardize",
    "IdentityAugmentation",
    "InstanceStandardize",
    "Jitter",
    "LossObjective",
    "MLP1d",
    "MaskedMSE",
    "NTXent",
    "PatchMask",
    "RandomMask",
    "RandomScale",
    "SpanMask",
    "Stages",
    "Standardize",
    "VICReg",
    "apply_mask",
    "bce_loss",
    "export_encoder",
    "identity_head",
    "identity_transform",
    "linear_head",
    "mae_decoder_head",
    "mlp_head",
    "mse_loss",
    "no_augmentation",
    "simclr_projector_head",
    "vicreg_projector_head",
]
