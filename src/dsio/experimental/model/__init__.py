"""Experimental model-side components: compositions, standardization and objectives.

Each block's maturity and evidence is listed in docs/component-warehouse/catalog.md. The
modules ``chain``, ``components`` and ``masking`` hold legacy experimental components under
the pre-1.0 clause in docs/component-admission.md: no compatibility promise, deleted at 1.0
if still unproven.
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
    Jitter,
    MaskedMSE,
    MLP1d,
    NTXent,
    RandomScale,
    VICReg,
    identity_head,
    identity_transform,
    linear_head,
    mae_decoder_head,
    mlp_head,
    no_augmentation,
    simclr_projector_head,
    vicreg_projector_head,
)
from dsio.experimental.model.compositions import MLP, Chain, Stages
from dsio.experimental.model.convolution import DenseConv1d
from dsio.experimental.model.layout import ChannelFirstToTimeMajor, TimeMajorToChannelFirst
from dsio.experimental.model.masked_objective import MaskedObjective
from dsio.experimental.model.masking import (
    CausalMask,
    PatchMask,
    RandomMask,
    SpanMask,
    apply_mask,
)
from dsio.experimental.model.objectives import RootMeanSquaredError, SupervisedObjective
from dsio.experimental.model.standardization import InstanceStandardize, Standardize

__all__ = [
    "MLP",
    "CausalMask",
    "Chain",
    "ChannelFirstToTimeMajor",
    "ComponentChain",
    "Conv1dEncoder",
    "CrossEntropy",
    "DenseConv1d",
    "EmbeddingEncoder",
    "FixedStandardize",
    "IdentityAugmentation",
    "InstanceStandardize",
    "Jitter",
    "LossObjective",
    "MLP1d",
    "MaskedMSE",
    "MaskedObjective",
    "NTXent",
    "PatchMask",
    "RandomMask",
    "RandomScale",
    "RootMeanSquaredError",
    "SpanMask",
    "Stages",
    "Standardize",
    "SupervisedObjective",
    "TimeMajorToChannelFirst",
    "VICReg",
    "apply_mask",
    "export_encoder",
    "identity_head",
    "identity_transform",
    "linear_head",
    "mae_decoder_head",
    "mlp_head",
    "no_augmentation",
    "simclr_projector_head",
    "vicreg_projector_head",
]
