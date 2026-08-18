"""Self-supervised pretraining: pretext objectives, masking, probes and label budgets.

The workflow this package exists for is pretrain on a large unlabelled corpus, probe or
finetune on a small labelled one, and evaluate on a third held-out cohort — with the three
having different schemas and different split logic. Unlabelled data typically outnumbers
labelled by two orders of magnitude, which is the normal situation and the reason SSL is
first-class here rather than an add-on.
"""

from dsio.ssl.budget import (
    BudgetError,
    BudgetSelection,
    assert_nested,
    budget_curve,
    group_rates,
    select_groups,
)
from dsio.ssl.masking import (
    MASKS,
    CausalMask,
    PatchMask,
    RandomMask,
    SpanMask,
    apply_mask,
    mask_strategy,
)
from dsio.ssl.methods import METHODS, MaskedReconstruction, SimCLR, SslMethod, VICReg, ssl_method
from dsio.ssl.module import SslModule
from dsio.ssl.probe import OnlineProbe, RankMeMonitor, embed, rankme

__all__ = [
    "MASKS",
    "METHODS",
    "BudgetError",
    "BudgetSelection",
    "CausalMask",
    "MaskedReconstruction",
    "OnlineProbe",
    "PatchMask",
    "RandomMask",
    "RankMeMonitor",
    "SimCLR",
    "SpanMask",
    "SslMethod",
    "SslModule",
    "VICReg",
    "apply_mask",
    "assert_nested",
    "budget_curve",
    "embed",
    "group_rates",
    "mask_strategy",
    "rankme",
    "select_groups",
    "ssl_method",
]
