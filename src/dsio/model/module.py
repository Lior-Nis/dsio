"""The component chain, as one LightningModule with one step implementation.

A fixed chain of slots with declared always-present / maybe-present invariants, so every
training paradigm is the same object with different pieces in it.

```
x -> preprocessor? -> transform -> backbone -> head
```

Three changes from the original, each fixing something that cost real time there:

**No stochastic slot, so nothing here can augment a validation batch.** The chain used to
carry two train-only slots, skipped unless ``self.training`` — a runtime flag a validation
loop could get wrong without anything in a config file revealing it. That property now holds
structurally instead: a pretext transform like masking lives on the *dataset*
(:class:`~dsio.dataset.dataset.WindowDataset`), so a training dataset built with ``mask=`` and a
validation dataset built without one is the whole mechanism. There is no flag here to check
and nothing to get wrong on the model side — ``encode`` runs the same chain regardless of
``self.training``.

**One step implementation, not three.** ``_common_step(batch, stage)`` removes the
train/val/test triplication, because the alternative is three near-identical methods that
drift apart.

**Predictions carry their row positions.** The batch dict carries ``row``, so predictions
can be aligned back to the fold that produced them by identity rather than by trusting
DataLoader ordering. This is the same failure class the fold loop refuses — an off-by-one
that scores row *i* against row *j*'s label and looks merely disappointing.

**A loss may report its own diagnostics, from inside the one forward pass this step
already does.** ``self.loss`` sees only ``(prediction, target)`` by contract — but a loss
that also implements ``diagnostics(prediction, target, x) -> dict[str, Tensor]`` gets it
called here and each entry logged under ``{stage}/{name}``. This dispatches on what the
loss object *is* (``getattr(self.loss, "diagnostics", None)``), the same shape every
registry in this codebase already uses to add a capability without every caller needing to
know which concrete type it is talking to — not on which task or method configured it, so
a loss with nothing to add costs one attribute lookup and a loss with something to add
costs no second forward pass to get it.

**One `LightningModule`, no paradigm subclass.** A pretraining run and a supervised one are
both this class: what differs is which backbone/head/loss/transform go into the slots, and
what the dataset behind the loader hands the ``(x, target)`` pair. Contrastive objectives
(SimCLR, VICReg) build their two views at collate time through
:class:`~dsio.dataset.dataset.TwoViewCollate`, so their loss is an ordinary
``(prediction, target)`` loss too (:class:`~dsio.model.components.NTXent`,
:class:`~dsio.model.components.VICReg`) and needs no module subclass.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from lightning import LightningModule
from torch import nn

Stage = Literal["train", "val", "test"]


class ComponentError(ValueError):
    """Raised when the component chain is missing a required piece or is misassembled."""


class DsioModule(LightningModule):
    """A model assembled from registered components, trained by one shared step.

    Required slots — ``transform``, ``backbone``, ``head``, ``loss`` — are never ``None``.
    ``transform`` defaults to identity rather than being optional, so the chain has one
    shape and ``forward`` needs no branch for it.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        head: nn.Module,
        loss: nn.Module,
        transform: nn.Module | None = None,
        preprocessor: nn.Module | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        target_key: str = "y",
    ) -> None:
        super().__init__()
        for name, component in (("backbone", backbone), ("head", head), ("loss", loss)):
            if component is None:
                raise ComponentError(f"{name} is required and may not be None")
        self.transform = transform if transform is not None else nn.Identity()
        self.backbone = backbone
        self.head = head
        self.loss = loss
        self.preprocessor = preprocessor

        self.lr = lr
        self.weight_decay = weight_decay
        self.target_key = target_key
        self.save_hyperparameters(ignore=["backbone", "head", "loss", "transform", "preprocessor"])

    # --- the chain --------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain up to and including the backbone, returning features.

        Separate from :meth:`forward` because SSL callbacks inspect backbone features and
        pretrained runs export them for downstream tasks. A head is a task's opinion about
        features; the features themselves outlive it.
        """
        if self.preprocessor is not None:
            x = self.preprocessor(x)
        x = self.transform(x)
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

    # --- one step, three stages -------------------------------------------------

    def _common_step(self, batch: dict[str, Any], stage: Stage) -> torch.Tensor:
        """The single implementation every stage shares."""
        x = batch["x"]
        target = batch[self.target_key]
        prediction = self(x)
        value = self.loss(prediction, target)
        if value.ndim > 0:
            value = value.mean()
        self.log(
            f"{stage}/loss",
            value,
            batch_size=x.shape[0],
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage == "val",
        )
        diagnostics = getattr(self.loss, "diagnostics", None)
        if diagnostics is not None:
            for name, diagnostic in diagnostics(prediction, target, x).items():
                self.log(f"{stage}/{name}", diagnostic, batch_size=x.shape[0], on_epoch=True)
        return value

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "test")

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, torch.Tensor]:
        """Return predictions *with* the row positions they belong to.

        Returning bare logits would make alignment a property of DataLoader ordering, which
        is true today and silently untrue the moment anyone shuffles a prediction loader or
        uses a sampler. Carrying the position makes it checkable instead.
        """
        prediction = self(batch["x"])
        return {
            "row": batch["row"],
            "prediction": prediction.detach(),
            self.target_key: batch[self.target_key].detach(),
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


def export_encoder(module: DsioModule) -> dict[str, torch.Tensor]:
    """The weights worth keeping from a trained module: everything the chain needs to
    produce features, none of what only the pretext objective needed.

    Excludes ``head``. A decoder trained to reconstruct masked spans has no meaning outside
    the pretext task, and shipping it invites someone to load it as though it were part of
    the model — the same is true of a contrastive projector, which exists only to give a
    loss a space to compare views in.

    A free function because exportability is a policy about which slots to keep, callable on
    any :class:`DsioModule` regardless of what trained it.
    """
    state: dict[str, torch.Tensor] = {}
    for name in ("preprocessor", "transform", "backbone"):
        component = getattr(module, name, None)
        if isinstance(component, nn.Module):
            for key, value in component.state_dict().items():
                state[f"{name}.{key}"] = value
    return state
