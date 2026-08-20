"""The component chain, as one LightningModule with one step implementation.

A fixed chain of slots with declared always-present / maybe-present invariants, so every
training paradigm is the same object with different pieces in it.

```
x -> preprocessor? -> augmentor? -> transform -> spectral_augmentor? -> backbone -> head
```

Three changes from the original, each fixing something that cost real time there:

**Augmentation is training-only, enforced rather than documented.** A chain that applies
whatever is configured whenever it is called will augment during validation, which makes the
metric noisy *and* irreproducible while looking entirely normal — and it is invisible in a
config file. The two stochastic slots are skipped unless ``self.training``.

**One step implementation, not three.** ``_common_step(batch, stage)`` removes the
train/val/test triplication, because the alternative is three near-identical methods that
drift apart.

**Predictions carry their row positions.** The batch dict carries ``row``, so predictions
can be aligned back to the fold that produced them by identity rather than by trusting
DataLoader ordering. This is the same failure class the fold loop refuses — an off-by-one
that scores row *i* against row *j*'s label and looks merely disappointing.
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
        augmentor: nn.Module | None = None,
        spectral_augmentor: nn.Module | None = None,
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
        self.augmentor = augmentor
        self.spectral_augmentor = spectral_augmentor

        self.lr = lr
        self.weight_decay = weight_decay
        self.target_key = target_key
        self.save_hyperparameters(ignore=[
            "backbone", "head", "loss", "transform",
            "preprocessor", "augmentor", "spectral_augmentor",
        ])

    # --- the chain --------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain up to and including the backbone, returning features.

        Separate from :meth:`forward` because this is what SSL pretraining hands to a
        downstream probe, and what an embedding-extraction stage caches. A head is a task's
        opinion about features; the features themselves outlive it.
        """
        if self.preprocessor is not None:
            x = self.preprocessor(x)
        # Stochastic slots are skipped outside training. Augmenting a validation batch
        # produces a metric that is both noisier and unreproducible, and nothing about the
        # config or the loss curve reveals it.
        if self.augmentor is not None and self.training:
            x = self.augmentor(x)
        x = self.transform(x)
        if self.spectral_augmentor is not None and self.training:
            x = self.spectral_augmentor(x)
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
