"""The pretraining module: the shared chain, driven by a pretext objective.

Reuses :class:`~dsio.nn.module.DsioModule`'s chain unchanged. The only difference is what
happens in a step — a pretext objective decides that, and the encoder does not know which
one it is being trained by. That is the property that makes an encoder pretrained by MAE
and one pretrained by VICReg interchangeable downstream.

**MAE has no step override any more.** Its mask lives on the training dataset
(:class:`~dsio.nn.data.WindowDataset`), which writes a NaN sentinel into the target at
every visible position, and its loss (:class:`~dsio.nn.components.MaskedMSE`) reads that
sentinel directly. That makes MAE's training step exactly
:class:`~dsio.nn.module.DsioModule`'s generic ``(prediction, target)`` step — no batch
dict, no subclass — so :class:`SslModule` no longer overrides it.

**There is deliberately no pretext validation loss.** A validation dataset never masks
(``val_dataset`` in :mod:`dsio.nn.data` has no ``mask`` parameter at all), so an unmasked
batch carries nothing under ``target_key`` a reconstruction loss could score against — no
sentinel, sometimes not even the key, and when a label provider is configured, whatever it
put there is a class label, not a signal. ``SslModule.validation_step`` reflects that
rather than trying to paper over it: it does not touch ``target_key``. The Lightning
validation loop still runs — that is what keeps :class:`~dsio.ssl.probe.OnlineProbe` and
:class:`~dsio.ssl.probe.RankMeMonitor` firing — but nothing here reports ``val/loss``,
which matches what :mod:`dsio.train.ssl_task` already says about pretraining: the pretext
loss is not the thing being estimated, so a held-out probe score is what stands in for it.

**Contrastive methods still drive their own step.** SimCLR and VICReg augment two views of
``x`` inside their own ``step`` and never produce a ``(prediction, target)`` pair the
generic step could consume — the negatives are the rest of the batch, not a value pulled
out of it. Their objective never reads ``target_key`` either, so an unmasked validation
batch is exactly as usable to them as a training one; :class:`ContrastiveModule` restores a
real validation step for that reason. Moving them onto the same dataset-driven contract as
MAE is Task 6b.

**The copy-vs-learned diagnostic survives training, off to the side of the step.** ADR
0011: *"MAE logs masked and visible error separately... if visible error collapses while
masked error does not, the model is copying."* ``self.loss`` alone cannot report this —
:class:`~dsio.nn.components.MaskedMSE` reads only ``(prediction, target)`` and the whole
point of the sentinel is that ``target`` no longer carries a visible-position value to
compare against. ``SslModule.on_train_batch_end`` computes it instead, from ``batch["x"]``
(which does still hold the true value at every visible position — masking only zeroes the
hidden ones) and one extra ``eval()``-mode forward pass, gated on ``self.loss`` actually
being a ``MaskedMSE`` so it is a no-op for anything built through
:class:`ContrastiveModule`. This is an additive hook, not a step override: it runs
alongside whatever ``training_step`` already did, rather than deciding what that was.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from dsio.nn.components import MaskedMSE
from dsio.nn.module import DsioModule, Stage
from dsio.ssl.methods import PretextObjective, SslMethod


class SslModule(DsioModule):
    """A :class:`DsioModule` whose head comes from a pretext objective.

    ``head`` here is the *objective's* head — a decoder, a projector — and is discarded
    when the encoder is exported. Keeping it in the same slot as a supervised head is
    deliberate: it means ``encode`` is the same call in both cases, so the handoff from
    pretraining to probing has no adapter.

    ``method`` is typed narrowly, as :class:`~dsio.ssl.methods.PretextObjective`: this
    class itself only ever calls ``build_head`` on it. A pretext objective that also needs
    to drive its own training step (today, the contrastive ones) uses
    :class:`ContrastiveModule` instead.
    """

    def __init__(self, *, method: PretextObjective, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.method = method

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """No pretext validation loss — see the module docstring for why.

        Still lets Lightning's validation loop run to completion (a real batch goes
        through ``encode``, nothing raises), which is what the online probe and RankMe
        monitor need in order to fire at all: both hook ``on_validation_epoch_end``, which
        only runs when a validation loop actually happened.
        """
        self.encode(batch["x"])
        return torch.zeros((), device=self.device)

    def on_train_batch_end(
        self, outputs: Any, batch: dict[str, Any], batch_idx: int
    ) -> None:
        """Log the masked-vs-visible reconstruction split, for the objective that has one.

        A no-op unless ``self.loss`` is a :class:`~dsio.nn.components.MaskedMSE` — a
        :class:`ContrastiveModule` (or any future subclass with a different loss) inherits
        this hook but never triggers the body, so it never pays the extra forward pass and
        never risks reading a ``target_key`` that, for it, is not a sentinel at all.
        """
        if not isinstance(self.loss, MaskedMSE):
            return
        x = batch["x"]
        target = batch[self.target_key]
        hidden = ~torch.isnan(target)
        visible = ~hidden
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                prediction = self(x)
        finally:
            self.train(was_training)
        masked_mse = nn.functional.mse_loss(prediction[hidden], target[hidden])
        # visible_mse compares the reconstruction against x, not target: x still carries
        # the true value at every visible position (masking only zeroes the hidden ones),
        # which is exactly what the sentinel target no longer does.
        visible_mse = nn.functional.mse_loss(prediction[visible], x[visible])
        self.log("train/masked_mse", masked_mse, batch_size=x.shape[0], on_epoch=True)
        self.log("train/visible_mse", visible_mse, batch_size=x.shape[0], on_epoch=True)

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, torch.Tensor]:
        """Predicting from a pretraining module means embedding, not classifying."""
        return {"row": batch["row"], "embedding": self.encode(batch["x"]).detach()}

    def encoder_state(self) -> dict[str, torch.Tensor]:
        """The weights worth keeping: everything the chain needs to produce features.

        The objective's head is excluded. A decoder trained to reconstruct masked spans has
        no meaning outside the pretext task, and shipping it invites someone to load it as
        though it were part of the model.
        """
        state: dict[str, torch.Tensor] = {}
        for name in ("preprocessor", "transform", "backbone"):
            component = getattr(self, name, None)
            if isinstance(component, nn.Module):
                for key, value in component.state_dict().items():
                    state[f"{name}.{key}"] = value
        return state


class ContrastiveModule(SslModule):
    """An :class:`SslModule` whose objective drives its own training *and* validation step.

    SimCLR and VICReg need ``x`` itself, not a ``(prediction, target)`` pair pulled out of
    the batch: the two views they compare are produced by augmenting ``x`` twice inside
    ``step``, and the loss over them has no "target" to speak of. Because ``step`` never
    reads ``target_key``, an unmasked validation batch is exactly as usable as a training
    one, so — unlike the base :class:`SslModule` — this class keeps a real validation loss.
    This is the pre-Task-6a behaviour, kept for the two methods that have not moved onto
    the dataset-driven contract yet; MAE is the one that has (see the module docstring).
    """

    def _common_step(self, batch: dict[str, Any], stage: Stage) -> torch.Tensor:
        x = batch["x"]
        loss, logs = cast(SslMethod, self.method).step(self, x)
        self.log(f"{stage}/loss", loss, batch_size=x.shape[0], on_epoch=True, prog_bar=False)
        for name, value in logs.items():
            self.log(f"{stage}/{name}", value, batch_size=x.shape[0], on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "val")
