"""The pretraining module: the shared chain, driven by a pretext objective.

Reuses :class:`~dsio.nn.module.DsioModule`'s chain unchanged. The only difference is what
happens in a step — a pretext objective decides that, and the encoder does not know which
one it is being trained by. That is the property that makes an encoder pretrained by MAE
and one pretrained by VICReg interchangeable downstream.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dsio.nn.module import DsioModule, Stage
from dsio.ssl.methods import SslMethod


class SslModule(DsioModule):
    """A :class:`DsioModule` whose loss comes from a pretext objective.

    ``head`` here is the *objective's* head — a decoder, a projector — and is discarded
    when the encoder is exported. Keeping it in the same slot as a supervised head is
    deliberate: it means ``encode`` is the same call in both cases, so the handoff from
    pretraining to probing has no adapter.
    """

    def __init__(self, *, method: SslMethod, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.method = method

    def _common_step(self, batch: dict[str, Any], stage: Stage) -> torch.Tensor:
        x = batch["x"]
        loss, logs = self.method.step(self, x)
        self.log(f"{stage}/loss", loss, batch_size=x.shape[0], on_epoch=True, prog_bar=False)
        for name, value in logs.items():
            self.log(f"{stage}/{name}", value, batch_size=x.shape[0], on_epoch=True)
        return loss

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
