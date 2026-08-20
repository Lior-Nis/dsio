"""The online probe and representation-quality metrics, as callbacks.

The usual shape for this is infrastructure: a callback that submits a downstream training
job, plus a loop that polls a checkpoint directory and tracks what it has already probed. Its
failure modes — a silently dead subprocess, a checkpoint probed twice, a probe reporting
against a checkpoint since overwritten — are all invisible from the training run.

A probe is a linear model on frozen features. It takes a fraction of a second. Making it a
callback removes the scheduler entirely.

**Why probe during pretraining at all.** The SSL loss is not monotonically related to
downstream quality; a run whose loss is still falling can already have passed its best
representation. Without a probe there is no signal to stop on.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader

from dsio.eval.metrics import compute


class Encodable(Protocol):
    """What a probe needs from a module: features, a device, and a training flag.

    Narrower than LightningModule on purpose. A probe works on anything that can encode,
    which is what lets the same callback measure a pretraining run and a supervised one.
    """

    training: bool
    device: Any

    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    def eval(self) -> Any: ...

    def train(self, mode: bool = True) -> Any: ...


def rankme(embeddings: torch.Tensor, epsilon: float = 1e-7) -> float:
    """Effective rank of an embedding matrix, from the entropy of its singular values.

    Garrido et al. (2023). The number to watch during SSL, because dimensional collapse is
    the characteristic failure and the loss cannot see it: an encoder using 3 of its 64
    dimensions can have a perfectly healthy loss curve and features that no downstream head
    can separate.

    Returns a value in ``[1, min(n, d)]``. Unsupervised, so it needs no labels and can run
    on the pretraining corpus itself.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"rankme expects [n, d], got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] < 2:
        raise ValueError("rankme needs at least two samples")
    singular = torch.linalg.svdvals(embeddings.float())
    total = singular.sum()
    if total <= 0:
        return 1.0
    p = singular / total + epsilon
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def embed(module: Encodable, loader: DataLoader[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Encode a loader into features and labels, leaving the module as it was found.

    Restoring the training flag matters: a probe that leaves the module in eval mode
    disables dropout and freezes BatchNorm for the rest of training, which degrades the run
    it was meant to measure and looks like the SSL method underperforming.
    """
    was_training = module.training
    module.eval()
    try:
        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for batch in loader:
            x = batch["x"].to(module.device)
            features.append(module.encode(x).detach().cpu().numpy())
            if "y" in batch:
                labels.append(np.asarray(batch["y"]))
    finally:
        module.train(was_training)
    return (
        np.concatenate(features) if features else np.empty((0, 0)),
        np.concatenate(labels) if labels else np.empty(0),
    )


class OnlineProbe(Callback):
    """Fit a linear model on frozen features and report what it scores.

    The probe never touches the encoder's gradients. It exists to *measure* pretraining,
    and a probe that backpropagated into the encoder would be supervised finetuning wearing
    a probe's name — with labels leaking into a representation that is supposed to be
    label-free.
    """

    def __init__(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        *,
        every_n_epochs: int = 1,
        metrics: tuple[str, ...] = ("accuracy", "roc_auc"),
        max_iter: int = 1000,
        prefix: str = "probe",
    ) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.every_n_epochs = max(1, every_n_epochs)
        self.metrics = metrics
        self.max_iter = max_iter
        self.prefix = prefix
        self.history: list[dict[str, float]] = []

    def on_validation_epoch_end(self, trainer: Trainer, module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        scores = self.run(cast("Encodable", module))
        self.history.append({"epoch": float(trainer.current_epoch), **scores})
        for name, value in scores.items():
            module.log(f"{self.prefix}/{name}", value, on_epoch=True)

    def run(self, module: Encodable) -> dict[str, float]:
        """Fit on the probe-train features and score on the probe-val features."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        x_train, y_train = embed(module, self.train_loader)
        x_val, y_val = embed(module, self.val_loader)
        if y_train.size == 0 or y_val.size == 0:
            raise ValueError("the probe needs labelled loaders; no 'y' was found in a batch")

        y_train = _hard(y_train)
        y_val = _hard(y_val)
        scores: dict[str, float] = {"rankme": rankme(torch.from_numpy(x_val))}
        if np.unique(y_train).size < 2:
            # Report the collapse rather than crash: a probe fold that is single-class is a
            # sampling problem to fix, not a reason to lose the whole pretraining run.
            scores["probe_unfittable"] = 1.0
            return scores

        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(max_iter=self.max_iter).fit(scaler.transform(x_train), y_train)
        transformed = scaler.transform(x_val)
        prediction = model.predict(transformed)
        probability = model.predict_proba(transformed)
        score = probability[:, 1] if probability.shape[1] == 2 else probability

        wanted = [name for name in self.metrics if np.unique(y_val).size >= 2 or "auc" not in name]
        scores.update(compute(wanted, y_val, prediction, score))
        return scores


class RankMeMonitor(Callback):
    """Log effective rank without needing labels, so it works on any pretraining corpus.

    Cheaper than the probe and complementary: RankMe says whether the representation has
    dimensions left, the probe says whether those dimensions carry the signal you care
    about. A run can look healthy on one and not the other, and knowing which is which is
    what tells you whether to change the method or change the data.
    """

    def __init__(
        self, loader: DataLoader[Any], *, every_n_epochs: int = 1, prefix: str = "quality"
    ) -> None:
        super().__init__()
        self.loader = loader
        self.every_n_epochs = max(1, every_n_epochs)
        self.prefix = prefix
        self.history: list[dict[str, float]] = []

    def on_validation_epoch_end(self, trainer: Trainer, module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        features, _ = embed(cast("Encodable", module), self.loader)
        value = rankme(torch.from_numpy(features))
        self.history.append({"epoch": float(trainer.current_epoch), "rankme": value})
        module.log(f"{self.prefix}/rankme", value, on_epoch=True)


def _hard(labels: np.ndarray) -> np.ndarray:
    """Binarise soft window labels for a probe, which needs classes rather than ratios."""
    return (labels > 0.5).astype(np.int64) if labels.dtype.kind == "f" else labels.astype(np.int64)
