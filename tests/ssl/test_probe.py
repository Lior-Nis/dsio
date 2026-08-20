"""The online probe and RankMe: measuring pretraining without a bash polling daemon."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("sklearn")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from dsio.nn.components import Conv1dEncoder, CrossEntropy, Jitter  # noqa: E402
from dsio.nn.module import DsioModule  # noqa: E402
from dsio.ssl import OnlineProbe, RankMeMonitor, embed, rankme  # noqa: E402


class Toy(Dataset):
    """Two separable clusters, so a linear probe on raw features should succeed."""

    def __init__(self, n: int = 64, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.y = (np.arange(n) % 2).astype(np.float32)
        base = rng.standard_normal((n, 2, 32)).astype("float32") * 0.2
        base[self.y == 1, 0] += 3.0
        self.x = base

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, object]:
        return {"x": torch.from_numpy(self.x[i]), "y": torch.tensor(self.y[i]), "row": i}


@pytest.fixture
def module() -> DsioModule:
    return DsioModule(
        backbone=Conv1dEncoder(channels=2, hidden=8, out_dim=8, depth=1),
        head=nn.Linear(8, 2),
        loss=CrossEntropy(threshold=0.5),
        augmentor=Jitter(0.5),
    )


@pytest.fixture
def loader() -> DataLoader:
    return DataLoader(Toy(), batch_size=16)


# --- RankMe ----------------------------------------------------------------------------


def test_rankme_is_high_for_a_full_rank_embedding() -> None:
    torch.manual_seed(0)
    assert rankme(torch.randn(200, 16)) > 12.0


def test_rankme_is_one_for_a_collapsed_embedding() -> None:
    """Dimensional collapse is the characteristic SSL failure and the loss cannot see it.

    An encoder using 3 of its 64 dimensions can have a perfectly healthy loss curve and
    features no downstream head can separate.
    """
    assert rankme(torch.ones(200, 16)) == pytest.approx(1.0, abs=0.01)


def test_rankme_falls_as_dimensions_are_lost() -> None:
    torch.manual_seed(0)
    full = torch.randn(200, 16)
    partial = full.clone()
    partial[:, 4:] = 0.0
    assert rankme(partial) < rankme(full)
    assert rankme(partial) == pytest.approx(4.0, abs=0.5)


def test_rankme_needs_a_matrix() -> None:
    with pytest.raises(ValueError, match=r"\[n, d\]"):
        rankme(torch.randn(10))
    with pytest.raises(ValueError, match="at least two samples"):
        rankme(torch.randn(1, 8))


def test_rankme_needs_no_labels(module: DsioModule, loader: DataLoader) -> None:
    """Which is what lets it run on the pretraining corpus itself."""
    features, _ = embed(module, loader)
    assert rankme(torch.from_numpy(features)) >= 1.0


# --- embedding -------------------------------------------------------------------------


def test_embed_restores_the_training_flag(module: DsioModule, loader: DataLoader) -> None:
    """The subtle bug: a probe that leaves the module in eval mode disables dropout and
    freezes BatchNorm for the rest of training, degrading the very run it measures — and
    it looks like the SSL method underperforming."""
    module.train()
    embed(module, loader)
    assert module.training is True

    module.eval()
    embed(module, loader)
    assert module.training is False


def test_embed_runs_the_encoder_in_eval_mode(module: DsioModule, loader: DataLoader) -> None:
    """Otherwise the augmentor fires and the measurement is of augmented features."""
    module.train()
    first, _ = embed(module, loader)
    second, _ = embed(module, loader)
    assert np.allclose(first, second)


def test_embed_returns_labels_when_present(module: DsioModule, loader: DataLoader) -> None:
    features, labels = embed(module, loader)
    assert features.shape == (64, 8)
    assert labels.shape == (64,)


def test_embed_leaves_no_gradients(module: DsioModule, loader: DataLoader) -> None:
    """A probe that backpropagated into the encoder would be supervised finetuning wearing
    a probe's name, with labels leaking into a label-free representation."""
    module.zero_grad()
    embed(module, loader)
    assert all(p.grad is None for p in module.parameters())


# --- the probe ---------------------------------------------------------------------------


def test_the_probe_separates_separable_features(module: DsioModule, loader: DataLoader) -> None:
    probe = OnlineProbe(loader, loader, metrics=("accuracy", "roc_auc"))
    scores = probe.run(module)
    assert scores["accuracy"] > 0.9
    assert "rankme" in scores


def test_the_probe_does_not_disturb_the_module(module: DsioModule, loader: DataLoader) -> None:
    module.train()
    before = [p.detach().clone() for p in module.parameters()]
    OnlineProbe(loader, loader).run(module)
    assert module.training is True
    assert all(torch.equal(a, b) for a, b in zip(before, module.parameters(), strict=True))


def test_a_single_class_probe_reports_instead_of_crashing() -> None:
    """A single-class probe fold is a sampling problem to fix, not a reason to lose a
    multi-hour pretraining run."""

    class OneClass(Toy):
        def __init__(self) -> None:
            super().__init__()
            self.y[:] = 1.0

    module = DsioModule(
        backbone=Conv1dEncoder(channels=2, hidden=8, out_dim=8, depth=1),
        head=nn.Linear(8, 2),
        loss=CrossEntropy(threshold=0.5),
    )
    loader = DataLoader(OneClass(), batch_size=16)
    scores = OnlineProbe(loader, loader).run(module)
    assert scores["probe_unfittable"] == 1.0


def test_the_probe_needs_labelled_loaders(module: DsioModule) -> None:
    class Unlabelled(Dataset):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, i: int) -> dict[str, object]:
            return {"x": torch.randn(2, 32), "row": i}

    loader = DataLoader(Unlabelled(), batch_size=4)
    with pytest.raises(ValueError, match="labelled loaders"):
        OnlineProbe(loader, loader).run(module)


def test_the_probe_fires_on_schedule(module: DsioModule, loader: DataLoader) -> None:
    """The alternative is a loop that polls a checkpoint directory from outside the run."""
    from lightning import Trainer

    probe = OnlineProbe(loader, loader, every_n_epochs=2)
    trainer = Trainer(
        max_epochs=4,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        callbacks=[probe],
    )
    trainer.fit(module, loader, loader)
    assert [entry["epoch"] for entry in probe.history] == [1.0, 3.0]


def test_rankme_monitor_records_a_history(module: DsioModule, loader: DataLoader) -> None:
    from lightning import Trainer

    monitor = RankMeMonitor(loader, every_n_epochs=1)
    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        callbacks=[monitor],
    )
    trainer.fit(module, loader, loader)
    assert len(monitor.history) == 2
    assert all(entry["rankme"] >= 1.0 for entry in monitor.history)
