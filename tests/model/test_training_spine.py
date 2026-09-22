"""The public training path is native Lightning over the two exact DSio classes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from lightning import Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from dsio.data.adapters import entity_examples
from dsio.data.loading import DsioDataModule
from dsio.data.splits import generate
from dsio.data.store import SignalStore
from dsio.model.module import DsioModule, ModuleError


class LabelledSamples(Dataset[Mapping[str, Any]]):
    def __init__(self, store: SignalStore, sample_ids: Sequence[str]) -> None:
        self.store = store
        self.sample_ids = tuple(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, position: int) -> Mapping[str, Any]:
        sample_id = self.sample_ids[position]
        return {
            "sample_id": sample_id,
            "x": torch.from_numpy(np.array(self.store.read_sample(sample_id)["data"], copy=True)),
            "y": torch.tensor(int(sample_id.rsplit("-", 1)[1]) % 2),
        }


def labelled_samples(
    store: SignalStore,
    examples: Any,
    sample_ids: Sequence[str],
) -> Dataset[Mapping[str, Any]]:
    del examples
    return LabelledSamples(store, sample_ids)


class ClassificationObjective(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("calls", torch.tensor(0))

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        stage: str,
    ) -> Mapping[str, torch.Tensor]:
        del stage
        self.calls.add_(1)
        prediction = model(batch["x"].float())
        target = batch["y"].long()
        return {
            "loss": F.cross_entropy(prediction, target),
            "accuracy": (prediction.argmax(dim=-1) == target).float().mean(),
        }


def _data_module(path: Path) -> DsioDataModule:
    with SignalStore.builder(path, channels=1, dtype="float32") as builder:
        for index in range(8):
            builder.add(
                f"sample-{index}",
                np.full((2, 1), index, dtype=np.float32),
                group=f"group-{index}",
            )
    store = SignalStore(path)
    examples = entity_examples(store)
    split = generate(
        examples,
        "group_shuffle",
        name="holdout",
        seed=11,
        parameters={"test_size": 0.25},
    )
    return DsioDataModule(
        store,
        examples,
        split,
        fold=0,
        roles={"train": "train", "validate": "test"},
        dataset_factory=labelled_samples,
        batch_size=2,
        seed=17,
    )


def _module() -> DsioModule:
    return DsioModule(
        model=nn.Sequential(nn.Flatten(), nn.Linear(2, 2)),
        objective=ClassificationObjective(),
        lr=0.01,
    )


def _trainer(path: Path, *, max_epochs: int) -> Trainer:
    return Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        default_root_dir=path,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )


def test_native_lightning_trains_and_resumes_the_exact_dsio_classes(tmp_path: Path) -> None:
    data = _data_module(tmp_path / "samples")
    module = _module()
    trainer = _trainer(tmp_path / "first", max_epochs=1)

    trainer.fit(module, datamodule=data)

    assert type(module) is DsioModule
    assert type(data) is DsioDataModule
    first_steps = trainer.global_step
    first_objective_calls = int(module.objective.calls)  # type: ignore[union-attr]
    checkpoint = trainer.checkpoint_callback.best_model_path
    assert first_steps > 0
    assert checkpoint.endswith(".ckpt")

    resumed = _module()
    resumed_trainer = _trainer(tmp_path / "resumed", max_epochs=2)
    resumed_trainer.fit(resumed, datamodule=data, ckpt_path=checkpoint)

    assert resumed_trainer.global_step > first_steps
    assert int(resumed.objective.calls) > first_objective_calls  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (torch.tensor(1.0), "return a mapping"),
        ({"accuracy": torch.tensor(1.0)}, "mandatory scalar tensor 'loss'"),
        ({"loss": 1.0}, "loss.*tensor"),
        ({"loss": torch.ones(2)}, "loss.*scalar"),
        ({"loss": torch.tensor(1.0), "accuracy": torch.ones(2)}, "accuracy.*scalar"),
    ],
)
def test_malformed_objective_results_fail_at_the_boundary(
    result: object,
    message: str,
) -> None:
    module = DsioModule(model=nn.Identity(), objective=lambda *_: result)  # type: ignore[arg-type]
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    batch = {"sample_id": ["sample"], "x": torch.ones(1, 1)}

    with pytest.raises(ModuleError, match=message):
        module.training_step(batch, 0)


def test_training_batches_require_ordered_sample_identity() -> None:
    module = _module()

    with pytest.raises(ModuleError, match="sample_id"):
        module.training_step({"x": torch.ones(1, 2), "y": torch.zeros(1)}, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": float("nan")},
        {"lr": float("inf")},
        {"weight_decay": float("nan")},
        {"weight_decay": float("inf")},
    ],
)
def test_optimizer_hyperparameters_must_be_finite(kwargs: dict[str, float]) -> None:
    with pytest.raises(ModuleError, match="finite"):
        DsioModule(model=nn.Linear(1, 1), objective=ClassificationObjective(), **kwargs)


@pytest.mark.parametrize("name", ["loss_step", "loss_epoch"])
def test_objective_metrics_cannot_collide_with_lightning_loss_names(name: str) -> None:
    module = DsioModule(
        model=nn.Identity(),
        objective=lambda *_: {"loss": torch.tensor(1.0), name: torch.tensor(2.0)},
    )

    with pytest.raises(ModuleError, match="reserved by Lightning"):
        module.training_step({"sample_id": ["sample"], "x": torch.ones(1, 1)}, 0)


@pytest.mark.parametrize(
    ("model", "sample_ids", "message"),
    [
        (nn.Identity(), ["only"], "2 predictions for 1 sample_id"),
        (nn.Identity(), ["left", "right"], "1 predictions for 2 sample_id"),
        (lambda _: {"prediction": torch.ones(1)}, ["only"], "must be a tensor"),
    ],
)
def test_prediction_output_must_match_sample_identity(
    model: object,
    sample_ids: list[str],
    message: str,
) -> None:
    configured = model if isinstance(model, nn.Module) else CallableModel(model)
    module = DsioModule(model=configured, objective=ClassificationObjective())
    x = torch.ones(2 if len(sample_ids) == 1 else 1, 1)

    with pytest.raises(ModuleError, match=message):
        module.predict_step({"sample_id": sample_ids, "x": x}, 0)


class CallableModel(nn.Module):
    def __init__(self, function: Any) -> None:
        super().__init__()
        self.function = function

    def forward(self, x: torch.Tensor) -> Any:
        return self.function(x)


def test_dsio_lightning_classes_cannot_be_subclassed() -> None:
    with pytest.raises(TypeError, match="DsioModule.*cannot be subclassed"):

        class ProjectModule(DsioModule):
            pass

    with pytest.raises(TypeError, match="DsioDataModule.*cannot be subclassed"):

        class ProjectDataModule(DsioDataModule):
            pass
