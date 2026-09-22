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
from torchmetrics import MeanMetric

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


class StaticObjective:
    def __init__(self, result: object) -> None:
        self.result = result

    def __call__(self, *args: object) -> object:
        del args
        return self.result


class InvalidOptimizerFactory:
    def __call__(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        return object()


class SchedulerMappingFactory:
    def __call__(
        self, optimizer: torch.optim.Optimizer
    ) -> Mapping[str, object]:
        return {
            "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
            "interval": "epoch",
            "frequency": 1,
        }


class InvalidSchedulerMappingFactory:
    def __call__(
        self, optimizer: torch.optim.Optimizer
    ) -> Mapping[str, object]:
        del optimizer
        return {"interval": "epoch"}


class TorchMetricsObjective(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.means = nn.ModuleDict(
            {
                "train_metric": MeanMetric(),
                "validate_metric": MeanMetric(),
                "test_metric": MeanMetric(),
            }
        )

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        stage: str,
    ) -> Mapping[str, object]:
        prediction = model(batch["x"])
        loss = prediction.square().mean()
        metric = self.means[f"{stage}_metric"]
        metric.update(prediction.detach().mean())
        return {"loss": loss, "mean": metric}


class SharedTorchMetricsObjective(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = MeanMetric()

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        stage: str,
    ) -> Mapping[str, object]:
        del stage
        prediction = model(batch["x"])
        loss = prediction.square().mean()
        self.mean.update(prediction.detach().mean())
        return {"loss": loss, "mean": self.mean}


class UnregisteredTorchMetricsObjective:
    def __init__(self) -> None:
        self.mean = MeanMetric()

    def __call__(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        stage: str,
    ) -> Mapping[str, object]:
        del stage
        prediction = model(batch["x"])
        loss = prediction.square().mean()
        self.mean.update(prediction.detach().mean())
        return {"loss": loss, "mean": self.mean}


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
        optimizer_factory=torch.optim.AdamW,
        optimizer_parameters={"lr": 0.01, "weight_decay": 0.0},
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
    module = DsioModule(model=nn.Identity(), objective=StaticObjective(result))
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    batch = {"sample_id": ["sample"], "x": torch.ones(1, 1)}

    with pytest.raises(ModuleError, match=message):
        module.training_step(batch, 0)


def test_training_batches_require_ordered_sample_identity() -> None:
    module = _module()

    with pytest.raises(ModuleError, match="sample_id"):
        module.training_step({"x": torch.ones(1, 2), "y": torch.zeros(1)}, 0)


def test_native_optimizer_and_scheduler_factories_return_lightning_configuration() -> None:
    module = DsioModule(
        model=nn.Linear(1, 1),
        objective=ClassificationObjective(),
        optimizer_factory=torch.optim.SGD,
        optimizer_parameters={"lr": 0.25},
        scheduler_factory=torch.optim.lr_scheduler.StepLR,
        scheduler_parameters={"step_size": 2, "gamma": 0.5},
    )

    configured = module.configure_optimizers()

    assert isinstance(configured, dict)
    assert type(configured["optimizer"]) is torch.optim.SGD
    assert configured["optimizer"].param_groups[0]["lr"] == 0.25
    assert type(configured["lr_scheduler"]) is torch.optim.lr_scheduler.StepLR


def test_native_lightning_scheduler_mapping_is_validated_and_preserved() -> None:
    module = DsioModule(
        model=nn.Linear(1, 1),
        objective=ClassificationObjective(),
        optimizer_factory=torch.optim.SGD,
        optimizer_parameters={"lr": 0.25},
        scheduler_factory=SchedulerMappingFactory(),
    )

    configured = module.configure_optimizers()

    assert configured["lr_scheduler"]["interval"] == "epoch"
    assert isinstance(
        configured["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.StepLR,
    )


def test_incomplete_lightning_scheduler_mapping_is_rejected() -> None:
    module = DsioModule(
        model=nn.Linear(1, 1),
        objective=ClassificationObjective(),
        scheduler_factory=InvalidSchedulerMappingFactory(),
    )

    with pytest.raises(ModuleError, match="mapping requires 'scheduler'"):
        module.configure_optimizers()


def test_native_torchmetrics_are_emitted_only_through_lightning_log() -> None:
    objective = TorchMetricsObjective()
    module = DsioModule(model=nn.Identity(), objective=objective)  # type: ignore[arg-type]
    logged: dict[str, object] = {}
    module.log = lambda name, value, **kwargs: logged.setdefault(name, value)  # type: ignore[method-assign,assignment]

    loss = module.training_step(
        {"sample_id": ["left", "right"], "x": torch.tensor([[1.0], [3.0]])},
        0,
    )

    assert loss == torch.tensor(5.0)
    assert logged["train/mean"] is objective.means["train_metric"]


def test_distinct_stage_metrics_complete_a_real_train_and_validation_loop(
    tmp_path: Path,
) -> None:
    module = DsioModule(
        model=nn.Sequential(nn.Flatten(), nn.Linear(2, 2)),
        objective=TorchMetricsObjective(),
    )
    trainer = _trainer(tmp_path / "metrics", max_epochs=1)

    trainer.fit(module, datamodule=_data_module(tmp_path / "metric-samples"))

    assert torch.isfinite(trainer.callback_metrics["train/mean"])
    assert torch.isfinite(trainer.callback_metrics["val/mean"])


def test_one_torchmetric_instance_cannot_cross_stage_log_names() -> None:
    module = DsioModule(model=nn.Identity(), objective=SharedTorchMetricsObjective())
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    batch = {"sample_id": ["sample"], "x": torch.ones(1, 1)}

    module.training_step(batch, 0)
    with pytest.raises(ModuleError, match="distinct metric instances"):
        module.validation_step(batch, 0)


def test_torchmetric_must_be_registered_on_the_lightning_module() -> None:
    module = DsioModule(model=nn.Identity(), objective=UnregisteredTorchMetricsObjective())
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]

    with pytest.raises(ModuleError, match="must be registered"):
        module.training_step(
            {"sample_id": ["sample"], "x": torch.ones(1, 1)},
            0,
        )


def test_optimizer_factory_must_return_a_native_optimizer() -> None:
    module = DsioModule(
        model=nn.Linear(1, 1),
        objective=ClassificationObjective(),
        optimizer_factory=InvalidOptimizerFactory(),
    )

    with pytest.raises(ModuleError, match="native torch optimizer"):
        module.configure_optimizers()


def test_anonymous_components_fail_before_training() -> None:
    with pytest.raises(ModuleError, match="named importable"):
        DsioModule(model=nn.Identity(), objective=lambda *_: {"loss": torch.tensor(1.0)})


@pytest.mark.parametrize("name", ["loss_step", "loss_epoch"])
def test_objective_metrics_cannot_collide_with_lightning_loss_names(name: str) -> None:
    module = DsioModule(
        model=nn.Identity(),
        objective=StaticObjective({"loss": torch.tensor(1.0), name: torch.tensor(2.0)}),
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


@pytest.mark.parametrize(
    "row",
    [torch.tensor([7]), torch.tensor(7), torch.tensor([[7], [8]]), [7]],
)
def test_prediction_rows_must_match_sample_identity(row: object) -> None:
    module = DsioModule(model=nn.Identity(), objective=ClassificationObjective())

    with pytest.raises(ModuleError, match="prediction row|rows for 2 sample_id"):
        module.predict_step(
            {
                "sample_id": ["left", "right"],
                "x": torch.ones(2, 1),
                "row": row,
            },
            0,
        )


@pytest.mark.parametrize(
    "row",
    [torch.tensor([0.0, 1.0]), torch.tensor([False, True]), torch.tensor([0j, 1j])],
)
def test_prediction_rows_must_be_integral(row: torch.Tensor) -> None:
    module = DsioModule(model=nn.Identity(), objective=ClassificationObjective())

    with pytest.raises(ModuleError, match="row must be a dense integer tensor"):
        module.predict_step(
            {
                "sample_id": ["left", "right"],
                "x": torch.ones(2, 1),
                "row": row,
            },
            0,
        )


@pytest.mark.parametrize(
    "row",
    [
        torch.quantize_per_tensor(torch.tensor([0.0, 1.0]), 1.0, 0, torch.qint8),
        torch.sparse_coo_tensor(
            torch.tensor([[0, 1]]),
            torch.tensor([0, 1]),
            size=(2,),
        ),
    ],
)
def test_prediction_rows_must_be_dense_ordinary_integers(row: torch.Tensor) -> None:
    module = DsioModule(model=nn.Identity(), objective=ClassificationObjective())

    with pytest.raises(ModuleError, match="row must be a dense integer tensor"):
        module.predict_step(
            {
                "sample_id": ["left", "right"],
                "x": torch.ones(2, 1),
                "row": row,
            },
            0,
        )


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
