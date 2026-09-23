"""Shared Lightning Trainer construction stays a transparent config mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

lightning = pytest.importorskip("lightning")

from dsio.train.trainer import TrainerConfig, build_callbacks, build_trainer  # noqa: E402


@pytest.mark.parametrize("has_validation", [True, False])
def test_build_callbacks_forwards_exact_checkpoint_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_validation: bool,
) -> None:
    import lightning.pytorch.callbacks as callback_module

    captured: dict[str, Any] = {}
    checkpoint = object()

    def fake_checkpoint(**kwargs: Any) -> object:
        captured.update(kwargs)
        return checkpoint

    monkeypatch.setattr(callback_module, "ModelCheckpoint", fake_checkpoint)
    config = TrainerConfig(monitor="val/accuracy", monitor_mode="max")

    callbacks = build_callbacks(config, tmp_path, has_validation=has_validation)

    expected = {
        "dirpath": tmp_path,
        "filename": "epoch{epoch:02d}",
        "save_top_k": 1,
        "auto_insert_metric_name": False,
    }
    if has_validation:
        expected.update(
            filename="epoch{epoch:02d}-val_accuracy{val/accuracy:.4f}",
            monitor="val/accuracy",
            mode="max",
        )
    assert callbacks == [checkpoint]
    assert captured == expected


@pytest.mark.parametrize(
    ("deterministic", "checkpoint", "expected_deterministic"),
    [(True, True, "warn"), (False, False, False)],
)
def test_build_trainer_maps_every_runtime_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deterministic: bool,
    checkpoint: bool,
    expected_deterministic: str | bool,
) -> None:
    captured: dict[str, Any] = {}
    built = object()

    def fake_trainer(**kwargs: Any) -> object:
        captured.update(kwargs)
        return built

    monkeypatch.setattr(lightning, "Trainer", fake_trainer)
    config = TrainerConfig(
        max_epochs=7,
        accelerator="cpu",
        devices=3,
        precision="64-true",
        gradient_clip_val=1.25,
        accumulate_grad_batches=4,
        early_stopping_patience=6,
        monitor="val/accuracy",
        monitor_mode="max",
        checkpoint=checkpoint,
        log_every_n_steps=13,
        enable_progress_bar=True,
        deterministic=deterministic,
        limit_val_batches=0.25,
        num_sanity_val_steps=5,
    )
    logger = object()
    callbacks = [object(), object()]

    result = build_trainer(config, tmp_path, logger, callbacks)

    assert result is built
    assert captured == {
        "max_epochs": 7,
        "accelerator": "cpu",
        "devices": 3,
        "precision": "64-true",
        "gradient_clip_val": 1.25,
        "accumulate_grad_batches": 4,
        "log_every_n_steps": 13,
        "enable_progress_bar": True,
        "enable_model_summary": False,
        "deterministic": expected_deterministic,
        "limit_val_batches": 0.25,
        "num_sanity_val_steps": 5,
        "default_root_dir": tmp_path,
        "logger": logger,
        "enable_checkpointing": checkpoint,
        "callbacks": callbacks,
    }


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [(None, type(None)), (0, int), (3, int), (0.0, float), (0.25, float), (1.0, float)],
)
def test_limit_val_batches_preserves_native_value_type(
    value: int | float | None,
    expected_type: type[object],
) -> None:
    configured = TrainerConfig(limit_val_batches=value).limit_val_batches

    assert configured == value
    assert type(configured) is expected_type


@pytest.mark.parametrize("value", [True, False, -1, -0.1, 1.1])
def test_limit_val_batches_rejects_non_native_ranges(value: object) -> None:
    with pytest.raises(ValidationError, match="limit_val_batches"):
        TrainerConfig(limit_val_batches=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, False, -1])
def test_num_sanity_val_steps_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValidationError, match="num_sanity_val_steps"):
        TrainerConfig(num_sanity_val_steps=value)  # type: ignore[arg-type]


def test_num_sanity_val_steps_preserves_lightning_default() -> None:
    assert TrainerConfig().num_sanity_val_steps is None
