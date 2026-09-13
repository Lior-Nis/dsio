"""Shared Lightning Trainer construction stays a transparent config mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

lightning = pytest.importorskip("lightning")

from dsio.train.trainer import TrainerConfig, build_trainer  # noqa: E402


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
        "default_root_dir": tmp_path,
        "logger": logger,
        "enable_checkpointing": checkpoint,
        "callbacks": callbacks,
    }
