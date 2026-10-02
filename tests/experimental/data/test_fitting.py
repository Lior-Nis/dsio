"""Training-role statistics never see held-out samples, and standardization is exact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dsio.data.adapters import entity_examples
from dsio.data.splits import generate
from dsio.data.store import SignalStore
from dsio.experimental.data import fit_standardization, record_fitted
from dsio.experimental.model import Standardize


def _store(path: Path, overrides: dict[str, float] | None = None) -> SignalStore:
    values = {f"s{index}": float(index) for index in range(8)}
    values.update(overrides or {})
    with SignalStore.builder(path, channels=2) as builder:
        for name, value in values.items():
            builder.add(name, np.array([[value, 5.0], [value + 1, 5.0]]), group=name)
    return SignalStore(path)


def _manifest(store: SignalStore):  # type: ignore[no-untyped-def]
    return generate(
        entity_examples(store), "group_shuffle", name="fit", seed=3, parameters={"test_size": 0.25}
    )


def test_statistics_use_only_the_training_role(tmp_path: Path) -> None:
    store = _store(tmp_path / "base")
    manifest = _manifest(store)
    held_out = list(manifest.fold(0).assignments["test"])

    fitted = fit_standardization(store, manifest, fold=0)
    rows = np.concatenate(
        [store.read_sample(sample_id)["data"] for sample_id in fitted["sample_ids"]]
    ).astype(np.float64)

    assert set(fitted["sample_ids"]) == set(manifest.fold(0).assignments["train"])
    assert fitted["mean"] == rows.mean(axis=0).tolist()
    assert fitted["scale"] == [rows.std(axis=0)[0], 1.0]  # zero-variance feature maps to 1

    tampered = _store(tmp_path / "tampered", {sample_id: 1000.0 for sample_id in held_out})
    assert fit_standardization(tampered, _manifest(tampered), fold=0)["mean"] == fitted["mean"]


def test_observed_statistics_ignore_missing_values_and_refuse_empty_features(
    tmp_path: Path,
) -> None:
    with SignalStore.builder(tmp_path / "nan", channels=2) as builder:
        for index in range(4):
            builder.add(f"s{index}", np.array([[float(index), np.nan]]), group=f"g{index}")
    store = SignalStore(tmp_path / "nan")
    manifest = _manifest(store)

    with pytest.raises(ValueError, match=r"features \[1\] have no observed value"):
        fit_standardization(store, manifest, fold=0, observed=True)
    assert np.isnan(fit_standardization(store, manifest, fold=0)["mean"][1])


def test_standardize_matches_the_consumer_arithmetic_bit_for_bit() -> None:
    mean, scale = [0.5, -2.0, 3.25], [1.5, 1.0, 0.75]
    x = torch.randn(4, 1, 3)
    expected = (x.float() - torch.tensor(mean).reshape(1, 1, -1)) / torch.tensor(scale).reshape(
        1, 1, -1
    )
    assert torch.equal(Standardize(mean, scale)(x), expected)

    pixels = torch.randint(0, 256, (2, 28, 28), dtype=torch.uint8)
    assert torch.equal(Standardize([0.0], [255.0])(pixels), pixels.float() / 255.0)


def test_standardize_checks_features_axis_and_statistics() -> None:
    channel_first = Standardize([1.0, 2.0], [1.0, 2.0], axis=1)
    assert channel_first(torch.ones(1, 2, 3)).shape == (1, 2, 3)
    with pytest.raises(ValueError, match="expected 2 features on axis 1"):
        channel_first(torch.ones(1, 3, 2))
    with pytest.raises(ValueError, match="same length"):
        Standardize([0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="non-zero"):
        Standardize([0.0], [0.0])


def test_fitted_values_are_logged_as_run_relative_evidence() -> None:
    from mlflow import MlflowClient

    client = MlflowClient()
    run = client.create_run(client.create_experiment("fitting-evidence"))
    path = record_fitted(run.info.run_id, "standardization", {"mean": [1.0], "scale": [2.0]})

    assert path == "fitted/standardization.json"
    downloaded = Path(client.download_artifacts(run.info.run_id, path))
    assert downloaded.read_text(encoding="utf-8").count('"mean"') == 1
    with pytest.raises(ValueError, match="invalid fitted artifact name"):
        record_fitted(run.info.run_id, "../escape", {})
