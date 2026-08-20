"""Pretraining and the encoder handoff, with the hardcoded-path bug made unrepresentable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("sklearn")

from dsio.artifacts.store import REGISTRY_ROOT_ENV, ModelRegistry  # noqa: E402
from dsio.config.schema import RunConfig  # noqa: E402
from dsio.data import SignalStore, WindowSpec, entity_examples  # noqa: E402
from dsio.data.store import DATA_ROOT_ENV  # noqa: E402
from dsio.eval import read_report  # noqa: E402
from dsio.nn import LABELS, labels  # noqa: E402
from dsio.runs import RunLedger  # noqa: E402
from dsio.splits import SplitFile  # noqa: E402
from dsio.train import check, execute  # noqa: E402
from dsio.train.ssl_task import SslPretrainTask  # noqa: E402
from dsio.train.torch_task import (  # noqa: E402
    Component,
    EncoderRef,
    TorchTask,
    TrainerConfig,
    load_encoder,
)


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "stores"))
    monkeypatch.setenv(REGISTRY_ROOT_ENV, str(tmp_path / "models"))
    rng = np.random.default_rng(0)
    with SignalStore.builder(tmp_path / "stores" / "tone", channels=2) as builder:
        for group in range(9):
            positive = group % 2 == 0
            t = np.arange(1600) / 100.0
            signal = (rng.standard_normal((1600, 2)) * 0.5).astype("float32")
            if positive:
                signal[:, 0] += (np.sin(2 * np.pi * 5 * t) * 2.0).astype("float32")
            builder.add(
                f"p{group}", signal, group=f"p{group}", attrs={"positive": int(positive)}
            )

    if "tone" not in LABELS:

        @labels("tone")
        def _tone(store: SignalStore) -> np.ndarray:
            out = np.zeros(store.n_rows, dtype=np.float32)
            for entity in store.entities:
                out[entity.start_row : entity.end_row] = float(entity.attrs["positive"])
            return out

    # Three hand-picked folds over the nine groups — this suite asserts on structure
    # (fold count, metric keys), not on prediction quality, so no balancing is needed.
    store = SignalStore(tmp_path / "stores" / "tone")
    digest = entity_examples(store).digest
    folds = [
        {"test": ["p0", "p1", "p2"], "val": ["p8"], "train": ["p3", "p4", "p5", "p6", "p7"]},
        {"test": ["p3", "p4", "p5"], "train": ["p0", "p1", "p2", "p6", "p7", "p8"]},
        {"test": ["p6", "p7", "p8"], "train": ["p0", "p1", "p2", "p3", "p4", "p5"]},
    ]
    for fold, parts in enumerate(folds):
        SplitFile(
            store=store.path.name,
            store_manifest_sha256=digest,
            name="k3",
            fold=fold,
            counts={part: len(members) for part, members in parts.items()},
            parts=parts,
        ).save(tmp_path / "splits" / "k3" / f"fold{fold}.yaml")
    return tmp_path


WINDOW = WindowSpec(length=128, stride=64, label_policy="majority")


def pretrain_task(root: Path, method: str = "mae", **overrides) -> SslPretrainTask:  # type: ignore[no-untyped-def]
    extra: dict[str, object] = {}
    if method == "mae":
        extra["mask"] = Component(name="span", params={"ratio": 0.5, "span": 16})
    else:
        extra["augmentor"] = Component(name="jitter", params={"sigma": 0.2})
    defaults = dict(
        store="tone",
        window=WINDOW,
        split="k3",
        splits_root=root / "splits",
        method=Component(name=method),
        backbone=Component(name="conv1d", params={"hidden": 8, "out_dim": 16, "depth": 1}),
        transform=Component(name="instance_standardize"),
        register_as=f"enc_{method}",
        labels="tone",
        batch_size=16,
        trainer=TrainerConfig(max_epochs=2, accelerator="cpu", devices=1, checkpoint=False),
        **extra,
    )
    return SslPretrainTask(**{**defaults, **overrides})


def run(config: RunConfig, root: Path):  # type: ignore[no-untyped-def]
    ledger = RunLedger(root / "runs")
    active = ledger.start(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with active:
        metrics = execute(config, active)
    return active, metrics


# --- config -------------------------------------------------------------------------


def test_mae_without_a_mask_is_rejected(corpus: Path) -> None:
    with pytest.raises(ValueError, match="needs a mask"):
        pretrain_task(corpus, "mae", mask=None)


def test_a_view_method_without_an_augmentor_is_rejected(corpus: Path) -> None:
    """Both views would be identical and the objective degenerate — a loss that goes
    straight to zero and an encoder that has learned nothing."""
    with pytest.raises(ValueError, match="two views"):
        pretrain_task(corpus, "simclr", augmentor=None)


def test_preflight_resolves_probe_only_names(corpus: Path) -> None:
    config = RunConfig(name="p", task=pretrain_task(corpus, labels="not_a_provider"))
    with pytest.raises(KeyError, match="unknown labels"):
        check(config)


# --- pretraining --------------------------------------------------------------------


@pytest.mark.parametrize("method", ["mae", "simclr", "vicreg"])
def test_every_method_pretrains_and_registers_an_encoder(method: str, corpus: Path) -> None:
    config = RunConfig(name=f"pre_{method}", seed=0, task=pretrain_task(corpus, method))
    check(config)
    active, metrics = run(config, corpus)

    assert metrics["feature_dim"] == 16.0
    assert metrics["train_windows"] > 0
    version = ModelRegistry().versions(f"enc_{method}")[-1]
    assert version.run_id == active.run_id
    assert version.size_bytes > 0


def test_pretraining_writes_no_evaluation_report(corpus: Path) -> None:
    """Pretraining is deliberately not a fold loop. A cross-validated masked-reconstruction
    MSE is a number nobody should act on, so it is not produced."""
    config = RunConfig(name="pre", seed=0, task=pretrain_task(corpus))
    active, _ = run(config, corpus)
    with pytest.raises(ValueError, match="no evaluation report"):
        read_report(active.artifacts_dir)


def test_the_encoder_records_its_lineage(corpus: Path) -> None:
    """Two models with identical bytes but different training data are different models."""
    config = RunConfig(name="pre", seed=0, task=pretrain_task(corpus))
    active, _ = run(config, corpus)
    version = ModelRegistry().versions("enc_mae")[-1]
    store = SignalStore(corpus / "stores" / "tone")
    assert version.config_hash == config.config_hash
    assert version.data_snapshot_ids == (store.manifest().signal_sha256,)
    assert version.provenance_digest
    written = json.loads((active.artifacts_dir / "encoder.json").read_text())
    assert written["digest"] == version.digest


def test_the_online_probe_runs_during_pretraining(corpus: Path) -> None:
    """The alternative is a subprocess plus a loop polling a directory from outside the run."""
    config = RunConfig(name="pre", seed=0, task=pretrain_task(corpus))
    _, metrics = run(config, corpus)
    assert "probe_rankme" in metrics
    assert "probe_accuracy" in metrics


def test_pretraining_on_a_missing_fold_fails_loudly(corpus: Path) -> None:
    config = RunConfig(name="pre", seed=0, task=pretrain_task(corpus, fold=7))
    with pytest.raises(ValueError, match="fold 7 is not in split family"):
        run(config, corpus)


# --- the handoff --------------------------------------------------------------------


def downstream_task(root: Path, encoder: EncoderRef | None) -> TorchTask:
    return TorchTask(
        store="tone",
        window=WINDOW,
        labels="tone",
        split="k3",
        splits_root=root / "splits",
        backbone=Component(name="conv1d", params={"hidden": 8, "out_dim": 16, "depth": 1}),
        head=Component(name="linear", params={"out_dim": 2}),
        loss=Component(name="cross_entropy", params={"threshold": 0.5}),
        transform=Component(name="instance_standardize"),
        encoder=encoder,
        batch_size=16,
        metrics=("accuracy", "roc_auc"),
        trainer=TrainerConfig(max_epochs=2, accelerator="cpu", devices=1, checkpoint=False),
    )


@pytest.fixture
def pretrained(corpus: Path) -> EncoderRef:
    run(RunConfig(name="pre", seed=0, task=pretrain_task(corpus)), corpus)
    version = ModelRegistry().versions("enc_mae")[-1]
    return EncoderRef(name=version.name, version=version.version, digest=version.digest)


def test_a_pinned_encoder_loads_into_a_downstream_run(corpus: Path, pretrained: EncoderRef) -> None:
    config = RunConfig(name="probe", seed=0, task=downstream_task(corpus, pretrained))
    check(config)
    active, metrics = run(config, corpus)
    assert "accuracy" in metrics
    report, _ = read_report(active.artifacts_dir)
    assert report.n_folds == 3


def test_a_tampered_digest_fails_closed(corpus: Path, pretrained: EncoderRef) -> None:
    """There is no path to hardcode and no way to say 'latest', so the remaining risk is a
    swapped artifact — which the registry re-hashes and refuses."""
    from dsio.nn import BACKBONES

    backbone = BACKBONES.get("conv1d")(channels=2, hidden=8, out_dim=16, depth=1)
    wrong = pretrained.model_copy(update={"digest": "0" * 64})
    with pytest.raises(Exception, match="digest"):
        load_encoder(wrong, backbone=backbone)


def test_there_is_no_way_to_ask_for_latest(pretrained: EncoderRef) -> None:
    """Resolving a moving alias at load time is how a reproduction silently becomes a
    different experiment. A hardcoded path is the same failure with worse ergonomics."""
    assert isinstance(pretrained.version, int)
    with pytest.raises(Exception):
        EncoderRef(name="enc_mae", version="latest", digest=pretrained.digest)  # type: ignore[arg-type]


def test_freezing_actually_freezes(corpus: Path, pretrained: EncoderRef) -> None:
    """Both halves. A frozen BatchNorm whose running statistics keep updating is not
    frozen, and the difference shows up as a probe that mysteriously outperforms its own
    linear separability."""
    from dsio.nn import BACKBONES

    backbone = BACKBONES.get("conv1d")(channels=2, hidden=8, out_dim=16, depth=1)
    report = load_encoder(pretrained, backbone=backbone)
    assert report["loaded_tensors"] > 0
    assert report["frozen_parameters"] > 0
    assert not any(p.requires_grad for p in backbone.parameters())
    assert backbone.training is False


def test_finetuning_leaves_the_encoder_trainable(corpus: Path, pretrained: EncoderRef) -> None:
    """A probe measures what the representation already contains; a finetune measures what
    it is a good starting point for. Reporting one as the other overstates the result."""
    from dsio.nn import BACKBONES

    backbone = BACKBONES.get("conv1d")(channels=2, hidden=8, out_dim=16, depth=1)
    load_encoder(pretrained.model_copy(update={"freeze": False}), backbone=backbone)
    assert all(p.requires_grad for p in backbone.parameters())


def test_loading_into_a_different_architecture_is_refused(
    corpus: Path, pretrained: EncoderRef
) -> None:
    """Silently loading a subset of weights produces a model that is part pretrained and
    part random, and reports as though it were fully pretrained."""
    from dsio.nn import BACKBONES

    mismatched = BACKBONES.get("conv1d")(channels=2, hidden=64, out_dim=16, depth=3)
    with pytest.raises(Exception):
        load_encoder(pretrained, backbone=mismatched)


def test_the_loaded_weights_actually_differ_from_a_fresh_init(
    corpus: Path, pretrained: EncoderRef
) -> None:
    """Guards against a load that silently no-ops and leaves the model randomly
    initialised — every other test here would still pass."""
    import torch

    from dsio.nn import BACKBONES

    fresh = BACKBONES.get("conv1d")(channels=2, hidden=8, out_dim=16, depth=1)
    before = [p.detach().clone() for p in fresh.parameters()]
    load_encoder(pretrained, backbone=fresh)
    assert not all(
        torch.equal(a, b) for a, b in zip(before, fresh.parameters(), strict=True)
    )
