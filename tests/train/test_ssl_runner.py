"""Pretraining and the encoder handoff, with the hardcoded-path bug made unrepresentable.

Task 6b deleted the pretext-objective registry (``dsio.ssl.methods``): a pretraining task
now names its ``backbone``/``head``/``loss`` directly, exactly like a supervised
``TorchTask``, plus exactly one of ``mask`` or ``augmentor``. ``pretrain_task()`` below
picks the right head/loss/mask/augmentor for a given ``method`` name purely as test
scaffolding — ``SslPretrainTask`` itself has no ``method`` field any more.
"""

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
from dsio.data.adapters import entity_examples  # noqa: E402
from dsio.data.store import DATA_ROOT_ENV, SignalStore  # noqa: E402
from dsio.data.views import WindowSpec  # noqa: E402
from dsio.eval.contract import PREDICTIONS_FILE  # noqa: E402
from dsio.model.registry import LABELS, labels  # noqa: E402
from dsio.runs.record import RunLedger  # noqa: E402
from dsio.splits.models import SplitFile, SplitFold  # noqa: E402
from dsio.train.runner import check, execute  # noqa: E402
from dsio.train.ssl_task import SslPretrainTask  # noqa: E402
from dsio.train.torch_task import (  # noqa: E402
    Component,
    EncoderRef,
    TorchTask,
    TrainerConfig,
    load_encoder,
)
from dsio.train.tracking import MlflowUnavailableError, resolve_tracking_uri  # noqa: E402


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
    SplitFile(
        store=store.path.name,
        store_manifest_sha256=digest,
        name="k3",
        folds=[
            SplitFold(
                index=fold,
                counts={part: len(members) for part, members in parts.items()},
                parts=parts,
            )
            for fold, parts in enumerate(folds)
        ],
    ).save(tmp_path / "splits" / "k3" / "split.yaml")
    return tmp_path


WINDOW = WindowSpec(length=128, stride=64, label_policy="majority")


#: Which head/loss/mask/augmentor a given pretext shape needs. Test scaffolding only:
#: SslPretrainTask itself has no notion of a "method" name, just backbone/head/loss plus
#: exactly one of mask/augmentor.
_METHOD_COMPONENTS: dict[str, dict[str, object]] = {
    "mae": {
        "head": Component(name="mae_decoder"),
        "loss": Component(name="masked_mse"),
        "mask": Component(name="span", params={"ratio": 0.5, "span": 16}),
    },
    "simclr": {
        "head": Component(name="simclr_projector"),
        "loss": Component(name="nt_xent"),
        "augmentor": Component(name="jitter", params={"sigma": 0.2}),
    },
    "vicreg": {
        "head": Component(name="vicreg_projector"),
        "loss": Component(name="vicreg"),
        "augmentor": Component(name="jitter", params={"sigma": 0.2}),
    },
}


def pretrain_task(root: Path, method: str = "mae", **overrides) -> SslPretrainTask:  # type: ignore[no-untyped-def]
    defaults = dict(
        store="tone",
        window=WINDOW,
        split="k3",
        splits_root=root / "splits",
        backbone=Component(name="conv1d", params={"hidden": 8, "out_dim": 16, "depth": 1}),
        transform=Component(name="instance_standardize"),
        register_as=f"enc_{method}",
        labels="tone",
        batch_size=16,
        trainer=TrainerConfig(max_epochs=2, accelerator="cpu", devices=1, checkpoint=False),
        **_METHOD_COMPONENTS[method],
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


def test_neither_mask_nor_augmentor_is_rejected(corpus: Path) -> None:
    """Without one of the two, nothing tells this task which training-dataset contract
    to build."""
    with pytest.raises(ValueError, match="exactly one of"):
        pretrain_task(corpus, "mae", mask=None)


def test_both_mask_and_augmentor_is_rejected(corpus: Path) -> None:
    """Both views would be identical and the objective degenerate — a loss that goes
    straight to zero and an encoder that has learned nothing — if augmentor were even
    consulted; setting both at once is ambiguous about which contract to build, so it is
    rejected rather than silently preferring one."""
    with pytest.raises(ValueError, match="exactly one of"):
        pretrain_task(
            corpus, "simclr", mask=Component(name="span", params={"ratio": 0.5, "span": 16})
        )


def test_preflight_resolves_probe_only_names(corpus: Path) -> None:
    config = RunConfig(name="p", task=pretrain_task(corpus, labels="not_a_provider"))
    with pytest.raises(KeyError, match="unknown labels"):
        check(config)


def test_preflight_rejects_a_fold_the_split_family_does_not_declare(corpus: Path) -> None:
    """The same guard `run_ssl_pretrain` has at the top of the runner, reached here from
    the CLI's pre-flight step instead -- both call `require_fold`, so both name the same
    missing fold with `SplitFile.fold`'s own message."""
    config = RunConfig(name="p", task=pretrain_task(corpus, fold=7))
    with pytest.raises(Exception, match=r"split 'k3' has no fold 7; it defines folds"):
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


@pytest.mark.parametrize("method", ["mae", "simclr", "vicreg"])
def test_every_method_produces_a_real_validation_loss(method: str, corpus: Path) -> None:
    """There is no ``SslModule.validation_step`` any more to special-case this away: the
    held-out fold goes through the same masked-dataset or two-view-collated contract as
    training, so DsioModule's one generic step produces a genuine val/loss for every
    method, not just the contrastive ones ``ContrastiveModule`` used to keep it for."""
    config = RunConfig(name=f"pre_{method}", seed=0, task=pretrain_task(corpus, method))
    _, metrics = run(config, corpus)
    assert "val_loss" in metrics
    assert np.isfinite(metrics["val_loss"])


@pytest.mark.parametrize("method", ["mae", "simclr", "vicreg"])
def test_frozen_module_validation_loss_is_stable_across_repeated_validations(
    method: str, corpus: Path
) -> None:
    """A fix-round-1 regression test: measured before the fix, five identical validation
    passes over a **frozen** MAE module (never trained, weights never change) reported
    val/loss in [0.97697, 1.01588, 1.00346, 0.99336, 0.99949] -- a 3.90% spread, with even
    the hidden fraction moving between passes (0.3789 vs 0.4038). The validation mask (and,
    for the contrastive methods, the two collated views) were being redrawn from torch's
    global RNG on every call -- exactly the failure the deleted ``self.training`` guard
    existed to prevent, reintroduced one layer down in the dataset. ``WindowDataset.
    mask_seed`` and ``TwoViewCollate.seed``, wired in ``build_loaders`` for the validation
    loader only, fix it: this asserts the spread is now exactly zero, not merely small.
    """
    from lightning import Trainer

    from dsio.data.adapters import SignalExamples
    from dsio.data.store import data_root
    from dsio.data.views import load_or_build
    from dsio.splits.folds import load_folds, split_path
    from dsio.train.ssl_task import build_loaders, build_module

    task = pretrain_task(corpus, method)
    store = SignalStore(data_root() / task.store)
    index = load_or_build(store, task.window)
    examples = SignalExamples(store, index)
    folds = load_folds(examples, split_path(task.splits_root, task.split))
    fold = next(f for f in folds if f.index == task.fold)

    module, _ = build_module(task, channels=store.channels, length=task.window.length)
    module.eval()
    _, val_loader = build_loaders(task, store, index, fold, seed=0)
    assert val_loader is not None

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    losses = [
        trainer.validate(module, val_loader, verbose=False)[0]["val/loss"] for _ in range(5)
    ]
    assert losses == [losses[0]] * 5, f"{method}: val/loss moved across repeated passes: {losses}"


def test_pretraining_writes_no_held_out_predictions(corpus: Path) -> None:
    """Pretraining deliberately produces no scored artifact. A masked-reconstruction MSE on
    held-out windows is a number nobody should act on, so it is not written.

    Before decision 6 this asserted that `read_report` found no evaluation report. That
    function is gone with the fold loop, but the property is unchanged: what pretraining
    must not leave behind is now `predictions.npz`, the file `pool_folds` would pick up.
    """
    config = RunConfig(name="pre", seed=0, task=pretrain_task(corpus))
    active, _ = run(config, corpus)
    assert not (active.artifacts_dir / PREDICTIONS_FILE).exists()


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
    """The message is `SplitFile.fold`'s, reused via `require_fold` rather than
    hand-rolled a second time here, so it names every fold the family actually has."""
    config = RunConfig(name="pre", seed=0, task=pretrain_task(corpus, fold=7))
    with pytest.raises(ValueError, match=r"split 'k3' has no fold 7; it defines folds"):
        run(config, corpus)


def test_a_bad_fold_fails_before_the_store_even_opens(corpus: Path) -> None:
    """Proves the guard runs before any data loading, not merely that it eventually
    fires: a store that does not exist would raise `StoreError` first if the fold check
    ran any later than the top of `run_ssl_pretrain` -- which is where it used to run,
    after the store, index and examples were already built."""
    config = RunConfig(
        name="pre", seed=0, task=pretrain_task(corpus, fold=7, store="does-not-exist")
    )
    with pytest.raises(ValueError, match=r"split 'k3' has no fold 7; it defines folds"):
        run(config, corpus)


# --- MLflow: decision 7's "a run fails without MLflow" -------------------------------


def test_preflight_fails_when_mlflow_is_unreachable(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:59999")
    config = RunConfig(name="pre", seed=0, task=pretrain_task(corpus))
    with pytest.raises(MlflowUnavailableError, match="localhost:59999"):
        check(config)


def test_mlflow_unreachable_fails_the_same_way_even_without_preflight(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:59999")
    config = RunConfig(name="pre", seed=0, task=pretrain_task(corpus))
    with pytest.raises(MlflowUnavailableError, match="localhost:59999"):
        run(config, corpus)


def test_mlflow_unreachable_fails_before_the_store_even_opens(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the guard runs before any data loading -- and before `require_fold` --
    not merely that it eventually fires: a nonexistent store on a bad fold would raise a
    fold or store error first if the MLflow check ran any later than the very top of
    `run_ssl_pretrain`."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:59999")
    config = RunConfig(
        name="pre", seed=0, task=pretrain_task(corpus, fold=7, store="does-not-exist")
    )
    with pytest.raises(MlflowUnavailableError, match="localhost:59999"):
        run(config, corpus)


def test_a_completed_pretrain_run_logs_its_metrics_to_mlflow(corpus: Path) -> None:
    """A refusal-only suite proves half the guard. With MLflow reachable -- the `file:`
    backend every test in this suite already uses, per `tests/conftest.py` -- a run must
    not merely be *allowed* to proceed: its metrics must actually land in MLflow, through
    the same `MLFlowLogger` the Trainer streams `self.log(...)` calls through."""
    config = RunConfig(name="pre-mlflow-logs", seed=0, task=pretrain_task(corpus))
    _, metrics = run(config, corpus)

    from mlflow.tracking import MlflowClient

    client = MlflowClient(resolve_tracking_uri())
    experiment = client.get_experiment_by_name(config.name)
    assert experiment is not None
    mlflow_runs = client.search_runs([experiment.experiment_id])
    assert len(mlflow_runs) == 1
    logged = mlflow_runs[0].data.metrics
    for name, value in metrics.items():
        assert logged[name] == pytest.approx(value)
    # `val/loss` is never in `metrics` (`run_ssl_pretrain`'s own returned dict) -- it only
    # ever reaches MLflow if the Trainer's own `self.log(...)` calls were actually streamed
    # through `mlflow_logger`, i.e. only if `Trainer(..., logger=mlflow_logger)` really
    # wired the two together, not merely if the final `log_metrics` call happened to fire.
    assert "val/loss" in logged


# --- the handoff --------------------------------------------------------------------


def downstream_task(root: Path, encoder: EncoderRef | None) -> TorchTask:
    return TorchTask(
        store="tone",
        window=WINDOW,
        labels="tone",
        split="k3",
        fold=0,
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
    with np.load(active.artifacts_dir / "predictions.npz", allow_pickle=False) as data:
        assert int(data["fold"]) == 0


def test_a_tampered_digest_fails_closed(corpus: Path, pretrained: EncoderRef) -> None:
    """There is no path to hardcode and no way to say 'latest', so the remaining risk is a
    swapped artifact — which the registry re-hashes and refuses."""
    from dsio.model.registry import BACKBONES

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
    from dsio.model.registry import BACKBONES

    backbone = BACKBONES.get("conv1d")(channels=2, hidden=8, out_dim=16, depth=1)
    report = load_encoder(pretrained, backbone=backbone)
    assert report["loaded_tensors"] > 0
    assert report["frozen_parameters"] > 0
    assert not any(p.requires_grad for p in backbone.parameters())
    assert backbone.training is False


def test_finetuning_leaves_the_encoder_trainable(corpus: Path, pretrained: EncoderRef) -> None:
    """A probe measures what the representation already contains; a finetune measures what
    it is a good starting point for. Reporting one as the other overstates the result."""
    from dsio.model.registry import BACKBONES

    backbone = BACKBONES.get("conv1d")(channels=2, hidden=8, out_dim=16, depth=1)
    load_encoder(pretrained.model_copy(update={"freeze": False}), backbone=backbone)
    assert all(p.requires_grad for p in backbone.parameters())


def test_loading_into_a_different_architecture_is_refused(
    corpus: Path, pretrained: EncoderRef
) -> None:
    """Silently loading a subset of weights produces a model that is part pretrained and
    part random, and reports as though it were fully pretrained."""
    from dsio.model.registry import BACKBONES

    mismatched = BACKBONES.get("conv1d")(channels=2, hidden=64, out_dim=16, depth=3)
    with pytest.raises(Exception):
        load_encoder(pretrained, backbone=mismatched)


def test_the_loaded_weights_actually_differ_from_a_fresh_init(
    corpus: Path, pretrained: EncoderRef
) -> None:
    """Guards against a load that silently no-ops and leaves the model randomly
    initialised — every other test here would still pass."""
    import torch

    from dsio.model.registry import BACKBONES

    fresh = BACKBONES.get("conv1d")(channels=2, hidden=8, out_dim=16, depth=1)
    before = [p.detach().clone() for p in fresh.parameters()]
    load_encoder(pretrained, backbone=fresh)
    assert not all(
        torch.equal(a, b) for a, b in zip(before, fresh.parameters(), strict=True)
    )
