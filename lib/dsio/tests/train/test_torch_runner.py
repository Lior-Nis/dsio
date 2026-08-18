"""The torch runner: Lightning's loop inside one fold, and the failures it must not repeat.

Trains real models on a tiny synthetic corpus. Slow relative to a unit test and worth it —
the claim being made is that cross-validation and Lightning compose, and only running both
tests that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from dsio.config.schema import RunConfig  # noqa: E402
from dsio.data import SignalStore, WindowSpec, entity_examples  # noqa: E402
from dsio.data.store import DATA_ROOT_ENV  # noqa: E402
from dsio.eval import read_report  # noqa: E402
from dsio.nn import LABELS, labels  # noqa: E402
from dsio.runs import RunLedger  # noqa: E402
from dsio.splits import SplitSpec, StratifyKey, write_splits  # noqa: E402
from dsio.train import check, execute  # noqa: E402
from dsio.train.torch_task import (  # noqa: E402
    Component,
    TorchTask,
    TrainerConfig,
    build_callbacks,
    build_module,
    sanitise_metric,
)


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Eight groups; half carry a tone in channel 0, and that is the label."""
    root = tmp_path / "stores"
    monkeypatch.setenv(DATA_ROOT_ENV, str(root))
    rng = np.random.default_rng(0)
    with SignalStore.builder(root / "tone", channels=2) as builder:
        for group in range(8):
            positive = group % 2 == 0
            t = np.arange(1200) / 100.0
            signal = (rng.standard_normal((1200, 2)) * 0.5).astype("float32")
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

    store = SignalStore(root / "tone")
    write_splits(
        entity_examples(store),
        SplitSpec(
            scheme="stratified_kfold",
            k=3,
            seed=0,
            stratify=(StratifyKey(name="positive", kind="categorical"),),
        ),
        name="k3",
        root=tmp_path / "splits",
    )
    return tmp_path


def make_task(root: Path, **overrides) -> TorchTask:  # type: ignore[no-untyped-def]
    defaults = dict(
        store="tone",
        window=WindowSpec(length=128, stride=64, label_policy="majority"),
        labels="tone",
        split="k3",
        splits_root=root / "splits",
        backbone=Component(name="conv1d", params={"hidden": 8, "out_dim": 8, "depth": 1}),
        head=Component(name="linear", params={"out_dim": 2}),
        loss=Component(name="cross_entropy", params={"threshold": 0.5}),
        transform=Component(name="instance_standardize"),
        batch_size=32,
        metrics=("accuracy", "roc_auc"),
        trainer=TrainerConfig(max_epochs=2, accelerator="cpu", devices=1, checkpoint=False),
    )
    return TorchTask(**{**defaults, **overrides})


# --- pre-flight ----------------------------------------------------------------------


def test_preflight_catches_a_typo_before_any_data_is_read(corpus: Path) -> None:
    """Deferred validation surfaces a typo only after the corpus has loaded."""
    config = RunConfig(
        name="typo", task=make_task(corpus, backbone=Component(name="conv1D"))
    )
    with pytest.raises(KeyError, match="did you mean"):
        check(config)


def test_preflight_requires_the_split_files_to_exist(corpus: Path) -> None:
    """Splits are provenance; their absence is an error, not licence to invent some."""
    config = RunConfig(name="nosplit", task=make_task(corpus, split="never_made"))
    with pytest.raises(Exception, match="dsio splits make"):
        check(config)


def test_a_label_policy_of_none_is_rejected_at_config_time(corpus: Path) -> None:
    with pytest.raises(ValueError, match="label_policy is 'none'"):
        make_task(corpus, window=WindowSpec(length=128, stride=64))


def test_an_unknown_metric_is_caught_before_training(corpus: Path) -> None:
    config = RunConfig(name="bad", task=make_task(corpus, metrics=("accuarcy",)))
    with pytest.raises(KeyError, match="unknown metric"):
        check(config)


# --- three failures that cost real time ----------------------------------------------


def test_metric_names_are_sanitised_for_filenames() -> None:
    """A / inside a format field is a path separator, so a metric name creates directories."""
    assert sanitise_metric("val/loss") == "val_loss"
    assert sanitise_metric("metrics/val_ap") == "metrics_val_ap"
    assert "/" not in sanitise_metric("a/b\\c=d")


def test_a_failing_callback_is_never_swallowed(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catching everything and logging a warning means a broken
    ModelCheckpoint silently disables checkpointing for a multi-hour run and the loss of
    the weights is discovered days later.

    Patching the constructor is the honest way to test this: Lightning tolerates most bad
    configurations, so a "realistic" broken config would prove nothing about whether the
    exception path swallows.
    """
    import lightning.pytorch.callbacks as callbacks

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("checkpoint directory is not writable")

    monkeypatch.setattr(callbacks, "ModelCheckpoint", explode)
    task = make_task(corpus, trainer=TrainerConfig(checkpoint=True))
    with pytest.raises(RuntimeError, match="not writable"):
        build_callbacks(task, corpus / "ckpt")


def test_checkpoint_filenames_contain_no_path_separator(corpus: Path) -> None:
    task = make_task(corpus, trainer=TrainerConfig(checkpoint=True, monitor="val/loss"))
    callbacks = build_callbacks(task, corpus / "ckpt")
    checkpoint = next(cb for cb in callbacks if hasattr(cb, "filename"))
    prefix = checkpoint.filename.split("{")[0]
    assert "/" not in prefix


def test_each_fold_gets_a_fresh_module(corpus: Path) -> None:
    """Not a reset one. load_state_dict back to a snapshot looks equivalent and is not:
    optimiser state and BatchNorm running statistics survive it, so fold 2 would start
    from fold 1's normalisation."""
    task = make_task(corpus)
    first = build_module(task, channels=2, length=128)
    second = build_module(task, channels=2, length=128)
    assert first is not second
    assert first.backbone is not second.backbone


# --- end to end ------------------------------------------------------------------------


def test_the_runner_produces_the_same_artifact_contract(corpus: Path, tmp_path: Path) -> None:
    """The point of the fold loop: a torch run and a tabular run are read identically."""
    config = RunConfig(name="tone", seed=0, task=make_task(corpus))
    check(config)

    ledger = RunLedger(tmp_path / "runs")
    run = ledger.start(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with run:
        metrics = execute(config, run)

    assert set(metrics) >= {"accuracy", "roc_auc", "coverage"}
    report, oof = read_report(run.artifacts_dir)
    assert report.n_folds == 3
    assert report.coverage == 1.0
    assert report.fold_fingerprint is not None
    assert oof is not None and len(oof) == report.predicted_rows
    assert sorted(oof.row_id.tolist()) == list(range(report.n_rows))


def test_the_runner_learns_a_separable_signal(corpus: Path, tmp_path: Path) -> None:
    """A pipeline that runs but cannot learn a tone from noise is wired wrong somewhere.

    The one test here that asserts on a *number* rather than a structure. It exists because
    every other test would pass on a chain that silently detached the gradient, shuffled
    labels against windows, or normalised the signal away — the plumbing would look
    perfect and the model would learn nothing.
    """
    task = make_task(
        corpus,
        lr=3e-3,
        trainer=TrainerConfig(
            max_epochs=30, accelerator="cpu", devices=1, checkpoint=False
        ),
    )
    config = RunConfig(name="tone", seed=0, task=task)
    ledger = RunLedger(tmp_path / "runs")
    run = ledger.start(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with run:
        metrics = execute(config, run)
    assert metrics["roc_auc"] > 0.9


def test_no_group_is_both_trained_on_and_tested(corpus: Path, tmp_path: Path) -> None:
    """The leakage guarantee, verified at the level the runner actually consumes."""
    from dsio.data import SignalExamples
    from dsio.data.views import load_or_build
    from dsio.splits import fold_paths, load_folds

    store = SignalStore(Path(corpus) / "stores" / "tone")
    task = make_task(corpus)
    row_labels = LABELS.get("tone")(store)
    index = load_or_build(store, task.window, labels=row_labels)
    folds = load_folds(SignalExamples(store, index), fold_paths(task.splits_root, task.split))

    groups = index.groups
    for fold in folds:
        assert not (set(groups[fold.train]) & set(groups[fold.test]))


def test_running_a_subset_of_folds_keeps_their_numbers(corpus: Path, tmp_path: Path) -> None:
    """fold1 must mean fold1 whether or not fold0 was run, or two runs stop comparing."""
    config = RunConfig(name="subset", seed=0, task=make_task(corpus, folds=(1, 2)))
    ledger = RunLedger(tmp_path / "runs")
    run = ledger.start(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with run:
        execute(config, run)
    report, _ = read_report(run.artifacts_dir)
    assert [fold.fold for fold in report.folds] == [1, 2]
    assert report.coverage < 1.0, "a subset of folds cannot cover every window"


def test_asking_for_folds_that_do_not_exist_fails_loudly(corpus: Path, tmp_path: Path) -> None:
    config = RunConfig(name="ghost", seed=0, task=make_task(corpus, folds=(9,)))
    ledger = RunLedger(tmp_path / "runs")
    run = ledger.start(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with pytest.raises(ValueError, match="match none of the split family"):
        execute(config, run)


def test_the_run_records_which_corpus_it_read(corpus: Path, tmp_path: Path) -> None:
    """A metric without the digest of the data behind it is not reproducible provenance."""
    import json

    config = RunConfig(name="prov", seed=0, task=make_task(corpus))
    ledger = RunLedger(tmp_path / "runs")
    run = ledger.start(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with run:
        execute(config, run)
    payload = json.loads((run.artifacts_dir / "windows.json").read_text())
    store = SignalStore(Path(corpus) / "stores" / "tone")
    assert payload["store_sha256"] == store.manifest().signal_sha256
    assert payload["split"] == "k3"
