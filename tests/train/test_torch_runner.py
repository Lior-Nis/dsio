"""The torch runner: Lightning's loop inside one fold, and the failures it must not repeat.

Trains real models on a tiny synthetic corpus. Slow relative to a unit test and worth it —
the claim being made is that cross-validation and Lightning compose, and only running both
tests that.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from dsio.config.schema import RunConfig  # noqa: E402
from dsio.data.adapters import entity_examples  # noqa: E402
from dsio.data.store import DATA_ROOT_ENV, SignalStore  # noqa: E402
from dsio.data.views import WindowSpec  # noqa: E402
from dsio.model.registry import LABELS, labels  # noqa: E402
from dsio.runs.record import start_run  # noqa: E402
from dsio.splits.models import SplitFile, SplitFold  # noqa: E402
from dsio.train.runner import check, execute  # noqa: E402
from dsio.train.torch_task import (  # noqa: E402
    Component,
    TorchTask,
    TrainerConfig,
    _fold_invariant_config_hash,
    build_callbacks,
    build_module,
    sanitise_metric,
)
from dsio.train.tracking import MlflowUnavailableError, resolve_tracking_uri  # noqa: E402


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

    # Three hand-picked folds over the eight groups, each mixing positive and negative
    # subjects across train/val/test so the separable-signal test below has both classes
    # to learn from and to score against in every part.
    store = SignalStore(root / "tone")
    digest = entity_examples(store).digest
    folds = [
        {"test": ["p0", "p1", "p2"], "val": ["p3"], "train": ["p4", "p5", "p6", "p7"]},
        {"test": ["p3", "p4", "p5"], "val": ["p6"], "train": ["p0", "p1", "p2", "p7"]},
        {"test": ["p6", "p7"], "val": ["p0"], "train": ["p1", "p2", "p3", "p4", "p5"]},
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


def make_task(root: Path, **overrides) -> TorchTask:  # type: ignore[no-untyped-def]
    defaults = dict(
        store="tone",
        window=WindowSpec(length=128, stride=64, label_policy="majority"),
        labels="tone",
        split="k3",
        fold=0,
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


# --- what identifies "the same experiment" -------------------------------------------


def test_naming_each_folds_run_distinctly_still_pools(corpus: Path) -> None:
    """The identity must ignore what merely labels a run.

    `dsio run p task.fold=$i --name exp-fold$i` is a natural way to drive a shell loop, and
    every one of those runs belongs to one cross-validation. If `name` or `tags` reached the
    hash, pooling a perfectly valid CV would be refused -- and a guard that rejects the
    obvious workflow gets worked around, which protects nothing at all.
    """
    left = RunConfig(name="exp-fold0", tags=("sweep", "a"), task=make_task(corpus, fold=0))
    right = RunConfig(name="exp-fold1", tags=("sweep", "b"), task=make_task(corpus, fold=1))
    assert _fold_invariant_config_hash(left) == _fold_invariant_config_hash(right)


def test_a_different_seed_is_a_different_experiment(corpus: Path) -> None:
    """`seed` stays in the identity where `name` does not: it changes the weights, so two
    folds seeded differently are two different training runs, not one CV."""
    left = RunConfig(name="exp", seed=1, task=make_task(corpus, fold=0))
    right = RunConfig(name="exp", seed=2, task=make_task(corpus, fold=1))
    assert _fold_invariant_config_hash(left) != _fold_invariant_config_hash(right)


def test_a_different_hyperparameter_is_a_different_experiment(corpus: Path) -> None:
    """The identity is derived from the task, not hardcoded to a constant -- which is the
    way this guard would fail while every refusal test above still passed."""
    left = RunConfig(name="exp", task=make_task(corpus, fold=0, lr=1e-3))
    right = RunConfig(name="exp", task=make_task(corpus, fold=1, lr=5e-3))
    assert _fold_invariant_config_hash(left) != _fold_invariant_config_hash(right)


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
    with pytest.raises(Exception, match="commit a split file"):
        check(config)


def test_preflight_rejects_a_fold_the_split_family_does_not_declare(corpus: Path) -> None:
    """The message is `SplitFile.fold`'s, reused via `require_fold` rather than
    hand-rolled a second time here -- the same message `SslPretrainTask` produces for
    the same mistake (see `test_pretraining_on_a_missing_fold_fails_loudly`)."""
    config = RunConfig(name="ghost-fold", task=make_task(corpus, fold=7))
    with pytest.raises(Exception, match=r"split 'k3' has no fold 7; it defines folds"):
        check(config)


def test_a_bad_fold_fails_the_same_way_even_without_preflight(
    corpus: Path, tmp_path: Path
) -> None:
    """Every caller that reaches `execute()` gets the same guard the CLI's pre-flight
    gives it, not just the one that remembered to call `check()` first -- every test
    below this one calls `execute()` directly, exactly like this."""
    config = RunConfig(name="ghost-fold", task=make_task(corpus, fold=7))
    run = start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with pytest.raises(Exception, match=r"split 'k3' has no fold 7; it defines folds"):
        execute(config, run)


def test_a_bad_fold_fails_before_the_store_even_opens(corpus: Path, tmp_path: Path) -> None:
    """Proves the guard runs before any data loading, not merely that it eventually
    fires: a store that does not exist would raise `StoreError` first if the fold check
    ran any later than the top of `run_torch`."""
    config = RunConfig(
        name="ghost-fold", task=make_task(corpus, fold=7, store="does-not-exist")
    )
    run = start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with pytest.raises(Exception, match=r"split 'k3' has no fold 7; it defines folds"):
        execute(config, run)


# --- MLflow: decision 7's "a run fails without MLflow" -------------------------------


def test_preflight_fails_when_mlflow_is_unreachable(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:59999")
    config = RunConfig(name="unreachable", task=make_task(corpus))
    with pytest.raises(MlflowUnavailableError, match="localhost:59999"):
        check(config)


def test_mlflow_unreachable_fails_the_same_way_even_without_preflight(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every caller that reaches `execute()` gets the same guard the CLI's pre-flight
    gives it -- the same property `test_a_bad_fold_fails_the_same_way_even_without_
    preflight` proves for `require_fold`, above."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:59999")
    config = RunConfig(name="unreachable", task=make_task(corpus))
    run = start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with pytest.raises(MlflowUnavailableError, match="localhost:59999"):
        execute(config, run)


def test_mlflow_unreachable_fails_before_the_store_even_opens(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the guard runs before any data loading -- and before `require_fold` --
    not merely that it eventually fires: a nonexistent store on a bad fold would raise a
    fold or store error first if the MLflow check ran any later than the very top of
    `run_torch`. This is the stronger claim decision 7 asks for: not "it raises
    eventually", but "it raises before anything expensive has happened"."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:59999")
    config = RunConfig(
        name="unreachable", task=make_task(corpus, fold=7, store="does-not-exist")
    )
    run = start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with pytest.raises(MlflowUnavailableError, match="localhost:59999"):
        execute(config, run)


def test_a_completed_run_logs_its_metrics_to_mlflow(corpus: Path, tmp_path: Path) -> None:
    """A refusal-only suite proves half the guard. With MLflow reachable -- the `file:`
    backend every test in this suite already uses, per `tests/conftest.py` -- a run must
    not merely be *allowed* to proceed: its held-out metrics must actually land in
    MLflow, through the same `MLFlowLogger` the Trainer streams `self.log(...)` calls
    through (`build_mlflow_logger`, `dsio.train.tracking`)."""
    config = RunConfig(name="mlflow-logs", task=make_task(corpus))
    run = start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    metrics = execute(config, run)

    from mlflow.tracking import MlflowClient

    client = MlflowClient(resolve_tracking_uri())
    experiment = client.get_experiment_by_name(config.name)
    assert experiment is not None
    mlflow_runs = client.search_runs([experiment.experiment_id])
    assert len(mlflow_runs) == 1
    logged = mlflow_runs[0].data.metrics
    for name, value in metrics.items():
        assert logged[name] == pytest.approx(value)
    # `val/loss` is never in `metrics` (the held-out `task.metrics` dict) -- it only ever
    # reaches MLflow if the Trainer's own `self.log(...)` calls were actually streamed
    # through `mlflow_logger`, i.e. only if `Trainer(..., logger=mlflow_logger)` really
    # wired the two together, not merely if the final `log_metrics` call happened to fire.
    assert "val/loss" in logged


def test_a_crash_while_stamping_provenance_still_fails_the_mlflow_run(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug 3: `stamp_provenance` used to run *outside* the `try`/`except BaseException`
    that flips a crashed run to MLflow's `FAILED` status. `stamp_provenance` is what
    actually creates the MLflow run (`mlflow_logger.experiment`'s lazy `create_run`,
    accessed on first `.run_id`/`.experiment` read) -- so a crash anywhere in the rest of
    its own work (logging the config/reproduce-script/diff artifacts, tagging
    `config_hash`) left a run that unambiguously exists, sitting in MLflow's default
    `RUNNING` status forever: neither finished nor failed, the record lying in a third
    way decision 7 does not allow.

    Reproduces exactly that shape: a fake `stamp_provenance` that does the one thing the
    real one does first -- reads `mlflow_logger.run_id` (creating the run) and records it
    onto `run.mlflow_run_id`, precisely mirroring `dsio.train.tracking.stamp_provenance`'s
    own first two lines -- then raises, before any of its own artifact/tag logging runs.
    That run must still end up `FAILED`, not stuck `RUNNING`.
    """
    import dsio.train.torch_task as torch_task

    def fake_stamp_provenance(run: object, mlflow_logger: object) -> None:
        mlflow_run_id = mlflow_logger.run_id  # type: ignore[attr-defined]
        run.mlflow_run_id = mlflow_run_id  # type: ignore[attr-defined]
        raise RuntimeError("boom: crash mid-provenance-stamp, after the run was created")

    monkeypatch.setattr(torch_task, "stamp_provenance", fake_stamp_provenance)

    config = RunConfig(name="prov-crash", seed=0, task=make_task(corpus))
    run = start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with pytest.raises(RuntimeError, match="boom: crash mid-provenance-stamp"):
        execute(config, run)

    assert run.mlflow_run_id is not None

    from mlflow.tracking import MlflowClient

    client = MlflowClient(resolve_tracking_uri())
    mlflow_run = client.get_run(run.mlflow_run_id)
    assert mlflow_run.info.status == "FAILED"


def test_two_folds_of_one_config_land_in_one_mlflow_experiment(
    corpus: Path, tmp_path: Path
) -> None:
    """`build_mlflow_logger` sets `experiment_name=config.name`, which is what makes a
    shell loop over folds (`dsio run p task.fold=$i`, same `name` every invocation)
    group natively in MLflow, rather than scattering one experiment per run."""
    from mlflow.tracking import MlflowClient

    for fold in (0, 1):
        config = RunConfig(name="cv-sweep", task=make_task(corpus, fold=fold))
        run = start_run(
            name=config.name,
            config=config.to_dict(),
            config_hash=config.config_hash,
            seed=config.seed,
        )
        execute(config, run)

    client = MlflowClient(resolve_tracking_uri())
    experiment = client.get_experiment_by_name("cv-sweep")
    assert experiment is not None
    mlflow_runs = client.search_runs([experiment.experiment_id])
    assert len(mlflow_runs) == 2


def test_a_label_policy_of_none_is_rejected_at_config_time(corpus: Path) -> None:
    with pytest.raises(ValueError, match="label_policy is 'none'"):
        make_task(corpus, window=WindowSpec(length=128, stride=64))


def test_predict_embedding_is_rejected_at_config_time(corpus: Path) -> None:
    """``predict`` used to exist only on ``SslPretrainTask``, whose runner never calls
    ``trainer.predict`` -- a knob with no wire behind it. It moved here, to the task whose
    ``fit_predict`` actually calls ``trainer.predict`` (below), which is what makes it
    reachable: constructing this task with ``predict="embedding"`` now fails at config
    time with an explanation, rather than silently building a module whose predict_step
    would (if ever exercised through this runner) return ``{"embedding": ...}`` and blow
    up ``_assemble``'s ``batch["prediction"]`` lookup with an opaque KeyError instead.
    """
    with pytest.raises(ValueError, match="predict='embedding' is not supported"):
        make_task(corpus, predict="embedding")


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
    """Lightning substitutes ``{metric_name:.4f}`` groups with a formatted number before the
    filename is used as a path, so a slash inside braces is harmless. A slash in the literal
    text around those groups is not: it becomes a directory separator, silently creating
    nested checkpoint directories that look like a corrupted run. ``sanitise_metric`` exists
    to keep the monitor name out of that literal text, so the assertion has to inspect the
    literal text specifically, not just whatever precedes the first ``{``.
    """
    task = make_task(corpus, trainer=TrainerConfig(checkpoint=True, monitor="val/loss"))
    callbacks = build_callbacks(task, corpus / "ckpt")
    checkpoint = next(cb for cb in callbacks if hasattr(cb, "filename"))
    literal_text = re.sub(r"\{[^}]*\}", "", checkpoint.filename)
    assert "/" not in literal_text


def test_each_fold_gets_a_fresh_module(corpus: Path) -> None:
    """Not a reset one. load_state_dict back to a snapshot looks equivalent and is not:
    optimiser state and BatchNorm running statistics survive it, so fold 2 would start
    from fold 1's normalisation."""
    task = make_task(corpus)
    first = build_module(task, channels=2, length=128)
    second = build_module(task, channels=2, length=128)
    assert first is not second
    assert first.backbone is not second.backbone


def test_build_module_wires_predict_into_the_module(corpus: Path) -> None:
    """``build_module`` -- the function ``fit_predict`` (below) actually calls, not a
    hand-built ``DsioModule`` -- must carry ``task.predict`` onto the module it returns.
    Before this, ``DsioModule``'s own default ('prediction') was reached by omission
    here, never by this task naming it; a caller had no way to see or change what a
    torch run's ``trainer.predict`` step reports.

    ``predict="embedding"`` is rejected at config time (see the test above), so the only
    way to check ``build_module`` actually threads whatever value ``task.predict`` holds
    -- rather than only ever happening to match ``DsioModule``'s own default -- is to
    reach a non-default value without going through that validator.
    ``model_copy(update=...)`` does exactly that: it does not re-run validators, unlike
    the constructor.
    """
    task = make_task(corpus)
    assert task.predict == "prediction"
    module = build_module(task, channels=2, length=128)
    assert module.predict == "prediction"

    embedding_task = task.model_copy(update={"predict": "embedding"})
    embedding_module = build_module(embedding_task, channels=2, length=128)
    assert embedding_module.predict == "embedding"


# --- end to end ------------------------------------------------------------------------


def test_the_runner_produces_the_same_artifact_contract(corpus: Path, tmp_path: Path) -> None:
    """One run, one fold: `predictions.npz` carries what a pooling reader needs later."""
    config = RunConfig(name="tone", seed=0, task=make_task(corpus))
    check(config)

    run = start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with run:
        metrics = execute(config, run)

    assert set(metrics) >= {"accuracy", "roc_auc"}

    with np.load(run.artifacts_dir / "predictions.npz", allow_pickle=False) as data:
        assert int(data["fold"]) == 0
        assert str(data["split"]) == "k3"
        assert len(data["row_id"]) == len(data["y_true"]) == len(data["y_pred"])
        assert "y_score" in data.files
        store = SignalStore(Path(corpus) / "stores" / "tone")
        assert str(data["split_digest"]) == store.manifest().signal_sha256
        # Critical 1: the fifth invariant `cross_validate` used to get for free from holding
        # one closure and one `Examples` -- a pooling reader needs both recorded per fold.
        assert str(data["window_digest"]) == config.task.window.digest  # type: ignore[attr-defined]
        assert str(data["config_identity"])


def test_pool_folds_refuses_folds_trained_under_different_backbones(
    corpus: Path, tmp_path: Path
) -> None:
    """Critical 1, end to end. The reviewer demonstrated the failure with real runs: three
    folds trained with different `hidden`/`depth`/`lr` pooled without complaint into a
    plausible-looking `{'accuracy': 0.375, 'roc_auc': 0.500}`. Two real single-fold runs
    here, one per backbone width, must refuse to pool rather than repeat that."""
    from dsio.eval.pool import pool_folds

    run_dirs = []
    for fold, hidden in ((0, 8), (1, 16)):
        task = make_task(
            corpus,
            fold=fold,
            backbone=Component(
                name="conv1d", params={"hidden": hidden, "out_dim": 8, "depth": 1}
            ),
        )
        config = RunConfig(name="mismatch", seed=0, task=task)
        run = start_run(
            name=config.name,
            config=config.to_dict(),
            config_hash=config.config_hash,
            seed=config.seed,
        )
        with run:
            execute(config, run)
        run_dirs.append(run.artifacts_dir)

    with pytest.raises(Exception, match="config"):
        pool_folds(run_dirs, metrics=("accuracy",))


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
    run = start_run(
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
    from dsio.data.adapters import SignalExamples
    from dsio.data.views import load_or_build
    from dsio.splits.folds import load_folds, split_path

    store = SignalStore(Path(corpus) / "stores" / "tone")
    task = make_task(corpus)
    row_labels = LABELS.get("tone")(store)
    index = load_or_build(store, task.window, labels=row_labels)
    folds = load_folds(SignalExamples(store, index), split_path(task.splits_root, task.split))

    groups = index.groups
    for fold in folds:
        assert not (set(groups[fold.train]) & set(groups[fold.test]))


def test_running_one_fold_writes_only_that_folds_predictions(
    corpus: Path, tmp_path: Path
) -> None:
    """fold1 must mean fold1, and running it alone must not touch fold0's or fold2's rows --
    the property fold-as-process needs so that N single-fold runs can be pooled later
    without one run's predictions silently covering another's positions."""
    from dsio.data.adapters import SignalExamples
    from dsio.data.views import load_or_build
    from dsio.splits.folds import load_folds, split_path

    config = RunConfig(name="subset", seed=0, task=make_task(corpus, fold=1))
    run = start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with run:
        execute(config, run)

    store = SignalStore(Path(corpus) / "stores" / "tone")
    task = make_task(corpus)
    index = load_or_build(store, task.window, labels=LABELS.get("tone")(store))
    folds = load_folds(SignalExamples(store, index), split_path(task.splits_root, task.split))
    fold1 = next(f for f in folds if f.index == 1)

    with np.load(run.artifacts_dir / "predictions.npz", allow_pickle=False) as data:
        assert int(data["fold"]) == 1
        assert sorted(data["row_id"].tolist()) == sorted(fold1.test.tolist())
        assert len(data["row_id"]) < len(index), "one fold cannot cover every window"


def test_the_run_records_which_corpus_it_read(corpus: Path, tmp_path: Path) -> None:
    """A metric without the digest of the data behind it is not reproducible provenance."""
    import json

    config = RunConfig(name="prov", seed=0, task=make_task(corpus))
    run = start_run(
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
    assert payload["fold"] == 0


# --- guards carried over from `cross_validate` -----------------------------------------


def test_a_short_prediction_fails_the_run_instead_of_scoring_silently(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard 1, inherited from the deleted `cross_validate` loop: predictions that do not
    line up with the fold they came from must fail the run, not produce a quiet,
    merely-disappointing number.

    `_assemble` already refuses a *row* mismatch (see the loader/fold disagreement check
    in `_assemble` itself), which is a stronger property than a plain count. This test
    proves the count guard fires on its own terms even so, by monkeypatching `_assemble`
    to bypass its own check and return one prediction short.
    """
    import dsio.train.torch_task as torch_task
    from dsio.eval.contract import FoldPrediction

    real_assemble = torch_task._assemble

    def truncated_assemble(batches: object, fold: object, window_labels: object) -> FoldPrediction:
        result = real_assemble(batches, fold, window_labels)
        score = None if result.y_score is None else result.y_score[:-1]
        return FoldPrediction(y_true=result.y_true[:-1], y_pred=result.y_pred[:-1], y_score=score)

    monkeypatch.setattr(torch_task, "_assemble", truncated_assemble)

    config = RunConfig(name="short", seed=0, task=make_task(corpus))
    run = start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with pytest.raises(Exception, match="the runner and the fold disagree about what was held out"):
        execute(config, run)


def test_a_fold_that_cannot_be_scored_fails_with_a_split_explanation(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard 4, inherited from the deleted `cross_validate` loop: a `MetricError` -- a
    fold with only one class, or a runner that produced no scores for a ranking metric --
    is re-raised as an `EvalError` pointing at stratification, not surfaced as a bare
    metric exception nobody connects to the split.
    """
    import dsio.train.torch_task as torch_task
    from dsio.eval.metrics import MetricError

    def explode(*args: object, **kwargs: object) -> dict[str, float]:
        raise MetricError("roc_auc is undefined when one class is absent")

    monkeypatch.setattr(torch_task, "compute", explode)

    config = RunConfig(name="unscoreable", seed=0, task=make_task(corpus))
    run = start_run(
        name=config.name,
        config=config.to_dict(),
        config_hash=config.config_hash,
        seed=config.seed,
    )
    with pytest.raises(Exception, match="check stratification"):
        execute(config, run)
