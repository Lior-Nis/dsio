"""The torch runner: one process, one fold (decision 6).

One invocation trains one config against one fold and is linear top to bottom: build config
-> build data -> build module -> ``Trainer.fit`` -> predict -> write artifacts -> stamp
provenance. A project-owned flow invokes this entry point N times for cross-validation,
not a loop inside this file. A ``Trainer`` is created, fitted and discarded within
one call to ``run_torch``; this file never learns how out-of-fold predictions across folds
are accumulated or pooled -- that is a separate reader, over separate runs' artifacts.

The consequence is that this file is short, and almost all of it is configuration. The
parts that are not configuration guard three failures that cost real time:

**Callbacks are never instantiated inside a bare ``except``.** Catching everything and
logging a warning means a misconfigured ``ModelCheckpoint`` silently disables checkpointing
for a multi-hour run, and nobody learns until they go looking for the weights.

**Metric names in filename templates are sanitised.** ``ap{metrics/val_ap:.3f}`` produced
six stray ``checkpoints/checkpoint-epoch=NN-metrics/`` *directories*, because the ``/``
inside the format field became a path separator.

**Checkpoints close over their lineage.** A checkpoint that reloads its encoder from a
hardcoded path fails on a fresh clone. An encoder here is identified by its MLflow run,
artifact path and digest, and verified on load.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from pydantic import Field, model_validator

from dsio.batches import PredictionBatch
from dsio.config.schema import TASKS, TaskConfig
from dsio.contracts import DsioModel, sha256_of
from dsio.data.adapters import SignalExamples
from dsio.data.splits.folds import load_folds, require_fold, split_path
from dsio.data.store import SignalStore, data_root
from dsio.data.views import WindowSpec, load_or_build
from dsio.dataset.dataset import WindowDataset, labelled_dataset, make_loader, make_target_loader
from dsio.eval.contract import PREDICTIONS_FILE, EvalError, Fold, FoldPrediction
from dsio.eval.metrics import METRICS, MetricError, compute
from dsio.model.module import DsioModule
from dsio.model.registry import (
    BACKBONES,
    HEADS,
    LABELS,
    LOSSES,
    PREPROCESSORS,
    TRANSFORMS,
)
from dsio.train.artifacts import ArtifactRef, load_artifact
from dsio.train.assembly import Component as Component
from dsio.train.assembly import accepted_shape_arguments, build_optional_component
from dsio.train.runner import preflight, runner
from dsio.train.tracking import (
    finite_metrics,
    log_run_artifacts,
    require_mlflow,
    tracked_run,
)
from dsio.train.trainer import TrainerConfig as TrainerConfig
from dsio.train.trainer import build_callbacks as build_callbacks
from dsio.train.trainer import build_trainer
from dsio.train.trainer import sanitise_metric as sanitise_metric

if TYPE_CHECKING:
    import torch

    from dsio.config.schema import RunConfig
    from dsio.runs.record import Run

SPLITS_ROOT = Path("splits")


def _fold_invariant_config_hash(config: RunConfig) -> str:
    """Content hash of what produced these predictions, with ``fold`` normalised out.

    Folds of one experiment must agree on everything except which fold they are: same
    backbone, same hyperparameters, same window spec, same everything but ``fold``. A
    pooling reader (`dsio.eval.pool.pool_folds`) writes this per fold into
    ``predictions.npz`` and refuses to pool files whose hashes differ -- the check
    `cross_validate` never had to make, because it only ever held one closure and one
    `Examples` in memory. Structural guarantee then, checked invariant now.

    Covers the task and the seed, and deliberately **not** ``name`` or ``tags``. Those
    label a run; they do not change the model that produced its predictions. Including
    them would make naming each fold's run distinctly in a project-owned flow refuse to
    pool a perfectly valid
    cross-validation. A guard that rejects the obvious workflow gets worked around, and a
    guard that is worked around protects nothing. ``seed`` stays in: it changes the
    weights, so two folds seeded differently really are two different training runs.
    """
    data = config.to_dict()
    identity = {"seed": data["seed"], "task": {**data["task"], "fold": None}}
    return sha256_of(identity)


class EncoderRef(DsioModel):
    """A pinned pretrained encoder, plus what to do with it.

    Wraps :class:`~dsio.train.artifacts.ArtifactRef`, which names a run, a path inside it
    and a content digest — nothing that can drift. An encoder loaded from a hardcoded
    absolute path fails on a fresh clone and — worse — a reproduction on the original
    machine silently picks up whatever has been written there since. A run id and a digest
    cannot do either, and unlike a registry name and version there is no second record of
    the digest that could disagree with the bytes.

    ``freeze`` distinguishes the two experiments people conflate: a *probe* measures what
    the representation already contains, while a *finetune* measures what it is a good
    starting point for. Reporting one as the other is how a pretraining result gets
    overstated.
    """

    run_id: str
    path: str
    digest: str
    freeze: bool = True
    strict: bool = Field(
        default=True,
        description="Require every encoder weight to load. Off only for deliberate surgery.",
    )

    def as_artifact_ref(self) -> ArtifactRef:
        return ArtifactRef(run_id=self.run_id, path=self.path, digest=self.digest)


@TASKS.register("torch")
class TorchTask(TaskConfig):
    """Train and evaluate a torch model on one fold of a canonical store."""

    kind: Literal["torch"] = "torch"

    store: str = Field(description="Store name under the data root.")
    window: WindowSpec
    labels: str = Field(description="Registered per-row label provider.")
    split: str = Field(description="Committed split family under splits_root.")
    splits_root: Path = SPLITS_ROOT
    fold: int = Field(
        description=(
            "Which fold this run trains and predicts. Required, unlike SslPretrainTask's "
            "`fold: int = 0`: pretraining has no 'all folds' alternative reading, so 0 is "
            "a real default there, but a default here would silently pick fold 0 whenever "
            "a caller meant every fold -- exactly the mistake fold-as-process (decision 6) "
            "rules out. One invocation trains one fold; cross-validation is N "
            "invocations from a project-owned flow, not a field on this task."
        ),
    )

    backbone: Component
    head: Component = Component(name="linear")
    loss: Component = Component(name="cross_entropy")
    transform: Component | None = None
    preprocessor: Component | None = None

    encoder: EncoderRef | None = Field(
        default=None,
        description="A pretrained encoder pinned by MLflow run, artifact path and digest.",
    )

    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=0, ge=0)
    trainer: TrainerConfig = TrainerConfig()

    payload_dtype: Literal["long"] | None = Field(
        default=None,
        description="Payload dtype the model receives. None (the default) is float32, "
        "which is what a signal wants whatever its bytes are packed as. 'long' hands the "
        "values through as integers, which is what an embedding lookup needs.",
    )
    metrics: tuple[str, ...] = ("accuracy", "f1_macro")

    @model_validator(mode="after")
    def _check(self) -> TorchTask:
        if not self.metrics:
            raise ValueError(
                "a run with no metrics produces predictions nobody can compare; "
                f"pick from {', '.join(METRICS.names())}"
            )
        if self.window.label_policy == "none":
            raise ValueError(
                "a supervised torch run needs window labels, but the window spec's "
                "label_policy is 'none'; set 'any', 'majority' or 'ratio'"
            )
        return self


# --- pre-flight ---------------------------------------------------------------------


@preflight("torch")
def check_torch(config: RunConfig) -> None:
    """Resolve every name before a single byte of the corpus is read.

    A schema that accepts unknown keys, or a component named by a bare string resolved at
    instantiation time, surfaces a typo only after the data has loaded. On a large corpus
    that is the difference between a typo costing microseconds and costing twenty minutes.

    ``require_mlflow`` runs first, ahead of every registry lookup below: decision 7 makes
    MLflow a run's hard dependency, and a run that cannot write to it should not spend even
    a typo-check's worth of time before saying so.
    """
    require_mlflow()
    task = config.task
    assert isinstance(task, TorchTask)

    LABELS.get(task.labels)
    BACKBONES.get(task.backbone.name)
    HEADS.get(task.head.name)
    LOSSES.get(task.loss.name)
    for optional, registry in (
        (task.transform, TRANSFORMS),
        (task.preprocessor, PREPROCESSORS),
    ):
        if optional is not None:
            registry.get(optional.name)
    for name in task.metrics:
        METRICS.get(name)

    # The split files are provenance, so their absence -- or a fold they do not declare
    # -- is a real error rather than an invitation to generate something unrecorded on
    # the fly. `require_fold` checks both, purely from the committed YAML.
    require_fold(task.splits_root, task.split, task.fold)


# --- assembly -----------------------------------------------------------------------


def build_module(task: TorchTask, *, channels: int, length: int) -> DsioModule:
    """Assemble the component chain for one fold.

    A fresh module per fold, never a reset one. ``load_state_dict`` back to an initial
    snapshot looks equivalent and is not: optimiser state, BatchNorm running statistics and
    any lazily-built buffer survive it, so fold 2 would start from fold 1's normalisation.
    """
    factory = BACKBONES.get(task.backbone.name)
    shape = accepted_shape_arguments(factory, {"channels": channels, "length": length})
    backbone = factory(**{**shape, **task.backbone.params})
    out_dim = getattr(backbone, "out_dim", None)
    head_params = dict(task.head.params)
    if out_dim is not None:
        head_params.setdefault("in_dim", out_dim)

    transform = build_optional_component(task.transform, TRANSFORMS)
    if task.encoder is not None:
        load_encoder(task.encoder, backbone=backbone, transform=transform)

    return DsioModule(
        backbone=backbone,
        head=HEADS.get(task.head.name)(**head_params),
        loss=LOSSES.get(task.loss.name)(**task.loss.params),
        transform=transform,
        preprocessor=build_optional_component(task.preprocessor, PREPROCESSORS),
        lr=task.lr,
        weight_decay=task.weight_decay,
    )


def load_encoder(
    reference: EncoderRef, *, backbone: Any, transform: Any = None
) -> dict[str, int]:
    """Load a pinned encoder into a freshly built chain, verifying it first.

    :func:`~dsio.train.artifacts.load_artifact` re-hashes the bytes and refuses a digest
    mismatch, so a corrupted or swapped encoder cannot be silently trained on top of. Freezing, when
    asked, also puts the backbone in eval mode: a frozen BatchNorm whose running statistics
    keep updating is not frozen, and the difference shows up as a probe that mysteriously
    outperforms its own linear separability.
    """
    import io

    import torch

    payload = load_artifact(reference.as_artifact_ref())
    bundle = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)

    components: dict[str, Any] = {"backbone": backbone}
    if transform is not None:
        components["transform"] = transform

    loaded = 0
    for prefix, component in components.items():
        subset = {
            key.removeprefix(f"{prefix}."): value
            for key, value in bundle["state_dict"].items()
            if key.startswith(f"{prefix}.")
        }
        if not subset:
            continue
        component.load_state_dict(subset, strict=reference.strict)
        loaded += len(subset)

    if reference.strict and loaded == 0:
        raise ValueError(
            f"encoder {reference.path} contained no weights for the "
            "configured components; the backbone almost certainly differs from the one "
            "that was pretrained"
        )

    frozen = 0
    if reference.freeze:
        for component in components.values():
            component.eval()
            for parameter in component.parameters():
                parameter.requires_grad_(False)
                frozen += 1
    return {"loaded_tensors": loaded, "frozen_parameters": frozen}


def payload_dtype_of(task: TorchTask) -> torch.dtype | None:
    """Resolve the config's payload dtype name to the dtype the dataset wants.

    A string is what crosses the config boundary, not a ``torch.dtype``: a config is
    hashed through :func:`~dsio.contracts.hashing.canonical_json`, which encodes
    primitives only, and a run record has to stay readable by something that has never
    imported torch. One name is accepted, deliberately -- a second belongs here when a
    second has a caller, not before.

    torch is imported inside the function, as everywhere else in this module, so that
    importing it costs nothing for a caller only inspecting a config.
    """
    if task.payload_dtype is None:
        return None
    import torch

    return {"long": torch.long}[task.payload_dtype]


# --- run ----------------------------------------------------------------------------


@runner("torch")
def run_torch(config: RunConfig, run: Run) -> dict[str, float]:
    """Train and predict one fold, writing the per-run artifact contract every runner writes.

    Linear top to bottom, per decision 6: build config -> build data -> build module ->
    fit -> predict -> write artifacts -> stamp provenance. Cross-validation is this
    function called N times, from a shell loop or an agent -- not a loop in here.
    """
    # `tracked_run` keeps MLflow reachability first and stamps provenance before any
    # store, model or Trainer work. It also marks an existing run failed if anything in
    # this task-specific body crashes, including after Lightning has finalized success.
    with tracked_run(config, run) as mlflow_logger:
        task = config.task
        assert isinstance(task, TorchTask)

        require_fold(task.splits_root, task.split, task.fold)

        store = SignalStore(data_root() / task.store)
        row_labels = np.asarray(LABELS.get(task.labels)(store))
        if row_labels.size != store.n_rows:
            raise ValueError(
                f"label provider {task.labels!r} returned {row_labels.size} values for "
                f"{store.n_rows} rows; labels must be per-row over the whole store"
            )

        index = load_or_build(store, task.window, labels=row_labels)
        examples = SignalExamples(store, index)
        folds = load_folds(examples, split_path(task.splits_root, task.split))
        # `require_fold` above already guarantees `task.fold` is declared, so this lookup
        # cannot fail on a live split file; kept as an assertion rather than silently
        # trusting it, so a TOCTOU (the file changing between the two reads) still fails
        # loudly. Same pattern as `run_ssl` in `ssl_task.py`.
        fold = next((candidate for candidate in folds if candidate.index == task.fold), None)
        assert fold is not None, (
            f"require_fold guaranteed fold {task.fold} exists in {task.split!r}"
        )

        window_labels = index.labels
        if window_labels is None:
            raise ValueError("the window index carries no labels; check the label policy")

        module = build_module(task, channels=store.channels, length=task.window.length)
        directory = run.artifacts_dir
        payload_dtype = payload_dtype_of(task)

        train_loader = make_target_loader(
            labelled_dataset(
                store,
                index,
                fold.train,
                labels=window_labels,
                payload_dtype=payload_dtype,
            ),
            batch_size=task.batch_size,
            shuffle=True,
            num_workers=task.num_workers,
            seed=config.seed + fold.index,
        )
        validation = fold.val if fold.val is not None and fold.val.size else None
        val_loader = (
            None
            if validation is None
            else make_target_loader(
                labelled_dataset(
                    store,
                    index,
                    validation,
                    labels=window_labels,
                    payload_dtype=payload_dtype,
                ),
                batch_size=task.batch_size,
                num_workers=task.num_workers,
                seed=config.seed,
            )
        )

        trainer = build_trainer(
            task.trainer,
            directory,
            mlflow_logger,
            build_callbacks(task.trainer, directory, has_validation=validation is not None),
        )
        trainer.fit(module, train_loader, val_loader)

        predict_loader = make_loader(
            WindowDataset(store, index, fold.test, payload_dtype=payload_dtype),
            batch_size=task.batch_size,
            num_workers=task.num_workers,
            seed=config.seed,
        )
        raw_batches = trainer.predict(module, predict_loader)
        if raw_batches is None:
            raise ValueError(f"{fold.name}: the predict loop returned no result")
        if raw_batches and isinstance(raw_batches[0], list):
            raise ValueError(f"{fold.name}: expected one prediction loader, got nested results")
        batches = cast("Sequence[PredictionBatch]", raw_batches)
        result = _assemble(batches, fold, window_labels)

        # Guard 1, carried over from the deleted `cross_validate`: predictions that do
        # not line up with the fold they came from. `_assemble` above already checks the
        # stronger property -- that the predicted rows are *exactly* the fold's test
        # positions -- but this is the guard the fold loop's docstring named, and its
        # message is worth keeping verbatim: a silent off-by-one in a runner would
        # otherwise score row i's prediction against row j's label and produce a number
        # that looks merely disappointing rather than wrong.
        if len(result.y_true) != fold.test.size:
            raise EvalError(
                f"{fold.name}: the runner returned {len(result.y_true)} predictions for "
                f"{fold.test.size} test rows; the runner and the fold disagree about what "
                "was held out"
            )

        # Guard 4, also carried over from the deleted `cross_validate`: a fold that
        # cannot be scored is a split problem before it is a metric problem, and the
        # message says so rather than surfacing whatever bare exception the metric
        # implementation happened to raise.
        try:
            values = compute(task.metrics, result.y_true, result.y_pred, result.y_score)
        except MetricError as error:
            raise EvalError(
                f"{fold.name}: {error}. A fold that cannot be scored is a split problem "
                "before it is a metric problem — check stratification."
            ) from error

        store_digest = store.identity
        _write_predictions(
            run.artifacts_dir,
            fold=fold,
            result=result,
            split=task.split,
            split_digest=store_digest,
            window_digest=task.window.digest,
            config_identity=_fold_invariant_config_hash(config),
        )
        (run.artifacts_dir / "windows.json").write_text(
            json.dumps(
                {
                    "store": store.path.name,
                    "store_sha256": store_digest,
                    "window_digest": task.window.digest,
                    "windows": len(index),
                    "split": task.split,
                    "fold": task.fold,
                },
                indent=2,
                sort_keys=True,
            )
        )

        # The held-out metrics above are computed after `trainer.predict`, from a plain
        # numpy comparison -- not from inside a `*_step` hook, so nothing during
        # training's own `self.log(...)` calls captures them. Logged explicitly, through
        # the same logger the Trainer streamed epoch metrics to, so the held-out numbers
        # land in the same MLflow run as everything else this fold produced.
        mlflow_logger.log_metrics(finite_metrics(dict(values)))
        log_run_artifacts(run, mlflow_logger)
    return dict(values)


def _write_predictions(
    directory: Path,
    *,
    fold: Fold,
    result: FoldPrediction,
    split: str,
    split_digest: str,
    window_digest: str,
    config_identity: str,
) -> None:
    """Write this run's held-out predictions: the per-run artifact a pooling reader pools N of.

    Under fold-as-process (decision 6) one run trains one fold, so what used to be one
    fold's slice appended into a shared ``OutOfFold`` across an in-process loop is now the
    whole file. ``split`` and ``split_digest`` -- the split family's name and the store
    digest it is bound to, the same digest ``windows.json`` records -- let a pooling
    reader (`dsio.eval.pool.pool_folds`) refuse to combine runs from different families or
    different store snapshots, the same binding ``SplitFile.store_manifest_sha256``
    already checks at load time, carried forward here because pooling happens in a
    different process than the one that validated it.

    ``window_digest`` and ``config_identity`` guard the invariant `cross_validate` used to
    get for free: a single closure and a single `Examples` meant every fold's ``y_true`` /
    ``y_pred`` / ``row_id`` structurally came from one model configuration and one
    coordinate system. Fold-as-process broke that structural guarantee into N separate
    processes, so it has to be checked instead -- ``window_digest`` (`task.window.digest`)
    catches folds built from different `WindowSpec`s, where `row_id` does not mean the same
    row from one file to the next; ``config_identity`` catches folds trained under a
    different backbone, hyperparameter or component, even when the window spec happens to
    match. Both are per-fold facts recorded here rather than derived at pooling time,
    because pooling happens in a different process than the one that trained the fold.

    ``y_score`` is omitted from the file entirely when the fold produced none, rather than
    written as zeros, so a pooling reader can tell "this fold scored nothing" from "this
    fold's scores happened to be zero".
    """
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "row_id": fold.test,
        "fold": np.asarray(fold.index, dtype=np.int64),
        "y_true": np.asarray(result.y_true),
        "y_pred": np.asarray(result.y_pred),
        "split": np.asarray(split),
        "split_digest": np.asarray(split_digest),
        "window_digest": np.asarray(window_digest),
        "config_identity": np.asarray(config_identity),
    }
    if result.y_score is not None:
        arrays["y_score"] = np.asarray(result.y_score)
    np.savez_compressed(directory / PREDICTIONS_FILE, **arrays)


def _assemble(
    batches: Sequence[PredictionBatch], fold: Fold, window_labels: np.ndarray
) -> FoldPrediction:
    """Reassemble predicted batches into fold order, keyed by the row each carried.

    The predictions come back in loader order, which happens to match fold order today.
    Sorting by the reported row makes that a checked fact rather than an assumption that
    quietly stops holding the moment anyone adds a sampler.
    """
    import torch

    if not batches:
        raise ValueError(f"{fold.name}: the predict loop produced no batches")
    rows = torch.cat([torch.as_tensor(batch["row"]) for batch in batches]).numpy()
    logits = torch.cat([batch["prediction"] for batch in batches]).float()

    order = np.argsort(rows, kind="mergesort")
    rows, logits = rows[order], logits[order]
    if not np.array_equal(rows, np.sort(fold.test)):
        raise ValueError(
            f"{fold.name}: predictions cover {rows.size} rows that do not match the "
            f"{fold.test.size} held out; the loader and the fold disagree"
        )

    truth = window_labels[rows]
    if logits.ndim == 2 and logits.shape[1] >= 2:
        probabilities = torch.softmax(logits, dim=-1)
        prediction = probabilities.argmax(dim=-1).numpy()
        score = probabilities[:, 1].numpy() if logits.shape[1] == 2 else probabilities.numpy()
    else:
        score = torch.sigmoid(logits.squeeze(-1)).numpy()
        prediction = (score > 0.5).astype(np.int64)

    return FoldPrediction(
        y_true=(truth > 0.5).astype(np.int64) if truth.dtype.kind == "f" else truth,
        y_pred=prediction,
        y_score=score,
    )
