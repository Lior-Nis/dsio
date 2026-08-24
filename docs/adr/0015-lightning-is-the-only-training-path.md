# 15. Lightning is the only first-class training path

Status: accepted (2026-08-20)
Supersedes: the multi-modality scope of the 2026-08-15 design spec — its Phase 4 ("tabular,
forecast, torch/Lightning") and the agentic modality of ADR 0013.
Implemented: yes. Plan 1 deleted `agents/`, `matrix/`, `tracking/` and the neutrality
abstractions listed below. Plan 2 deleted `train/tabular.py`, the `forecast` extra, and
dissolved `ssl/` as a directory.

## Context

The spine reached 13,345 lines across 14 packages in four days. A large share of that was
not domain logic but *framework neutrality*: machinery whose only job was to let one code
path drive scikit-learn, Nixtla and Lightning uniformly.

That neutrality was never free, and it was not load-bearing either. `train/torch_task.py`
imported eight dsio subsystems solely to plug a `LightningModule` into a modality-neutral
fold loop. `eval/metrics.py` reimplemented fourteen metrics in numpy because scikit-learn was
an optional extra. `runs/seeding.py` hand-rolled DataLoader worker seeding. And `tracking/`
defined an `ExperimentTracker` protocol with `MultiTracker` and `NullTracker` implementations
— **which nothing in `src/` ever imported**. It was built, and never wired in.

Every one of those is something Lightning already ships. The spine was paying, once per
subsystem, to rebuild a contract it already had.

The countervailing argument — that neutrality is what lets a tabular baseline and a deep
model be compared honestly — turns out to be answered elsewhere. The comparison layer works
on out-of-fold predictions in a fixed artifact shape, and that shape has nothing to say about
what produced the numbers.

## Decision

Torch and Lightning are the single first-class training path. No scikit-learn runner, no
Nixtla, no reinforcement learning, no agentic training.

Every abstraction that existed to be framework-neutral is deleted rather than maintained:

| Deleted | Replaced by |
|---|---|
| modality dispatch inside `eval.loop.FitPredict` (choosing an sklearn, Nixtla or Lightning fit_predict by config) | nothing to dispatch any more — at this decision, `cross_validate` (`eval/loop.py`) was still called directly, always with a Lightning `fit_predict` closure (`train/torch_task.py`); Plan 3a (ADR 0017) later removed `cross_validate` itself, once there was only ever one closure to call |
| `eval/metrics.py` implementations | evaluated, **rejected** — kept in numpy (see below) |
| `tracking/` (`ExperimentTracker`, `MultiTracker`, `MlflowTracker`) | Lightning `Logger` / `MLFlowLogger` |
| `agents/` | out of scope |

**`runs/seeding.py` was also evaluated for replacement by `lightning.seed_everything(seed,
workers=True)` and rejected**, not partially replaced: all 86 lines are still live, and
both entry points that seed a run (`dsio run` in `cli/run_cmd.py`, and `execute()` in
`train/runner.py`) call dsio's own `seed_everything`, never Lightning's. Its
`dataloader_kwargs`/`_seed_worker` pair is load-bearing — it is what makes a result
independent of DataLoader worker count, a property `lightning.seed_everything` alone does
not provide.

**`eval/metrics.py` was tried and reverted.** Two independent reasons, either sufficient on
its own:

1. Six of the eight classification metrics (`accuracy`, `balanced_accuracy`, `f1`,
   `f1_macro`, `precision`, `recall`) cannot reach the 1e-12 scikit-learn parity
   `tests/eval/test_metrics.py` pins them to through any documented, per-call means in
   torchmetrics 1.9.0: `torchmetrics/utilities/compute.py`'s `_safe_divide` casts
   confusion-matrix counts with an unconditional `num.float()`, always producing
   `float32` regardless of input dtype. The remaining two (`average_precision`,
   `roc_auc`) lose precision the same way through a different mechanism — a bare Python
   `1.0` in `_binary_clf_curve` that promotes through torch's process-wide default
   dtype rather than the input tensor's — and setting that default dtype for the
   duration of a call is a side effect a metrics module should not impose on whatever
   else is running in the process. Measured against this file's own test fixtures,
   `roc_auc` came out ~1.8e-08 off scikit-learn's float64 result — four orders of
   magnitude past the pin.
2. The rationale above ("a hand-rolled version gets wrong the first time it sees two
   GPUs") does not apply to this call path. At this decision, `METRICS`/`compute()` was
   reached only from `eval/loop.py`, pooling an `OutOfFold` of plain `np.ndarray`, and
   from `ssl/probe.py` — both single-process, numpy in and out, computed after training
   rather than during it. Plan 2b — the plan this decision belongs to — then dissolved the
   `ssl/` directory, moving `ssl/probe.py`'s callbacks into `train/callbacks.py`, and Plan
   3a (ADR 0017) removed `eval/loop.py` and `OutOfFold`; `compute()` is reached
   today from `eval/pool.py::pool_folds` (pooling a run's `predictions.npz` files),
   from `train/torch_task.py`, and from `train/callbacks.py`. The property this
   argument rests on is unchanged: every one of those call sites is still
   single-process, numpy in and out, computed after training rather than during it.
   Training-time metric logging is a separate path through Lightning's `self.log`
   (`nn/module.py`). These metrics never run as accumulated GPU tensors and never
   reduce across processes, so torchmetrics's distributed-reduction benefit is not
   available here to justify the precision cost above.

We do **not** adopt `LightningCLI`. It is YAML-config-driven, which ADR 0001 rejects for
reasons that have not changed: structure belongs in Python, and YAML is a recorded output of
a run rather than an authored input to one.

A tabular baseline remains reachable — as a plain, deliberately un-abstracted script sharing
the ledger and the committed split files but *none* of the model abstractions. Two runners
that share provenance and nothing else are less code than one runner pretending both are the
same thing, which is what the dispatch layer was.

## Consequences

Committing to one framework is what makes the skeleton lean. Lightning already provides the
mix-and-match contract the neutrality layer was rebuilding, so adopting it *deletes* code
instead of adding it.

The cost is real and worth naming: forecasting is out. A Nixtla model has no first-class home
any more, and the `forecast` extra goes with it. That extra was never implemented in the first
place — the template offered the modality and then generated a project whose only preset was a
scikit-learn baseline. That gap is the sharpest evidence that breadth had outrun depth, and it
is the reason this ADR chooses depth.

The second cost is that the decision is expensive to reverse. Re-admitting a second framework
means rebuilding a dispatch layer from scratch. That trade is deliberate: a skeleton that does
one thing well beats one that does three things through an adapter.
