# 15. Lightning is the only first-class training path

Status: accepted (2026-08-20)
Supersedes: the multi-modality scope of the 2026-08-15 design spec — its Phase 4 ("tabular,
forecast, torch/Lightning") and the agentic modality of ADR 0013.
Implemented: partially. Plan 1 deleted `agents/`, `matrix/`, `tracking/` and the neutrality
abstractions listed below. `train/tabular.py`, the `forecast` extra and `ssl/` as a directory
go in Plan 2.

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
| `eval.loop.FitPredict` dispatch | `Trainer` |
| `eval/metrics.py` implementations | `torchmetrics` |
| `tracking/` (`ExperimentTracker`, `MultiTracker`, `MlflowTracker`) | Lightning `Logger` / `MLFlowLogger` |
| most of `runs/seeding.py` | `lightning.seed_everything(seed, workers=True)` |
| `agents/` | out of scope |

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
