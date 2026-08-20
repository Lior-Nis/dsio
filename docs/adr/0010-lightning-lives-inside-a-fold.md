# 10. Lightning's loop lives inside a fold, and the component chain has enforced invariants

Status: accepted (2026-08-18)
Implements: the plan's Phase 4 (torch runner), and ADR 0004 items 9 and 10.
Superseded in part by ADR 0017 (accepted 2026-08-20, implemented in Plan 3): Lightning's
loop no longer sits inside a fold loop, because there is no fold loop — one run is one fold.
The component chain and its enforced invariants survive; ADR 0015 makes them the only
training path rather than one of three.

## Context

ADR 0008 built the fold loop and asserted it would drive "an sklearn pipeline, a Lightning
trainer and a Nixtla rolling origin without adapters". Only one of those was built, so that
was a claim rather than a result. Lightning is the hard case: it owns a training loop, and
so does cross-validation.

## Decision

**They are not the same loop and they do not compete.** A `Fold` is one call to
`fit_predict`; a `Trainer` is constructed, fitted and discarded inside that call. The outer
loop never learns what a Trainer is, and the runner never learns how out-of-fold
predictions are accumulated, pooled or scored.

The result is that `torch_task.py` is mostly configuration. `FitPredict` needed no change
to accommodate Lightning, which is the answer to the question ADR 0008 left open.

**The component chain is ported from FORGE's `pipeline/base.py`**, the best pattern in that
repo: fixed slots with declared always-present / maybe-present invariants, so every training
paradigm is one object with different pieces in it.

```
x -> preprocessor? -> augmentor? -> transform -> spectral_augmentor? -> backbone -> head
```

`transform` defaults to identity rather than being optional, so the chain has one shape and
`forward` needs no branch. One registry per slot, not one registry of models: FORGE's
documented change-amplification cost was that adding a pretext task meant touching seven
places, and independent slots make a new backbone one decorator.

**`encode` is separate from `forward`.** A head is a task's opinion about features; the
features outlive it. This is what SSL pretraining hands to a probe and what an
embedding-cache stage will store, so the seam exists before Phase 5 needs it.

## Three changes from the original, each fixing something that cost real time

**Augmentation is training-only, enforced rather than documented.** FORGE's chain applies
whatever is configured whenever it is called. Augmenting during validation makes the metric
noisier *and* irreproducible while looking entirely normal — nothing in the config or the
loss curve reveals it. The two stochastic slots are skipped unless `self.training`, and both
halves are pinned: eval mode is deterministic, train mode is not.

**Callbacks are never constructed inside a bare `except`.** FORGE's `instantiate_callbacks`
catches everything and logs a warning, so a misconfigured `ModelCheckpoint` silently
disables checkpointing for a multi-hour run and the loss of the weights is found days later.
A misconfigured callback is a configuration bug and must stop the run.

**Metric names in filename templates are sanitised.** `ap{metrics/val_ap:.3f}` produced six
stray `checkpoints/checkpoint-epoch=NN-metrics/` *directories*, because the `/` inside the
format field became a path separator. Lightning offers no escaping, so `sanitise_metric`
does it.

## Smaller decisions that are load-bearing

**A fresh module per fold, never a reset one.** `load_state_dict` back to an initial
snapshot looks equivalent and is not: optimiser state, BatchNorm running statistics and any
lazily-built buffer survive it, so fold 2 would start from fold 1's normalisation.

**Predictions carry their row positions.** `predict_step` returns `row` alongside the
logits, and `_assemble` sorts by it and checks the set against the fold's test array.
Returning bare logits would make alignment a property of DataLoader ordering — true today,
silently untrue the moment anyone adds a sampler.

**Labels live outside the store.** The store is a canonical record of what was *measured*; a
label is an interpretation, and interpretations get revised. A relabelled cohort must not
force a re-ingest of the signal, and two labelling schemes over one corpus must not mean two
copies of the bytes. Hence the `LABELS` registry of per-row providers.

**Shape arguments are filtered by signature; user params are not.** `Conv1dEncoder` pools
over time and is genuinely length-agnostic, so making it declare a `length` it ignores would
be a lie that later reads as a constraint. User params stay strict and unfiltered, so a typo
in a config still fails loudly rather than vanishing into a `**kwargs` catch-all. Components
are registered as classes rather than through wrapper functions precisely so the signature
is real.

**Committed splits are required, not optional.** The runner resolves `fold_paths` in
pre-flight and fails with "commit a split file there first" rather than generating
something unrecorded on the fly. Splits are provenance (ADR 0006); a torch run that
invents its own has no provenance to cite.

## What building this surfaced

**A split bug that only a real model could reveal.** `generate` gave validation a whole
rotation bucket, making train `(k-2)/k`. At k=3 that is a 33% training set against 67% of
test-plus-validation — smaller than what it is evaluated on, which is not what anyone means
by "3-fold cross-validation". Every structural test passed: the parts were disjoint, total,
correctly stratified and correctly resolved. The only symptom was a model that would not
learn, and that reads as a modelling problem rather than a splitting one.

Validation is now carved out of the training portion by `val_fraction`, taken round-robin
across the non-test buckets so stratification survives into it. `val_fraction=0` drops the
part entirely. `test_training_is_the_largest_part_even_at_k_equals_three` pins it for
k = 3, 4, 5.

This is the second time in two phases that building the consumer found a defect the
producer's own tests could not: ADR 0008 found the empty-test-span bug the same way.

## Consequences

CPU torch and lightning join the dev group while remaining optional extras, with an explicit
`pytorch-cpu` index so CI does not pull CUDA wheels. uv sources are not propagated to
consumers, so a generated project chooses its own. The reasoning matches fsspec in ADR 0009:
a runner that CI cannot execute is a runner whose correctness is a claim.

One test asserts on a *number* rather than a structure — `roc_auc > 0.9` on a separable
tone. It exists because every other test would pass on a chain that detached the gradient,
shuffled labels against windows, or normalised the signal away: the plumbing would look
perfect and the model would learn nothing.

The suite is now 332 tests and about 26 seconds wall clock.
