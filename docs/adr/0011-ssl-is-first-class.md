# 11. SSL is first-class: pretraining is not a fold loop, and the probe is not a subprocess

Status: accepted (2026-08-18)
Implements: the plan's Phase 5.

## Context

The stated ratio for FORGE is roughly **100:1 unlabelled to labelled by hours** — 42.2M
unlabelled patches against 197k labelled. That is the normal situation for the ventures dsio
serves, and it is why SSL is a first-class concern here rather than an add-on.

FORGE proved the science and paid for the infrastructure. Seven near-duplicate pipeline
classes, three of which are the same MAE with a different masking mode. A downstream probe
that shelled out to `train_classification.py` via subprocess/SBATCH behind a
`ThreadPoolExecutor`, plus `probe_loop.sh` polling a checkpoint directory every fifteen
minutes with a `.probed_checkpoints` resume file. And classification checkpoints that
reloaded their MAE encoder from a hardcoded absolute path.

## Decisions

**Pretraining is not a fold loop.** There is no held-out score to pool: the pretext loss is
not the quantity being estimated, the encoder is the output, and its quality is measured
downstream. Forcing it through `cross_validate` would produce a cross-validated
masked-reconstruction MSE — a number nobody should act on. `ssl_pretrain` is one fit
producing one artifact, and `test_pretraining_writes_no_evaluation_report` pins that.

**One objective per family, not seven pipelines.** MAE (generative), SimCLR (contrastive),
VICReg (redundancy reduction). Each is a small object with one method,
`step(module, x) -> (loss, logs)`, over the unchanged component chain from ADR 0010. The
backbone does not know which objective is training it, which is the property that makes an
MAE-pretrained encoder and a VICReg-pretrained one interchangeable downstream.

**Masking is its own module.** It is the axis FORGE actually varied, so it is a slot rather
than a constructor argument: a fourth mode is a decorator, not a fourth pipeline. Every
strategy returns `True` where a position is **hidden**, asserted rather than documented —
the opposite convention is equally natural, and mixing them trains the model on exactly the
positions it was meant to predict, a leak whose only symptom is a suspiciously good
reconstruction loss.

**Every method logs the diagnostic that reveals its own failure mode**, because in SSL the
loss does not. A collapsed SimCLR encoder mapping everything to one point has a *low* loss
and useless features. So SimCLR logs mean off-diagonal cosine, VICReg logs embedding
standard deviation and its variance penalty, and MAE logs masked and visible error
separately — if visible error collapses while masked error does not, the model is copying.

**RankMe (Garrido et al. 2023) as a callback.** Effective rank from the entropy of the
singular values. Dimensional collapse is the characteristic SSL failure and the loss cannot
see it: an encoder using 3 of its 64 dimensions can have a healthy loss curve and features
no head can separate. It needs no labels, so it runs on the pretraining corpus itself.

**The online probe is a callback, not a scheduler.** A linear model on frozen features takes
a fraction of a second. Making it a callback deletes the subprocess, the SBATCH submission,
the thread pool, the fifteen-minute polling loop and the resume file — along with their
failure modes, which were invisible from the training run: a silently dead subprocess, a
checkpoint probed twice, a probe reporting against a checkpoint since overwritten.

Two details in the probe are load-bearing. It restores the module's training flag, because
leaving it in eval mode disables dropout and freezes BatchNorm for the rest of training —
degrading the very run it measures, and looking like the SSL method underperforming. And it
never backpropagates: a probe that did would be supervised finetuning wearing a probe's
name, with labels leaking into a representation that is supposed to be label-free.

## The encoder handoff

The bug FORGE shipped is now unrepresentable. An encoder is registered in the model
registry, whose `ModelRef` has no way to express "latest", and a downstream `TorchTask`
names it by `name`, `version` and `digest`. There is no path to hardcode. The registry
re-hashes on load and refuses a mismatch, so the remaining risk — a swapped artifact — fails
closed.

`freeze` distinguishes the two experiments people conflate: a *probe* measures what the
representation already contains, a *finetune* measures what it is a good starting point for.
Freezing also puts the component in eval mode, because a frozen BatchNorm whose running
statistics keep updating is not frozen, and the difference shows up as a probe that
mysteriously outperforms its own linear separability.

Loading into a mismatched architecture is refused rather than partially applied. Silently
loading a subset of weights produces a model that is part pretrained and part random, and
reports as though it were fully pretrained.

## Label budgets

Ported from FORGE's `_subset_n_patients`, which gets two things right:

**The budget is a number of groups, not of windows.** Sampling windows would draw from every
subject at every budget, so "10% of labels" would still mean every subject was labelled.
Real labelling effort is per-subject, and a model that has seen a little of everyone is in a
far easier position than one that has seen a few people completely.

**The draw is stratified over the positive rate.** With ~16% of DeFOG patients near zero
FOG, a small uniform draw is routinely positive-starved, and the resulting curve measures
the luck of the draw rather than the value of a label. Measured rather than asserted:
`test_stratified_selection_beats_uniform_on_rate_fidelity` runs 40 seeds and compares mean
absolute rate shift.

**Every arm gets the same groups at the same budget.** That is what makes probe / random /
scratch comparable; otherwise the gap between arms partly measures which subjects each got.

One change from the original: budgets are **nested** by default, so K=4's groups are a subset
of K=8's. Without nesting a dip in the curve is ambiguous — the budget, or a harder set of
subjects — and resolving that ambiguity costs a full re-run at every budget with several
seeds. Nesting is implemented as breadth-first bisection of the rate ranking, so every
prefix still spans the full range; `assert_nested` checks the property rather than trusting
it.

## What building this surfaced

The first implementation of the nested ordering was wrong in a way that looked right. It
selected indices correctly and then appended them in reverse index order — needed for safe
list popping — so short prefixes received the **highest**-rate groups exclusively, the exact
opposite of a stratified draw. Every structural test passed: the budgets were nested, the
counts were right, the seed mattered. Only the rate-fidelity and range-span assertions
caught it.

This is the same lesson as the previous two phases in a third form: the properties that
matter here are statistical, so the tests that catch real defects have to be statistical
too. `assert_nested` alone would have shipped it.

## Consequences

`dsio.ssl` depends on `dsio.nn`, which keeps torch confined to two packages plus the two
torch runners. `dsio.eval` remains a leaf.

What is deliberately not built: no momentum encoder (BYOL/MoCo), no JEPA-style latent
prediction, no multi-crop. Each is a new `SslMethod` and none is needed to prove the seam.
Embedding caching is also not wired to the stage cache yet — `SslModule.predict_step`
returns embeddings and `encode` is the stable seam, so the pieces exist, but the cached
artifact does not.
