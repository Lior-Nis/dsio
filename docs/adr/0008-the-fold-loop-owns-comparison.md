# 8. The fold loop is framework code, and comparison is enforced against a noise floor

Status: accepted (2026-08-18)
Implements: ADR 0004 items 8 and 11; the plan's Phase 3.

## Context

Phase 1 gave dsio a spine and Phase 2 gave it a data layer, but nothing joined them:
`train/` and `cli/` imported neither `dsio.data` nor `dsio.splits`. The store, the window
index, the split files and the stage cache were 164 tests of well-verified island.

Meanwhile the two surveyed repos both state the same contract in prose and then hand-write
it per script:

- `kaggler` re-implements `for fold: fit → predict → accumulate → score` in four scripts,
  and they do not agree on what "the score" means — one averages per fold, another pools.
- FORGE's evaluation predictions are cached by checking whether a column exists, and its
  paper numbers are recovered by globbing log filenames.

The consequence in both is the same: two runs of "the same" experiment are not comparable
without reading two scripts first.

## Decision

**One fold loop, in `dsio.eval`, driven by row positions.** A `Fold` is integer positions
into whatever is being folded. That is the one currency every modality shares — it indexes a
dataframe, a numpy array and a `WindowIndex` identically — so the loop needs no adapters and
no knowledge of the framework underneath. `cross_validate(folds, fit_predict, ...)` is the
whole interface.

**`dsio.eval` is a leaf, enforced by import-linter.** It may not import `data`, `splits`,
`train`, `runs`, `artifacts` or `cli`. The moment it can see a `WindowIndex` or a
`Run`, someone types one into a signature and the loop stops being usable for tabular,
forecast and torch alike. Dependency points the other way: `splits` builds folds, runners
call the loop.

**Metrics are implemented in numpy, not re-exported from sklearn.** sklearn is an optional
extra; making the artifact contract depend on it would leave the most important artifact in
the system untested in a bare install, and would drag it into a torch-only project. The
tests pin every metric against sklearn to 1e-12, including tie-heavy cases, so "we wrote our
own" cannot drift into "ours is subtly different". Average precision is summed step-wise —
`auc(recall, precision)` interpolates between operating points the model cannot achieve and
is optimistically biased.

**The canonical out-of-fold artifact is `oof.npz`, not parquet.** numpy is a hard dependency;
pyarrow is not. Predictions are kept rather than only metrics, because a metric is a lossy
summary chosen before you knew what you would need to ask, and predictions answer questions
you have not thought of yet — subgroup error analysis, calibration, threshold selection,
ensembling, the noise floor. It costs kilobytes.

**Pooled and per-fold metrics are both reported.** Pooled over the concatenated out-of-fold
predictions is the headline, because averaging per-fold AP across folds that each contain
three positives is close to meaningless. Per-fold spread is the uncertainty. With a single
fold the spread is reported as *absent* rather than 0.0: zero would read as "this result has
no uncertainty", which is the opposite of true and which the verdict machinery would then
act on.

**Coverage is recorded.** A purged temporal split legitimately predicts a fraction of rows.
Reading a pooled metric without knowing it covers 40% of the data is how a result gets
over-claimed.

**A verdict requires clearing a noise floor.** Ported from `kaggler`'s `self_model.verdict`:
an experiment is a win only if its improvement over its baseline exceeds the fold spread.

## What dsio adds to the ported verdict

kaggler estimates the floor from the candidate's own fold spread, which is the best
available when you cannot be sure two runs used the same folds. dsio commits its folds to
YAML and fingerprints the assignment, which buys two things kaggler could not have:

**The paired test.** When two runs provably held out the same rows, the floor is the
standard error of the per-fold *differences*. Fold-to-fold variation the two models share —
one fold simply being harder — cancels. In `test_paired_comparison_sees_what_the_unpaired_floor_buries`
a real and perfectly consistent 0.01 improvement is a WIN when paired and NEUTRAL when not,
on identical numbers. Pairing sharpens the test without lowering the bar: an improvement
that appears in two folds and reverses in the other two stays neutral either way.

**The refusal.** Two runs with different fold fingerprints are not compared at all.
`kaggler/cv.py` states the doctrine — *the fold assignment is the single source of truth;
comparing runs on different folds is meaningless* — and has no way to check it. The
fingerprint makes it enforceable, and a refusal is worth more than a confident, meaningless
delta. It can be waived explicitly, which then downgrades to the unpaired floor.

The fingerprint covers held-out positions only, so a baseline fitted on a subset is still
comparable to a candidate fitted on everything.

**A second, independent floor.** `sampling_noise(n_rows)` — kaggler's `lb_noise_sigma`,
generalised away from leaderboards — is the sampling sigma of a proportion metric on a
finite evaluation set. Consistency across folds says nothing about whether the evaluation
set was ever large enough to resolve the difference. `minimum_detectable_rows(delta)` asks
that question before the experiment rather than after: resolving 0.001 accuracy takes a
quarter of a million evaluation rows.

## What the loop refuses

Each of these is a named test, and each corresponds to a failure that produces a plausible
wrong number rather than a crash:

- predictions whose length disagrees with the fold's test set — an off-by-one in a runner
  would otherwise score row *i* against row *j*'s label, and the result looks merely
  disappointing rather than wrong, so nobody investigates;
- a row predicted by two folds — the pooled metric silently reweights toward the duplicates.
  Checked in `folds_from_splits` too, so it fails before anything is fitted;
- scores present for some folds and absent for others — pooling a mixture of probabilities
  and hard labels produces a ranking metric that means nothing;
- a fold that cannot be scored — a test fold with no positives has no average precision, and
  the error says so as a *split* problem, because that is what it is.

`Fold` validates its own train/test/val disjointness on construction. That is already
guaranteed for folds built from split files, but folds built from a sklearn splitter or by
hand in a notebook carry no such guarantee, and this is the last place to catch a leaked row
before it inflates a score.

## Consequences

The tabular runner shrank to loading a dataset, building folds and returning a
`FoldPrediction`; it owns no accumulation, scoring or artifact layout. It now
cross-validates by default instead of scoring a single holdout, and it refits on all rows
for the shipped model — the cross-validated score estimates how well the *procedure*
generalises, so the shipped model should use all the evidence.

Grouping is chosen from the data, not from config. A config flag to disable grouping is a
config flag to produce a wrong number; a stratified fold that splits a subject across train
and test is worse than an imbalanced one that does not.

## What this surfaced

Building the seam immediately found a bug that neither half could have found alone:
`walk_forward` accepted a `test_fraction` giving a test span narrower than one window.
`TimeSpan.contains` keeps only windows lying entirely inside a span, so every fold was
structurally valid and empty. The split, the files and the folds all looked fine, and the
failure surfaced much later as "nothing to score" — pointing at the evaluation rather than
at the window length that caused it. It now fails at generation with the actual cause.

This is the argument for building the seam before the second and third runners rather than
after.
