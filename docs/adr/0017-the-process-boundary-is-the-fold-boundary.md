# 17. The process boundary is the fold boundary

Status: accepted (2026-08-20)
Supersedes: ADR 0008 ("The fold loop owns comparison") in its mechanism, and ADR 0012 ("The
ledger is the resume state") entirely.
Implemented: yes, by Plan 3a. `SplitFile` (`splits/models.py`) holds every fold of a family
as an ordered list and validates cross-fold test disjointness in `_validate_folds` at
`SplitFile.load` time. `TorchTask.fold` (`train/torch_task.py`) is a required field, no
default. `run_torch` is a linear single-fold runner: build config, resolve the named fold,
fit, predict, write `predictions.npz`, stamp provenance — no in-process loop, no callback.
`eval/pool.py::pool_folds` reads N single-fold `predictions.npz` files back into one
`Pooled` set. `verdict.compare` (`eval/verdict.py`) pairs two `Pooled` sets on split family,
store digest and fold-index set, using `_correspondence`.

## Context

ADR 0008 made the fold loop framework code: `for fold: fit -> predict -> accumulate -> score`,
written once so that "the score" means the same thing in every script. The reasoning holds —
hand-written per script, that loop acquires a slightly different definition of the score each
time, averaged per fold here and pooled there.

But the loop was framework code *because* it had to drive scikit-learn, Nixtla and Lightning
through one `FitPredict` callable. ADR 0015 removes that requirement, and what is left is
inversion of control for its own sake: `train/torch_task.py` hands a closure to a loop that
calls back to construct a `Trainer` per fold. The run script becomes something you read
inside-out, and a breakpoint lands in a callback three frames from anything meaningful.

There is a second cost. An in-process loop over N folds is one process that either finishes
all N or loses all N. ADR 0012 answered that with a resumable job matrix keyed on config hash
— 700 lines whose entire purpose was to recover something the process boundary would have
given for free.

## Decision

`dsio run` trains **one config against one fold**, and is linear top to bottom: build config →
build data → build module → `Trainer.fit` → predict → write artifacts → stamp provenance.

There is no `cross_validate`, no `CVReport`, no in-process fold loop. Cross-validation is
running the entry point N times, from a shell loop or an agent. The interface between the
loop and the run is `TorchTask.fold` (`train/torch_task.py`) — `split` (which committed
file) and `fold` (which index) live on the task config, not on `RunConfig` itself, and are
reached from outside exactly like any other config field: `dsio run <preset>
task.fold=2` through the CLI's existing `nested.path=value` overrides. No new interface
was added; the existing override grammar already reached this field.

Three properties follow for free, none of which needed code:

- **Resume.** A fold that dies is rerun by name. This is what `matrix/` was for.
- **Parallelism.** Four folds across four GPUs is four invocations, not a scheduler.
- **Grouping.** One MLflow experiment, N runs, tagged with the split name.

## Consequences

The guarantees ADR 0008 bought do not disappear; they move, and two of them get stronger.
`cross_validate`'s docstring actually named **four** invariants, not the three the plan
first catalogued when it set out to delete the function — the fourth (a fold that cannot be
scored is a split problem) was easy to miss because it reads as a metrics concern rather
than a fold-loop one:

1. **Predictions that do not line up with the fold they came from.** Now checked in
   `run_torch` itself (`train/torch_task.py`), immediately after `_assemble` reassembles
   predicted batches into fold order — a per-run check at prediction-write time, exactly as
   the plan called for.
2. **A row predicted by two folds.** Now checked in `pool_folds` (`eval/pool.py`), at
   pooling time, over the `row_id` each fold's file recorded.
3. **Scores present for some folds and absent for others.** Also checked in `pool_folds`,
   by comparing how many of the N files carry a `y_score` key.
4. **A fold that cannot be scored.** Checked in `run_torch`, immediately after guard 1,
   by catching the metrics layer's `MetricError` and re-raising it as a split problem.

Guards 1 and 4 landed in the runner because they are properties of one fold's own
predictions and can be checked the moment that fold finishes; guards 2 and 3 landed in
`pool_folds` because they are properties that only exist once more than one fold's output
is in hand. None of the four were dropped — the plan's warning that "a task that deletes
code without rehoming its guard has not finished" is the reason to state where each one
went rather than just that `cross_validate` is gone.

**Cross-fold disjointness** moves from run time to load time. One split file holds all folds
as an ordered list, so `SplitFile._validate_folds` checks it via
`_assert_test_parts_disjoint_across_folds` (`splits/models.py`) when the file is read —
strictly earlier than a check inside a loop, and it no longer costs a walk over every test
row before the first model is fitted.

**The paired noise floor** still fires. Each run's `predictions.npz` records the split
family, the store digest and its fold index; `verdict.compare`'s `_correspondence` helper
checks all three match before pairing. This additionally catches a case the in-process loop
could not see: two runs that both name fold 2 of a split family called `"fam"`, but were
read back from different store snapshots, are refused on the digest mismatch even though
the fold index and split name alone would look identical. That claim used to be asserted in
this ADR's prose alone; it is now
`test_comparing_fold_2_of_one_split_against_fold_2_of_another_is_refused`
(`tests/eval/test_verdict.py`), and disabling the digest check in `_correspondence` fails
it.

**Pooled out-of-fold metrics** become a function over N prediction files instead of an
accumulator. Pooling is still the better estimator than averaging per-fold scores, and it is
still available — it just reads artifacts rather than holding state.

The cost is process startup per fold: a fresh interpreter, a torch import, CUDA
initialisation. Tens of seconds against training runs measured in hours, which is why this
trade is affordable here and would not be for a model that trains in two seconds.

The subtler cost is that nothing now enforces that N folds were all run. A shell loop that
silently skips fold 3 produces a pooled metric over four folds that looks entirely normal.
`pool_folds` (`eval/pool.py`) can refuse that — it takes an opt-in `expected_folds` and
raises if any of those indices never showed up among the files it was handed — but nothing
makes a caller pass it. `pool_folds` never opens a split file itself; it only ever sees the
paths it is given, so knowing "this experiment should have had folds 0-3" has to come from
somewhere else, supplied explicitly: a caller that has already loaded the `SplitFile` can
pass `[f.index for f in split_file.folds]` and turn a shell loop's silently skipped fold
into a refusal. Until a caller does that, the incomplete-N-folds case is still silent.
