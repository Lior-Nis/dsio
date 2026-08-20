# 17. The process boundary is the fold boundary

Status: accepted (2026-08-20)
Supersedes: ADR 0008 ("The fold loop owns comparison") in its mechanism, and ADR 0012 ("The
ledger is the resume state") entirely.
Implemented: no. This is Plan 3.

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
running the entry point N times, from a shell loop or an agent. `RunConfig` gains two fields —
`split` (which committed file) and `fold` (which index) — and that is the entire interface
between the loop and the run.

Three properties follow for free, none of which needed code:

- **Resume.** A fold that dies is rerun by name. This is what `matrix/` was for.
- **Parallelism.** Four folds across four GPUs is four invocations, not a scheduler.
- **Grouping.** One MLflow experiment, N runs, tagged with the split name.

## Consequences

The guarantees ADR 0008 bought do not disappear; they move, and two of them get stronger.

**Cross-fold disjointness** moves from run time to load time. One split file holds all folds
as an ordered list, so `SplitFile`'s validator checks it when the file is read — strictly
earlier than a check inside a loop, and it no longer costs a walk over every test row before
the first model is fitted.

**The paired noise floor** still fires. Each run records the split digest plus its fold index,
and comparison checks correspondence across the two sets. This additionally catches a case the
in-process loop could not see: comparing fold 2 of split A against fold 2 of split B.

**Pooled out-of-fold metrics** become a function over N prediction files instead of an
accumulator. Pooling is still the better estimator than averaging per-fold scores, and it is
still available — it just reads artifacts rather than holding state.

The cost is process startup per fold: a fresh interpreter, a torch import, CUDA
initialisation. Tens of seconds against training runs measured in hours, which is why this
trade is affordable here and would not be for a model that trains in two seconds.

The subtler cost is that nothing now enforces that N folds were all run. A shell loop that
silently skips fold 3 produces a pooled metric over four folds that looks entirely normal.
The committed split file is the defence — it names how many folds exist, so a comparison can
refuse when the evidence is incomplete.
