# Plan 3a — The process boundary is the fold boundary

Implements locked decision 6 of `docs/superpowers/specs/2026-08-20-dsio-lean-design.md`.
Deliberately excludes decisions 7 and 8 (MLflow, the compose stack): this plan needs no
running infrastructure, so it can be verified by the existing suite. Plan 3b introduces
Docker.

## What is true today

`dsio run` trains one config across **all** folds in a single process. `cross_validate`
(`eval/loop.py:43`, 176 lines) drives `for fold: fit -> predict -> accumulate -> score`,
called from `torch_task.py:465`. It returns a `CVReport` (`eval/contract.py:150`) plus an
`OutOfFold`. `verdict.compare` (`verdict.py:156`) takes two `CVReport`s and pairs their
per-fold metric series via `fold_fingerprint` to get a noise floor. `SplitFile`
(`splits/models.py`) holds **one** fold per file, with `fold: int | None` naming which.

## What decision 6 requires

`dsio run` trains **one config against one fold** and is linear top to bottom. `RunConfig`
gains `split` and `fold`; the spec calls that "the entire interface between the loop and the
run". Cross-validation becomes N invocations from a shell loop or an agent, which buys
resume, parallelism and native MLflow grouping.

## What must not be lost

This is the substance of the plan. `cross_validate` is not only a loop — its docstring names
**three things it refuses to let pass**, and deleting the function deletes the guards unless
they are rehomed deliberately:

1. **Predictions that do not line up with the fold they came from.** A silent off-by-one in
   a runner scores row *i*'s prediction against row *j*'s label and produces a number that
   looks disappointing rather than wrong. Under one-fold-per-process this becomes a
   *per-run* check at prediction-write time.
2. **A row predicted by two folds.** Either the folds overlap or a runner returned the wrong
   positions; both make a pooled metric double-count. This becomes a *pooling-time* check in
   the new reader.
3. **Scores present for some folds and absent for others.** Pooling those computes a ranking
   metric over a mixture of probabilities and hard labels. Also a pooling-time check.

Two further guarantees the spec names explicitly:

4. **Cross-fold test disjointness** moves from run time to **load** time — one split file
   holds all folds as an ordered list, so `SplitFile`'s validator catches it strictly
   earlier than today.
5. **The paired noise floor still fires.** Today it pairs per-fold series inside one
   `CVReport`. It must now pair across **two sets of N single-fold runs**, matching on split
   digest plus fold index. The spec notes this additionally catches "fold 2 of split A
   compared against fold 2 of split B" — a case the current fingerprint cannot see.

**A task that deletes code without rehoming its guard has not finished.** Every deletion
below names where its invariant lands.

## Tasks

### Task 1 — `SplitFile` holds every fold, and validates disjointness at load
Replace `fold: int | None` with an ordered list of folds. The validator gains the cross-fold
test-disjointness check (guarantee 4). Keep the store-manifest binding and the existing
within-part checks.
**Verify:** hand-write a split file whose fold 1 and fold 3 test parts share a group; loading
it must raise `SplitError` naming both folds. Break the validator, watch the test fail,
restore, confirm `git diff` clean.

### Task 2 — `RunConfig` gains `split` and `fold`
Two fields, plus validation that `fold` indexes a fold the named split actually has.
**Verify:** a config naming fold 7 of a 5-fold split fails at config-build time, not at
train time. Round-trip through YAML still reconstructs an equal object (spec Verification 7).

### Task 3 — `dsio run` trains one fold; delete the in-process loop
Remove the `cross_validate` call from `torch_task.py:465` and make the runner linear: build
config -> build data -> build module -> `fit` -> `predict` -> write artifacts -> stamp
provenance. Guard 1 lands here as a check that written predictions correspond to the fold's
own row positions.
**Verify:** a runner returning positions from the wrong fold must fail the run, not produce a
quiet number. Prove the guard bites.

### Task 4 — Pooled out-of-fold metrics as a reader
A ~30-line function reading N `predictions.npz` files and pooling them. Guards 2 and 3 live
here: reject a row predicted twice, and reject a mixture of scored and unscored folds.
**Verify:** one test per guard, each proven by breaking the guard.

### Task 5 — `verdict.compare` pairs across runs, not within a report
Rework `compare` to take two sets of single-fold runs, matching on split digest plus fold
index (guarantee 5). Retire `fold_fingerprint` in favour of that correspondence.
**Verify:** comparing fold 2 of split A against fold 2 of split B must refuse, which the
current fingerprint cannot catch. Comparing matched sets must still produce the paired floor.

### Task 6 — Delete what is now unreachable
`cross_validate`, `CVReport`, `OutOfFold`, `fold_fingerprint`, `write_report`/`read_report`
as they fall out of use. Delete only what nothing imports; anything still referenced means an
earlier task did not finish.
**Verify:** the full check, plus `git grep` for each deleted name across `src/` and `tests/`.

### Task 7 — Docs
ADR for decision 6. Update ADR 0008 ("the fold loop owns comparison"), which this plan
contradicts, and ADR 0010 ("Lightning lives inside a fold"). Note that ADR 0015's replacement
table was corrected in Plan 2b for claiming `cross_validate` was already gone — it is this
plan that makes that true.

### Task 8 — Fresh-clone proof
Clone the branch, install, run the full check, and demonstrate fold-as-process: a shell loop
over 5 folds produces 5 runs, and rerunning fold 2 alone reproduces its metrics exactly
(spec Verification 4). Report the code-only line count via `scripts/count_code.py` against
the 4,672 baseline (the merged figure the spec records; an earlier draft of this plan said
4,640, which was measured mid-fix-wave at 73e8da2 rather than at merged main).

## Global constraints

Python >=3.12; torch/lightning only in the `cpu`/`gpu` extras. Every uv command needs
`--extra cpu`.
**VERIFY:** `uv run --extra cpu pytest && uv run --extra cpu ruff check . && uv run --extra cpu mypy && uv run --extra cpu lint-imports`
ruff line-length 100, rules `E,F,W,I,UP,B,BLE,SIM,RUF`; no bare `except`, a typed re-raise is
fine. mypy `disallow_untyped_defs`. Baseline at plan start: **410 tests**, **3 contracts**,
**4,672 code lines**.

**This repository has produced nineteen checks that could not fail.** Every guard added or
touched must be verified by breaking what it guards, observing the failure, restoring, and
confirming `git diff` is clean. Deleting a guarded behaviour and keeping its test green is
the failure mode this plan is most exposed to, because it is mostly deletion.
