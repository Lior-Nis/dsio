# Story 6.2: Make replay identity deterministic in CI

Status: review

## Story

As a maintainer about to migrate every consumer,
I want the replay-identity flake in the aggregate CMI flow diagnosed and eliminated,
so that migration parity rests on trustworthy replays.

## Acceptance Criteria

1. Given a replay-identity mismatch in any replay test, when the test fails, then the failure message includes a field-level diff of both provenance records. The diff covers git commit, dirty patch digest, lockfile, DSio version, environment and component identities.
2. Given the hypothesis that the first export dirties a clean checkout, when export writes are traced in a clean worktree, then it is confirmed or refuted, with evidence recorded here. If confirmed, no DSio export, evaluation or inference path writes inside the consumer's git worktree, and a test proves `git status --porcelain` is unchanged after a full flow.
3. "Resolved" means either the root cause is fixed with a regression test, or the aggregate CMI replay test has passed 20 consecutive times in a clean CI-equivalent worktree with provenance-diff capture in place.

## Tasks / Subtasks

- [x] Task 1: Field-level replay diagnostics (AC 1)
  - [x] 1.1 `tests/replay.py::assert_same_identities`: on mismatch, reports each differing stage. For each stage it downloads both runs' `provenance.json` and diffs every dotted field. When `execution.git.*` moved, it lists the files named in each `git.patch`. When `configuration.checkpoint_digest` moved, it loads both training checkpoints and reports tensor drift (max abs diff) or non-tensor metadata differences.
  - [x] 1.2 `tests/test_replay_assertions.py`: identical pass, a changed config field, a dirtied checkout naming the stray file, missing run IDs, and checkpoint weight drift.
  - [x] 1.3 All 11 replay tests (`tests/kaggle_portfolio/*`, `tests/reference_flows/*`) use the helper instead of a bare `==`.
  - [x] 1.4 `tests/conftest.py` puts the repo root on `sys.path`, so every suite can import `tests.*` helpers.
- [x] Task 2: Test the dirty-checkout hypothesis (AC 2)
  - [x] 2.1 Evidence (below) refutes it. A guard in the CMI replay test still asserts that `git status --porcelain` is unchanged across both flows.
- [x] Task 3: Resolution (AC 3), via the "20 consecutive green runs with provenance-diff capture in place" branch; the root cause is unconfirmed.
  - [x] 3.1 20/20 consecutive local CMI replay passes, with the working tree untouched during the loop. An earlier loop logged 1 failure in 20 without output; files were being edited during that loop, which the new checkout guard reports as a failure. It is recorded here as unverified.
  - [x] 3.2 20/20 consecutive CMI replay passes on a GitHub-hosted `ubuntu-latest` runner. This used the temporary step "Replay stress (temporary, Story 6.2)" in PR #79, CI run 36779671321, 2026-09-30, and 20 `1 passed` lines in the job log. The full CI suite passed in the same run. The temporary commit was dropped before merge.

## Dev Notes

### Diagnosis evidence (2026-10-01)

- **The failure.** `main` CI run 36450485533, attempt 1 (commit 59b7db6), `tests/kaggle_portfolio/test_child_mind.py:392`. The `train` identity matched across the two runs; `export`, `evaluation` and `inference` all differed. Attempt 2 and the PR run for the same commit passed.
- **Dirty-checkout hypothesis: refuted.** Execution identity captures `git status --porcelain` plus a patch that includes untracked, non-ignored files (`tracking/execution/git.py:34-74`). Every node records it, `train` included. Equal train identities therefore prove the checkout was identical when both trainings started. A local full CMI replay left `git status --porcelain` unchanged.
- **What moves at export.** Export identity additionally hashes `checkpoint_digest` (`reference_projects/kaggle/child_mind/tasks/downstream.py:56-72`), the SHA-256 of the full Lightning checkpoint bytes. Train identity is computed **before** training, so equal train identities say nothing about equal weights.
- **Ruled out:**
  - ModelCheckpoint filename versioning: CMI uses `checkpoint=False`.
  - Model-initialization RNG: `seed_everything(seed)` precedes construction.
  - Loader workers: `num_workers=0`.
  - Lightning's thread-count setter: DDP launchers only.
  - Nondeterministic-op warnings: none in the CI log, even with `deterministic="warn"`.
- **Plausible mechanism.** CPU float reductions depend on the intra-op thread count (verified locally: `x.sum()` differs in the last bits at 4 threads). MKL BLAS is the build's BLAS (`BLAS_INFO=mkl`). No alignment dependence reproduced locally on AVX-512, with or without `MKL_CBWR`. The CI runner's CPU differs, and the root cause is unconfirmed.
- **Consequence.** Any recurrence now prints the moved provenance field and, for a checkpoint digest, which tensors drifted and by how much. That turns the next CI failure into a root-cause report instead of two hashes.

## Dev Agent Record

### Agent Model Used

Claude Opus 5.5 (claude-opus-5-5)

### Completion Notes List

- AC 1: `tests/replay.py` plus 5 unit tests. All 11 replay tests use it.
- AC 2: the dirty-checkout hypothesis is refuted by identity structure and a local trace. A guard in the CMI replay asserts the checkout is unchanged.
- AC 3: 20/20 locally and 20/20 on a GitHub runner, with capture in place. If the flake recurs, the failure now names the moved provenance field and, for a moved checkpoint digest, the drifting tensors.

### File List

- `tests/replay.py` (new)
- `tests/test_replay_assertions.py` (new)
- `tests/conftest.py`
- `tests/kaggle_portfolio/test_{bike_sharing,child_mind,child_mind_sequence,digit_recognizer,essay_scoring,parkinsons_fog,rogii,store_sales,titanic}.py`
- `tests/reference_flows/test_{supervised,self_supervised}_flow.py`
