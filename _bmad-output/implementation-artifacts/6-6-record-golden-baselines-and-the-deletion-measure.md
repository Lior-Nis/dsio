# Story 6.6: Record golden baselines and the deletion measure

Status: done

## Story

As a maintainer migrating consumers,
I want every consumer's pre-migration metrics and code measures recorded against a fixed commit,
so that each migration proves it preserved behavior and reduced consumer code.

## Acceptance Criteria

1. Given the baseline commit (the head of Epic 6 before this story's measurements), when each of the 11 consumers runs its contract tier, then its evaluation metrics are committed as golden values that its contract tests assert, and contract tests fail on any unexpected change.
2. Given each Kaggle consumer whose dataset is under `~/Datasets`, when it runs its representative tier on the live MLflow server with a run-isolated `PREFECT_HOME`, then its evaluation run URI, metrics, commit and hardware are recorded in `docs/component-warehouse/parity-baseline.md`.
3. Given the consumer tree, when the measurement tool runs, then it reports non-ingestion lines per consumer (every line outside `data.py`, `tasks/data.py` and ingestion-only modules such as `parkinsons_fog/scale/`), and the baseline is recorded. A syntax-tree check lists every `nn.Module`/`Dataset` subclass and every locally defined value passed as an objective, collate function, normalizer or validator. This is the violation baseline that Story 11.3 must reduce to the candidates register.

## Tasks / Subtasks

- [x] Task 1: Golden contract metrics (AC 1).
  - [x] 1.1 `tests/golden.py::assert_golden_metrics` compares against `tests/golden_metrics.json` with a relative tolerance of 1e-6. Contract flows replay exactly on one machine, but CPU float reductions differ in the last bits across CPU generations and CI runners, and a real change moves metrics by orders of magnitude more. `DSIO_UPDATE_GOLDEN=1` regenerates the goldens deliberately.
  - [x] 1.2 All 11 replay tests assert their goldens: 12 keys, with CMI split by mode.
  - [x] 1.3 Mutation check: `titanic.accuracy` changed 0.5 → 0.51 fails with the moved metric named. A 2e-7 change sits inside tolerance by design.
- [x] Task 2: Representative tier (AC 2).
  - [x] 2.1 `tools/representative.py` runs a consumer's own flow on its real data. It targets the live MLflow server (default `http://localhost:5000`), uses an isolated `PREFECT_HOME` and a fresh workspace, and records metrics, run IDs, commit, dirty flag, hardware and duration. `--compare --tolerance` gives the parity stories 7.7, 8.7, 9.6 and 10.6 their check.
  - [x] 2.2 Recorded 8 of the 9 Kaggle consumers from a clean tree in `docs/component-warehouse/parity-baseline.{json,md}`.
    - **Commit:** `fe37300`, except Titanic at `38845c4`. The two differ only in the runner's report rendering, not in consumer code.
    - **Hardware:** CMI sequence ran on CUDA (RTX 5070 Ti, temporary `torch 2.13.0+cu130` wheel, restored with `uv sync --locked`). The others ran on CPU (AMD Ryzen 9 9900X, 12 threads).
    - **Bike Sharing:** contract tier only. `~/Datasets/dsio-kaggle-portfolio/bike-sharing` is empty, and no Kaggle credentials are configured to fetch it. The report lists it explicitly.
    - **Cross-check:** the numbers reproduce the September representative evidence exactly:
      - Essay QWK 0.4738;
      - ROGII RMSE 16.4913 against a 16.5338 baseline;
      - Store Sales RMSLE 0.5282;
      - Parkinson mean AP 0.2208;
      - CMI QWK 0.3597 (tabular) and 0.3688 (fused);
      - CMI sequence QWK 0.3793 with ablation 0.3292, matching PR #76.

      So Epic 6 changed no consumer behavior.
- [x] Task 3: Deletion measure (AC 3). `tools/consumer_metrics.py` reports lines per category (ingestion, model-side, wiring, flow) for every consumer, keeping nested consumers separate, plus the banned-definition scan: `nn.Module`/`Dataset` subclasses, and local functions passed as `objective`, `collate_fn`, `normalizer`, `validator` or `dataset_factory`.
- [x] Task 4: `docs/component-warehouse/deletion-baseline.md` records the deletion measure at `fe37300`:
  - **Kaggle non-ingestion lines:** 5,794 (model-side 1,265, wiring 3,634, flow 895);
  - **Banned local definitions:** 70, listed individually.

## Dev Notes

- The inventory figures in `brownfield.md` (1,529 / 4,361) predate Story 6.1's repository-wide formatting commit. The authoritative baseline is the table this story records.
- Titanic, bike sharing and digit data live in `~/Datasets/dsio-kaggle-portfolio/`; the other competitions live in `~/Datasets/<slug>`.

## Dev Agent Record

### Agent Model Used

Claude Opus 5.5 (claude-opus-5-5)

### Completion Notes List

- **Process lesson.** Running `uv run` in the same tree mid-run silently re-synced the venv back to the CPU torch wheel underneath the CUDA baseline. That run was discarded and redone with the tree untouched. Environment-mutating runs now get exclusive use of their tree.
- The suites pass: replay, golden, measurement and runner tests (71 + 6). The contract goldens hold, and a deliberate 0.5→0.51 change fails.

### File List

- `tests/golden.py`, `tests/golden_metrics.json` (new); the 11 replay tests
- `tools/consumer_metrics.py`, `tools/representative.py` (new); `tests/test_consumer_metrics.py`, `tests/test_representative.py` (new)
- `docs/component-warehouse/{parity-baseline.json,parity-baseline.md,deletion-baseline.md}` (new)
