# Story 6.6: Record golden baselines and the deletion measure

Status: in-progress

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
- [ ] Task 2: Representative tier (AC 2).
  - [x] 2.1 `tools/representative.py` runs a consumer's own flow on its real data. It targets the live MLflow server (default `http://localhost:5000`), uses an isolated `PREFECT_HOME` and a fresh workspace, and records metrics, run IDs, commit, dirty flag, hardware and duration. `--compare --tolerance` gives the parity stories 7.7, 8.7, 9.6 and 10.6 their check.
  - [ ] 2.2 Record all 9 Kaggle consumers from a clean tree at the baseline commit. CMI sequence runs on CUDA (RTX 5070 Ti) with a temporary consumer-side CUDA wheel; the others run on CPU.
- [x] Task 3: Deletion measure (AC 3). `tools/consumer_metrics.py` reports lines per category (ingestion, model-side, wiring, flow) for every consumer, keeping nested consumers separate, plus the banned-definition scan: `nn.Module`/`Dataset` subclasses, and local functions passed as `objective`, `collate_fn`, `normalizer`, `validator` or `dataset_factory`.
- [ ] Task 4: Record the measures in `parity-baseline.md` and the sprint status.

## Dev Notes

- The inventory figures in `brownfield.md` (1,529 / 4,361) predate Story 6.1's repository-wide formatting commit. The authoritative baseline is the table this story records.
- Titanic, bike sharing and digit data live in `~/Datasets/dsio-kaggle-portfolio/`; the other competitions live in `~/Datasets/<slug>`.

## Dev Agent Record

### Agent Model Used

Claude Opus 5.5 (claude-opus-5-5)

### Completion Notes List

### File List
