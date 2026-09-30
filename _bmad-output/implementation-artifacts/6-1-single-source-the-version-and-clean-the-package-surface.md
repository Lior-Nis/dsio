# Story 6.1: Single-source the version and clean the package surface

Status: done

## Story

As a consumer pinning DSio,
I want the version I install, the version DSio reports and the version recorded in provenance to be one value,
so that evidence identifies exactly what produced it.

## Acceptance Criteria

1. Given the built wheel, when `dsio.__version__`, `importlib.metadata.version("dsio")` and the provenance-recorded DSio version are compared, then all three are equal, and a test fails if they ever diverge.
2. `pyproject.toml` is the only place the version is written. `src/dsio/__init__.py` derives `__version__` from installed metadata and contains no version literal.
3. Given the duplicate `ComponentError` (`config/components.py:32` and `model/chain.py:12`), when the package is inspected, then it is defined once, in `dsio.config.components`. Every former import site uses that definition, and `dsio.model.chain.ComponentError`/`dsio.model.module.ComponentError` remain importable as the same class.
4. Given the repository, when `uv run ruff format --check .` runs in CI, then it passes. The pre-existing differences are normalized in one formatting-only commit, and frozen design history under `docs/superpowers/` is excluded from the formatter, not rewritten.

## Tasks / Subtasks

- [x] Task 1: Version single-sourcing (AC 1, 2)
  - [x] 1.1 RED: add `tests/test_version.py` asserting that `dsio.__version__ == importlib.metadata.version("dsio")` and that no version literal exists in `src/dsio/__init__.py`. It fails today: `0.1.0` != `0.2.0`.
  - [x] 1.2 RED: extend `tests/test_built_distribution.py::test_wheel_installs_with_dependencies_and_root_import_is_inert` (or a sibling test using the same installed environment) to assert that `dsio.__version__` inside the installed wheel equals the wheel METADATA `Version`.
  - [x] 1.3 GREEN: `src/dsio/__init__.py`: `__version__ = importlib.metadata.version("dsio")`. Keep the module import-inert: no MLflow, Prefect or torch imports.
  - [x] 1.4 Assert that the provenance `dsio_version` (`tracking/provenance.py:147` uses `version("dsio")`) equals `dsio.__version__`, in an existing provenance test.
- [x] Task 2: One `ComponentError` (AC 3)
  - [x] 2.1 RED: test that `dsio.model.chain.ComponentError is dsio.config.components.ComponentError`, and likewise for `dsio.model.module.ComponentError`.
  - [x] 2.2 GREEN: delete the class in `model/chain.py` and import it from `dsio.config.components`. In `model/module.py`, drop the `ConfiguredComponentError` alias and catch `ComponentError` directly. Verify the `try` block at `module.py:~110-128` cannot newly swallow chain errors (no chain call inside it).
  - [x] 2.3 Existing tests `tests/model/test_module.py:52,182` and `tests/config/test_component_resolution.py:67` pass unchanged.
- [x] Task 3: Formatter gate (AC 4)
  - [x] 3.1 `pyproject.toml`: add `[tool.ruff.format] exclude = ["docs/superpowers/**"]` (the bare directory pattern does not match) with a comment (frozen design history).
  - [x] 3.2 Run `uv run ruff format .` and commit **only** formatting as its own commit: about 68 Python files (src, tests, reference_projects, benchmarks) plus `README.md` code blocks. Then run `ruff check`, `mypy` and the tests.
  - [x] 3.3 `.github/workflows/ci.yml`: add a `Format` step running `uv run ruff format --check .` next to `Lint`.
- [x] Task 4: Full gate: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, `uv run lint-imports`, `uv build`.

## Dev Notes

- **Current state.**
  - `pyproject.toml:3` says `version = "0.2.0"`. `src/dsio/__init__.py:3` says `__version__ = "0.1.0"`.
  - Provenance records `importlib.metadata.version("dsio")` (`src/dsio/tracking/provenance.py:147`), so run evidence already says 0.2.0 while `dsio.__version__` lies. No test compares them.
  - The build backend is hatchling (`pyproject.toml:79-86`). A static version in `[project]` stays the single source. Do **not** switch to hatch dynamic versioning; that would move the source of truth into `__init__.py`.
- **Import inertness.** `tests/test_built_distribution.py::test_wheel_installs_with_dependencies_and_root_import_is_inert` proves `import dsio` has no side effects. `importlib.metadata` is stdlib and safe. An editable or uv-synced checkout is installed, so metadata lookup succeeds in tests and CI (`uv sync --locked`).
- **`ComponentError` semantics.** The config one says "A reusable component cannot be named, imported, configured, or validated". The chain one says "An encoder/head component chain is incomplete or incompatible". Merging them is safe because both subclass `ValueError` and no code distinguishes the two classes. `dsio.model.module` re-exports `ComponentError` in `__all__` (`module.py:384`) and must keep doing so.
- **Formatter scope.** `ruff 0.16.3` also formats Markdown code blocks: `README.md:71`, `docs/superpowers/plans/2026-08-20-cut-the-dead-weight.md`, `docs/superpowers/specs/2026-08-15-dsio-design.md`. `docs/adr` is already excluded via `extend-exclude`. Keep the formatting commit separate so review can skip it.
- **Out of scope.** The `__pycache__`-only directories under `src/dsio/` are untracked (local housekeeping, not a repo change). Do not bump the version; 0.3.0 happens in Story 11.3.

### Project Structure Notes

- New test: `tests/test_version.py` (top-level, alongside `tests/test_distribution.py`).
- No new package modules.

### References

- [Source: _bmad-output/planning-artifacts/epics-component-warehouse.md#Story 6.1]
- [Source: _bmad-output/specs/spec-component-warehouse/SPEC.md#CAP-11]
- [Source: _bmad-output/specs/spec-component-warehouse/brownfield.md#Defects and hygiene to fix in v1]

## Dev Agent Record

### Agent Model Used

Claude Opus 5.5 (claude-opus-5-5)

### Debug Log References

- Red: `tests/test_version.py` (3) and the provenance tag assertion failed before the fix (`0.1.0` != `0.2.0`, duplicate `ComponentError`).
- `ruff format` exclude `["docs/superpowers"]` did not match. Two historical docs were reformatted, then restored, and the glob was corrected to `docs/superpowers/**`.

### Completion Notes List

- `dsio.__version__` reads `importlib.metadata.version("dsio")`. `pyproject.toml` is the only version literal. An uninstalled source tree raises `ImportError` with an install hint instead of a bare `PackageNotFoundError`. A fallback value was rejected because it would break the reported == provenance invariant.
- The package root exposes no public non-module names. The metadata module is imported privately as `_metadata`, and a test guards it.
- Tests assert reported == metadata (`tests/test_version.py`). Inside the built wheel they assert installed `__version__` == wheel METADATA == the provenance `dsio.version` tag (`tests/test_built_distribution.py`). The in-repo provenance test compares against `dsio.__version__`.
- The no-literal test rejects `Assign` and `AnnAssign` literals and any string equal to the installed version. Mutation-checked against both a leaked `version` import and `__version__: str = "0.2.0"`.
- `ComponentError` is defined only in `dsio.config.components`. `model/chain.py` imports it, and `DsioModule` catches it directly. The adversarial review confirmed no try block reaches chain code, so behavior is unchanged.
- The formatting-only commit covers 68 Python files plus `README.md` code blocks. The review confirmed identical syntax trees for all Python files. CI now runs `ruff format --check .`.
- Environment note, not a repo change: a stale user install `~/.local/lib/python3.12/site-packages/dsio-0.1.0.dist-info` makes `PYTHONPATH=src python3 -c "import dsio"` report 0.1.0. The uv-managed venv and CI are unaffected.
- Gates: full suite 1021 passed, 1 deselected. `ruff check`, `ruff format --check`, `mypy` (97 files) and `lint-imports` are clean, and `uv build` succeeds. The distribution tests were re-run after the review fixes.

### File List

- `src/dsio/__init__.py`
- `src/dsio/model/chain.py`
- `src/dsio/model/module.py`
- `tests/test_version.py` (new)
- `tests/test_built_distribution.py`
- `tests/tracking/provenance/test_mlflow.py`
- `pyproject.toml`
- `.github/workflows/ci.yml`
- 68 Python files and `README.md` (formatting-only commit)
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
