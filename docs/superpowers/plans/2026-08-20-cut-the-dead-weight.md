# Cut the Dead Weight — Implementation Plan (1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete every subsystem the lean design cuts, without changing any surviving behaviour, keeping the test suite green after every task.

**Architecture:** Pure subtraction. No module is redesigned, no interface changes shape, no file moves to a new package. Two exceptions are behaviour-preserving repairs the deletions force: `data/cache.py` is replaced by a much smaller skip-if-exists staging function, and a hidden `data <-> splits` import cycle is broken by relocating one function to the layer that owns it. Everything else is `git rm` plus the consumer edits that follow.

**Tech Stack:** Python 3.12, pydantic 2, numpy 2, typer, pytest, ruff, mypy, import-linter, uv.

**Spec:** `docs/superpowers/specs/2026-08-20-dsio-lean-design.md`

## Global Constraints

- Work inside `lib/dsio/`. This plan does **not** collapse the workspace or remove Copier — that is Plan 2.
- Python `>=3.12`. Do not introduce new runtime dependencies; this plan only removes.
- `ruff` line-length 100, rule set `E,F,W,I,UP,B,BLE,SIM,RUF`. `BLE` (blind except) is load-bearing — never silence it with a bare `except`.
- `mypy` runs with `disallow_untyped_defs = true` over `src/dsio`. Every function you write or edit needs annotations.
- **Do not delete `train/tabular.py`.** `tests/conftest.py` lines 37 and 69 construct fixtures from `TabularTask`; removing it takes the entire suite down. Plan 2 replaces the fixture first.
- **Do not delete `splits/temporal.py`.** Purged, embargoed walk-forward is deliberately kept (spec, "Kept, and why").
- The full verification command, referred to below as **VERIFY**:
  ```bash
  cd lib/dsio && uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
  ```
- Every task ends with VERIFY passing and a commit. If VERIFY fails, fix it inside that task — never carry a red suite into the next one.

---

### Task 1: Establish the baseline

**Files:**
- Modify: none

**Interfaces:**
- Consumes: nothing
- Produces: a recorded baseline line count and a known-green suite that every later task is compared against

- [ ] **Step 1: Confirm the suite is green before touching anything**

Run:
```bash
cd lib/dsio && uv sync && uv run pytest -q
```
Expected: all tests pass. If anything fails on a clean checkout, stop and report — the rest of this plan assumes a green baseline.

- [ ] **Step 2: Record the starting size**

Run:
```bash
cd lib/dsio && find src -name '*.py' | xargs wc -l | tail -1
```
Expected: `13345 total` (or close, if the tree has drifted). Write the number down; Task 13 compares against it.

- [ ] **Step 3: Confirm the other gates pass**

Run:
```bash
cd lib/dsio && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: all three clean.

- [ ] **Step 4: Create the working branch**

```bash
git checkout -b lean/01-cut-dead-weight
```

---

### Task 2: Delete `tracking/`

`tracking/` has **zero consumers** — nothing in `src/` imports it. It was built and never wired in. Lightning's `MLFlowLogger` replaces it (spec, decision 1).

**Files:**
- Delete: `src/dsio/tracking/` (entire package: `__init__.py`, `base.py`, `mlflow_sink.py`)
- Modify: `pyproject.toml` — remove the `mlflow` and `wandb` optional-dependency extras

**Interfaces:**
- Consumes: nothing
- Produces: nothing (removal only)

- [ ] **Step 1: Prove there are no consumers**

Run:
```bash
cd lib/dsio && grep -rn "from dsio.tracking\|import dsio.tracking" src/ tests/
```
Expected: no output. If anything appears, stop — the dependency map has drifted and this task needs revisiting.

- [ ] **Step 2: Delete the package**

```bash
cd lib/dsio && git rm -r src/dsio/tracking
```

- [ ] **Step 3: Remove the now-unused extras from `pyproject.toml`**

Delete these two lines from `[project.optional-dependencies]`:
```toml
mlflow = ["mlflow>=3.0"]
wandb = ["wandb>=0.19"]
```

- [ ] **Step 4: VERIFY**

Run:
```bash
cd lib/dsio && uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Delete tracking/: Lightning's Logger already is this, and nothing imported it"
```

---

### Task 3: Delete `agents/`

Agentic work is out of scope under Lightning-only (spec, decision 1). Only tests import it.

**Files:**
- Delete: `src/dsio/agents/` (entire package), `tests/agents/`

**Interfaces:**
- Consumes: nothing
- Produces: nothing

- [ ] **Step 1: Confirm only tests depend on it**

Run:
```bash
cd lib/dsio && grep -rln "from dsio.agents" src/ tests/
```
Expected: exactly `tests/agents/test_agents.py` and `tests/agents/test_agent_task.py`. No `src/` paths.

- [ ] **Step 2: Delete the package and its tests**

```bash
cd lib/dsio && git rm -r src/dsio/agents tests/agents
```

- [ ] **Step 3: Check for a stale registry entry**

Run:
```bash
cd lib/dsio && grep -rn "agent" src/dsio/config/ src/dsio/cli/
```
Expected: no output. If a task kind or preset references agents, delete that line too.

- [ ] **Step 4: VERIFY**

Run:
```bash
cd lib/dsio && uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Delete agents/: an eval harness, not a training path, and out of scope"
```

---

### Task 4: Delete `matrix/` and its CLI

Fold-as-process makes resume free (spec, decision 6); Optuna and MLflow sweeps handle search externally.

**Files:**
- Delete: `src/dsio/matrix/`, `src/dsio/cli/matrix_cmd.py`, `tests/matrix/`
- Modify: `src/dsio/cli/main.py` — remove the `matrix_cmd` import and its `registered_commands.extend(...)` line
- Modify: `pyproject.toml` — remove the `search` extra and `optuna` from the dev group; remove `dsio.cli.matrix_cmd` from the CLI independence import contract

**Interfaces:**
- Consumes: nothing
- Produces: nothing

- [ ] **Step 1: Delete the package, the command, and the tests**

```bash
cd lib/dsio && git rm -r src/dsio/matrix src/dsio/cli/matrix_cmd.py tests/matrix
```

- [ ] **Step 2: Unmount the sub-app in `src/dsio/cli/main.py`**

Remove `matrix_cmd,` from the `from dsio.cli import (...)` block, and delete these two lines:
```python
# matrix_cmd's commands sit at the top level: `dsio matrix`, `dsio search`.
app.registered_commands.extend(matrix_cmd.app.registered_commands)
```

- [ ] **Step 3: Drop the dependencies and the contract entry in `pyproject.toml`**

Remove `search = ["optuna>=4.5"]` from `[project.optional-dependencies]`. Remove the `"optuna>=4.5",` line and its preceding comment from `[dependency-groups].dev`. In the `"CLI command modules are mutually independent"` contract, remove `"dsio.cli.matrix_cmd",` from the `modules` list.

- [ ] **Step 4: VERIFY**

Run:
```bash
cd lib/dsio && uv sync && uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: all pass. `uv sync` is needed here because dependencies changed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Delete matrix/: fold-as-process makes resume free, sweeps live outside"
```

---

### Task 5: Keep the effective-sample-size maths, delete the rest of multiplicity

`multiplicity.py` (493) and `select.py` (202) implement DSR, CSCV/PBO and block bootstrap, which need hundreds of trials to mean anything. Two functions survive because overlapping windows genuinely inflate N regardless of trial count (spec, "What is cut, and why").

**Files:**
- Create: `src/dsio/eval/ess.py`
- Create: `tests/eval/test_ess.py`
- Delete: `src/dsio/eval/multiplicity.py`, `src/dsio/eval/select.py`, `tests/eval/test_multiplicity.py`
- Modify: `src/dsio/eval/__init__.py`

**Interfaces:**
- Consumes: nothing
- Produces: `dsio.eval.ess.autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray` and `dsio.eval.ess.effective_sample_size(x: np.ndarray, max_lag: int = 50) -> float`, both used by later comparison code in Plan 3

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_ess.py`:
```python
import numpy as np
import pytest

from dsio.eval.ess import autocorrelation, effective_sample_size


def test_independent_series_has_ess_close_to_n():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    assert effective_sample_size(x) == pytest.approx(4000, rel=0.25)


def test_correlated_series_has_ess_below_n():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=4000)
    x = np.convolve(noise, np.ones(20) / 20, mode="same")
    assert effective_sample_size(x) < 1000


def test_autocorrelation_at_lag_zero_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    assert autocorrelation(x, max_lag=5)[0] == pytest.approx(1.0)


def test_ess_never_exceeds_n():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    assert effective_sample_size(x) <= 200
```

- [ ] **Step 2: Run it to confirm it fails**

Run:
```bash
cd lib/dsio && uv run pytest tests/eval/test_ess.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'dsio.eval.ess'`.

- [ ] **Step 3: Create `src/dsio/eval/ess.py` by moving the two functions**

Open `src/dsio/eval/multiplicity.py`, find `autocorrelation` and `effective_sample_size`, and move them verbatim into a new `src/dsio/eval/ess.py` with this header:

```python
"""Effective sample size for autocorrelated evidence.

Overlapping windows are not independent examples. A metric computed over 10,000
windows drawn with stride 100 from a 500-step window has nothing like 10,000
independent observations, and any interval that assumes it does is too narrow.
This is kept from the deleted multiplicity layer because it is true at any number
of trials, where deflation and PBO need hundreds to say anything.
"""

from __future__ import annotations

import numpy as np
```

Keep the function bodies exactly as they were. If either references a helper defined elsewhere in `multiplicity.py`, move that helper too.

- [ ] **Step 4: Run the test to confirm it passes**

Run:
```bash
cd lib/dsio && uv run pytest tests/eval/test_ess.py -q
```
Expected: PASS.

- [ ] **Step 5: Delete the rest and fix the package exports**

```bash
cd lib/dsio && git rm src/dsio/eval/multiplicity.py src/dsio/eval/select.py tests/eval/test_multiplicity.py
```

In `src/dsio/eval/__init__.py`, delete the entire `from dsio.eval.multiplicity import (...)` block and any `from dsio.eval.select import (...)` block, then add:
```python
from dsio.eval.ess import autocorrelation, effective_sample_size
```
Remove every name those deleted blocks contributed from `__all__` if one is present, and add `"autocorrelation"` and `"effective_sample_size"`.

- [ ] **Step 6: VERIFY**

Run:
```bash
cd lib/dsio && uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "Keep effective sample size, delete deflation and PBO

Overlapping windows inflate N at any number of trials. Deflated Sharpe and
CSCV need hundreds of configurations to say anything, and Lightning-only
training gives tens."
```

---

### Task 6: Break the hidden `data <-> splits` cycle

`data/adapters.py` line 183 does a **function-local** `from dsio.splits.temporal import window_times` to dodge a circular import. `window_times` computes a window's time span from a `WindowIndex` and a store, which is view knowledge, not splitting knowledge.

**Files:**
- Modify: `src/dsio/data/views.py` — receives `window_times`
- Modify: `src/dsio/splits/temporal.py` — loses `window_times`, imports it from `data.views` if still needed
- Modify: `src/dsio/data/adapters.py:183` — local import becomes a module-level one
- Modify: `pyproject.toml` — add a contract forbidding `data` from importing `splits`
- Test: `tests/data/test_examples.py` (existing coverage of `SignalExamples.times()`)

**Interfaces:**
- Consumes: nothing
- Produces: `dsio.data.views.window_times(...)` — same signature it had in `splits.temporal`

- [ ] **Step 1: Add the contract that will fail**

In `lib/dsio/pyproject.toml`, append:
```toml
[[tool.importlinter.contracts]]
# data/adapters.py used a function-local import to dodge this cycle. The layer
# direction is splits -> data, one way; a local import that hides a cycle is a
# cycle, and the next person to add one will not know it was deliberate.
name = "The data layer never imports splits"
type = "forbidden"
source_modules = ["dsio.data"]
forbidden_modules = ["dsio.splits"]
```

- [ ] **Step 2: Run the contract check to confirm it fails**

Run:
```bash
cd lib/dsio && uv run lint-imports
```
Expected: FAIL — the new contract is broken by `dsio.data.adapters -> dsio.splits.temporal`. (import-linter does detect function-local imports.)

- [ ] **Step 3: Move `window_times` into `data/views.py`**

Cut the whole `window_times` function out of `src/dsio/splits/temporal.py` and paste it into `src/dsio/data/views.py`, below `WindowIndex`. Keep the signature and body identical. If it needs `TimeSpan` or another type from `temporal`, do **not** import it back — inline the plain types it actually uses (`np.ndarray`, `int`, `str`).

In `src/dsio/splits/temporal.py`, add at the top:
```python
from dsio.data.views import window_times
```
and keep re-exporting it, so existing `from dsio.splits.temporal import window_times` call sites keep working.

- [ ] **Step 4: Make the adapter import module-level**

In `src/dsio/data/adapters.py`, delete the local import on line 183 and add at the top of the file:
```python
from dsio.data.views import window_times
```

- [ ] **Step 5: VERIFY**

Run:
```bash
cd lib/dsio && uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: all pass, including the new contract.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Break the data/splits cycle: window_times is view knowledge

A function-local import is still a cycle; it just hides from the linter. The
contract now says out loud that splits depends on data and never the reverse."
```

---

### Task 7: Delete `splits/stratify.py`

406 lines of multi-key greedy assignment with pairwise local search. Stratification is a modelling choice, not a skeleton concern (spec, "What is cut, and why").

**Files:**
- Delete: `src/dsio/splits/stratify.py`, `tests/data/test_stratify.py`
- Modify: `src/dsio/splits/models.py` — drop the `stratify` import, the `stratify`/`stratify_by` fields, the `keys()` property, the validator branch, the `balance` field, and the `to_yaml` reference
- Modify: `src/dsio/splits/__init__.py`
- Modify: `src/dsio/splits/generate.py` — drop its stratify usage (this file is deleted in Task 8; keep it importable until then)

**Interfaces:**
- Consumes: nothing
- Produces: `SplitSpec` without stratification fields; `SplitFile` without `balance`

- [ ] **Step 1: Delete the module and its tests**

```bash
cd lib/dsio && git rm src/dsio/splits/stratify.py tests/data/test_stratify.py
```

- [ ] **Step 2: Strip stratification from `src/dsio/splits/models.py`**

Delete line 33 (`from dsio.splits.stratify import BalanceReport, StratifyKey`). Then delete:
- the `stratify: tuple[StratifyKey, ...] = Field(...)` field (around line 66)
- the `stratify_by: str | None = Field(...)` field (around line 73)
- the `keys()` property (around line 106)
- the `balance: BalanceReport | None = Field(...)` field on `SplitFile` (around line 137)

In the `_check` validator, delete this branch:
```python
if self.scheme == "stratified_kfold" and not (self.stratify or self.stratify_by):
```
and the duplicate-name check that follows it (the `names = [key.name for key in self.stratify]` block).

Remove `"stratified_kfold",` from the `Scheme` Literal.

In `to_yaml` (around line 214), delete the fragment:
```python
+ (f", stratified by {self.spec.stratify_by}" if self.spec.stratify_by else "")
```

- [ ] **Step 3: Strip stratify use from `src/dsio/splits/generate.py`**

Delete its `from dsio.splits.stratify import ...` line and any call into it. Where a stratified assignment was chosen, fall through to the plain shuffled assignment already in `_shuffled`. This file is deleted entirely in the next task — the goal here is only to keep the tree importable.

- [ ] **Step 4: Clean the package exports**

In `src/dsio/splits/__init__.py`, remove every name that came from `stratify` (`BalanceReport`, `StratifyKey`, and any others), from both the import block and `__all__`.

- [ ] **Step 5: VERIFY**

Run:
```bash
cd lib/dsio && uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: all pass. Any test asserting on `stratified_kfold` or `balance` should be deleted, not weakened.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Delete stratify.py: balancing keys is a modelling choice, not a spine one"
```

---

### Task 8: Delete split generation, keep the reader and the temporal maths

`generate.py` reimplements sklearn's `GroupKFold`, `StratifiedGroupKFold` and `LeaveOneGroupOut`. Generation is an offline script whose output is a committed YAML file, so it can depend on sklearn freely (spec, "What is cut, and why"). `temporal.py` stays.

**Files:**
- Delete: `src/dsio/splits/generate.py`
- Modify: `src/dsio/splits/models.py` — delete the `SplitSpec` class and the `spec` field on `SplitFile`
- Modify: `src/dsio/splits/__init__.py`
- Modify: `src/dsio/cli/splits_cmd.py` — remove `make` and `make-temporal` (the whole file goes in Task 11; keep it importable)
- Test: `tests/data/test_splits.py` — delete tests that call `generate`/`write_splits`

**Interfaces:**
- Consumes: nothing
- Produces: `SplitFile` with fields `schema_version`, `store`, `store_manifest_sha256`, `group_key`, `name`, `fold`, `counts`, `notes`, `parts`, `temporal`

- [ ] **Step 1: Delete the generator**

```bash
cd lib/dsio && git rm src/dsio/splits/generate.py
```

- [ ] **Step 2: Remove `SplitSpec` from `src/dsio/splits/models.py`**

Delete the entire `class SplitSpec(DsioModel):` block (including `_check`, `digest`, and the `Scheme` Literal above it, which nothing else uses now). On `SplitFile`, delete the field:
```python
spec: SplitSpec | None = None
```
and any `self.spec` reference in `to_yaml` — replace the provenance header line with a static one, since generation no longer records itself:
```python
# The generating script is a project concern now; `notes` carries whatever it
# wants to say about how this file was produced.
```

- [ ] **Step 3: Strip the generation commands**

In `src/dsio/cli/splits_cmd.py`, delete the `make` and `make-temporal` commands and the `from dsio.splits.generate import ...` line. Leave `show` and `check` in place — Task 11 deletes the file entirely.

- [ ] **Step 4: Delete the generation tests**

In `tests/data/test_splits.py`, delete every test that calls `generate`, `write_splits`, `generate_temporal` or `write_temporal_splits`, and any that constructs a `SplitSpec`. Keep every test that loads a `SplitFile` from YAML and resolves it — those cover the surviving path. Where a deleted test provided a fixture the survivors need, replace it with a literal YAML string written to `tmp_path`.

- [ ] **Step 5: VERIFY**

Run:
```bash
cd lib/dsio && uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Delete split generation: sklearn already has the splitters

Generation is an offline script whose output is a committed file, so it can
depend on anything. The skeleton only needs to read what it produced."
```

---

### Task 9: Replace the stage cache with skip-if-exists staging

`cache.py` is 478 lines: a `CachePolicy` Protocol with four implementations and a `Codec` Protocol with four more, for a job that is "hash the config, write the file, skip if the key matches" (spec, "What is cut, and why").

**Files:**
- Create: `src/dsio/data/staging.py`
- Create: `tests/data/test_staging.py`
- Delete: `src/dsio/data/cache.py`, `tests/data/test_cache.py`
- Modify: `src/dsio/data/__init__.py`, `src/dsio/cli/envelope.py`

**Interfaces:**
- Consumes: `dsio.contracts.hashing.canonical_json`
- Produces: `dsio.data.staging.stage(name: str, config: dict[str, Any], build: Callable[[Path], None], *, root: Path | None = None) -> Path` and `dsio.data.staging.StagingError`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_staging.py`:
```python
from pathlib import Path

import pytest

from dsio.data.staging import StagingError, stage


def test_builds_once_and_skips_on_repeat(tmp_path: Path):
    calls = []

    def build(out: Path) -> None:
        calls.append(out)
        out.write_bytes(b"payload")

    first = stage("windows", {"length": 500}, build, root=tmp_path)
    second = stage("windows", {"length": 500}, build, root=tmp_path)

    assert first == second
    assert len(calls) == 1
    assert first.read_bytes() == b"payload"


def test_different_config_is_a_different_path(tmp_path: Path):
    def build(out: Path) -> None:
        out.write_bytes(b"x")

    a = stage("windows", {"length": 500}, build, root=tmp_path)
    b = stage("windows", {"length": 250}, build, root=tmp_path)
    assert a != b


def test_a_failed_build_leaves_nothing_behind(tmp_path: Path):
    def build(out: Path) -> None:
        out.write_bytes(b"partial")
        raise RuntimeError("boom")

    with pytest.raises(StagingError):
        stage("windows", {"length": 500}, build, root=tmp_path)

    def good(out: Path) -> None:
        out.write_bytes(b"complete")

    assert stage("windows", {"length": 500}, good, root=tmp_path).read_bytes() == b"complete"
```

- [ ] **Step 2: Run it to confirm it fails**

Run:
```bash
cd lib/dsio && uv run pytest tests/data/test_staging.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'dsio.data.staging'`.

- [ ] **Step 3: Write `src/dsio/data/staging.py`**

```python
"""Staging: build a derived artifact once, keyed by the config that produced it.

The predecessor was a pluggable policy-and-codec framework. The job it did is
small: hash the config, name a path from it, build if the path is missing. A
project that later needs environment-sensitive keys adds a field to the dict it
passes in, which is ten lines rather than a Protocol.

A build that raises leaves nothing behind. A half-written stage that looks
complete is worse than a missing one, because the next run reuses it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from dsio.contracts.hashing import canonical_json
from hashlib import sha256


class StagingError(RuntimeError):
    """Raised when a stage could not be built."""


def stage(
    name: str,
    config: dict[str, Any],
    build: Callable[[Path], None],
    *,
    root: Path | None = None,
) -> Path:
    """Return the path to a staged artifact, building it if it is not there."""
    base = Path(root) if root is not None else Path("stores")
    key = sha256(canonical_json(config).encode()).hexdigest()[:16]
    target = base / name / key
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target

    partial = target.with_suffix(".partial")
    try:
        build(partial)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise StagingError(f"stage {name!r} failed to build: {exc}") from exc
    partial.replace(target)
    return target
```

`canonical_json` is `def canonical_json(value: Any) -> str` (`src/dsio/contracts/hashing.py:60`), which is why the key encodes it before hashing.

- [ ] **Step 4: Run the test to confirm it passes**

Run:
```bash
cd lib/dsio && uv run pytest tests/data/test_staging.py -q
```
Expected: PASS, all three tests.

- [ ] **Step 5: Delete the cache and repoint its consumers**

```bash
cd lib/dsio && git rm src/dsio/data/cache.py tests/data/test_cache.py
```

In `src/dsio/data/__init__.py`, delete the whole `from dsio.data.cache import (...)` block and add:
```python
from dsio.data.staging import StagingError, stage
```

In `src/dsio/cli/envelope.py`, delete `from dsio.data.cache import CacheError` and replace it with:
```python
from dsio.data.staging import StagingError
```
Then in the `_CODES` table, replace the `CacheError` entry with `StagingError` mapped to the same error code and retryability it had.

- [ ] **Step 6: VERIFY**

Run:
```bash
cd lib/dsio && uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "Replace the stage cache with staging: 478 lines to 40

Eight strategy classes and two Protocols for hash-the-config, write-the-file,
skip-if-present. The policies were never swapped."
```

---

### Task 10: Delete `data/remote.py`

Kept at design time only because cloud training was planned; cloud is now explicitly deferred, so it is written against imagined requirements (spec, "What is cut, and why"). ADR 0009 is **suspended, not superseded** — revisit it during the cloud design.

**Files:**
- Delete: `src/dsio/data/remote.py`, `tests/data/test_remote.py`
- Modify: `src/dsio/data/__init__.py`, `src/dsio/cli/envelope.py`, `src/dsio/cli/data_cmd.py`
- Modify: `pyproject.toml` — remove `fsspec` from the dev group and drop the `data` extra's `fsspec` entry
- Modify: `docs/adr/0009-remotes-are-content-addressed.md` — mark it suspended

**Interfaces:**
- Consumes: nothing
- Produces: nothing

- [ ] **Step 1: Delete the module and its tests**

```bash
cd lib/dsio && git rm src/dsio/data/remote.py tests/data/test_remote.py
```

- [ ] **Step 2: Remove the exports and the error codes**

In `src/dsio/data/__init__.py`, delete the `from dsio.data.remote import (...)` block and every name it contributed to `__all__`.

In `src/dsio/cli/envelope.py`, delete:
```python
from dsio.data.remote import RemoteError, RemoteIntegrityError
```
Delete both entries from the `_CODES` table, and delete `REMOTE = "remote"` from the `ErrorCode` enum.

- [ ] **Step 3: Strip the remote commands**

In `src/dsio/cli/data_cmd.py`, delete the `push`, `pull` and `status` commands and the `from dsio.data.remote import ...` line. Leave `ls`, `show`, `verify` and `index` — Task 11 deletes the file.

- [ ] **Step 4: Drop the dependency**

In `pyproject.toml`, remove `"fsspec>=2025.1",` and its explanatory comment from `[dependency-groups].dev`, and change the `data` extra to `data = ["pyarrow>=18"]`.

- [ ] **Step 5: Mark ADR 0009**

At the top of `docs/adr/0009-remotes-are-content-addressed.md`, directly under the `Status:` line, add:
```markdown
Suspended (2026-08-20): the implementation is removed with `data/remote.py` because cloud
training is deferred. The decision is not overturned — revisit it on its merits when the
cloud design has real requirements to answer to.
```

- [ ] **Step 6: VERIFY**

Run:
```bash
cd lib/dsio && uv sync && uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "Delete remote.py: written against imagined cloud requirements

ADR 0009 suspended rather than superseded — the design gets revisited when
there is a real cloud setup to answer to."
```

---

### Task 11: Cut the CLI from 26 commands to one

A command exists only if it is needed *before* there is a Python session (spec, decision 9). That leaves `dsio run`.

**Files:**
- Delete: `src/dsio/cli/data_cmd.py`, `src/dsio/cli/splits_cmd.py`, `src/dsio/cli/eval_cmd.py`, `src/dsio/cli/runs_cmd.py`, `src/dsio/cli/artifacts_cmd.py`, `tests/cli/test_data_splits_eval_cli.py`
- Modify: `src/dsio/cli/main.py`, `src/dsio/cli/run_cmd.py`, `tests/cli/test_cli.py`
- Modify: `pyproject.toml` — delete the CLI independence contract entirely

**Interfaces:**
- Consumes: nothing
- Produces: a `dsio` app exposing exactly `run` (and `presets` folded into it)

- [ ] **Step 1: Delete the command modules**

```bash
cd lib/dsio && git rm src/dsio/cli/data_cmd.py src/dsio/cli/splits_cmd.py \
  src/dsio/cli/eval_cmd.py src/dsio/cli/runs_cmd.py src/dsio/cli/artifacts_cmd.py \
  tests/cli/test_data_splits_eval_cli.py
```

- [ ] **Step 2: Reduce `src/dsio/cli/main.py`**

Replace the import block and every `add_typer` / `registered_commands.extend` line with just:
```python
from dsio.cli import run_cmd
from dsio.cli.envelope import emit, failure

app = typer.Typer(
    name="dsio",
    help="Reproducible ML/DL experimentation.",
    no_args_is_help=True,
    add_completion=False,
)
app.registered_commands.extend(run_cmd.app.registered_commands)
```
Leave `main()`, `_is_usage_error` and `_is_abort` untouched.

- [ ] **Step 3: Fold `presets` into a bare `run`**

In `src/dsio/cli/run_cmd.py`, change the `run` command's preset argument to be optional and list the registry when it is omitted:
```python
@app.command()
@json_command
def run(
    preset: Annotated[str | None, typer.Argument(help="Preset to run; omit to list them.")] = None,
    overrides: Annotated[list[str] | None, typer.Argument()] = None,
) -> dict[str, Any]:
    """Run a preset, or list the available presets when called bare."""
    if preset is None:
        return ok(presets=sorted(PRESETS.names()))
    ...
```
Keep the existing body for the non-`None` branch exactly as it is.

Then delete the separate command — it is `def list_presets()` decorated with
`@app.command("presets")` and `@json_command` at `src/dsio/cli/run_cmd.py:88`. Move its
payload verbatim into the `preset is None` branch so bare `dsio run` still reports each
preset's accepted arguments, not just its name:

```python
    if preset is None:
        _bootstrap()
        return {
            "presets": {
                name: [
                    param
                    for param in preset_parameters(name)
                    if param not in {"args", "kwargs"}
                ]
                for name in PRESETS.names()
            }
        }
```

`Registry.names() -> tuple[str, ...]` is at `src/dsio/config/registry.py:79`.

- [ ] **Step 4: Delete the contract that no longer has subjects**

In `pyproject.toml`, delete the whole `"CLI command modules are mutually independent"` contract block. One command module cannot be mutually independent of anything.

- [ ] **Step 5: Prune `tests/cli/test_cli.py`**

Delete every test that invokes a removed command. Keep and, if needed, adapt: the envelope shape test (`{ok, error, code, retryable}`), the unknown-preset error-code test, and the `dsio run <preset>` happy path. Add one test that bare `dsio run` lists presets:
```python
def test_bare_run_lists_presets(cli_runner):
    result = cli_runner.invoke(app, ["run"])
    assert result.exit_code == 0
    assert "presets" in json.loads(result.stdout)
```
Match `cli_runner` to whatever fixture the file already uses.

- [ ] **Step 6: VERIFY**

Run:
```bash
cd lib/dsio && uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "Cut the CLI to one command

Inspection commands existed because the spine was a library you could not
edit. Integrity-on-load and the clean-tree gate live in the loader and the
registry, not in the commands that called them."
```

---

### Task 12: Vectorise the cross-fold disjointness check

`_assert_test_parts_are_disjoint` iterates `fold.test.tolist()` into a Python dict. On a windowed corpus across ten folds that is tens of millions of dict operations, paid on every run before a single model is fitted.

**Files:**
- Modify: `src/dsio/splits/folds.py`
- Test: `tests/eval/test_folds_from_splits.py`

**Interfaces:**
- Consumes: `dsio.eval.contract.Fold`
- Produces: same function, same exception type, same message shape

- [ ] **Step 1: Write the failing test**

Add to `tests/eval/test_folds_from_splits.py`:
```python
import numpy as np
import pytest

from dsio.eval.contract import Fold
from dsio.splits.folds import _assert_test_parts_are_disjoint
from dsio.splits.models import SplitError


def test_overlapping_test_parts_are_rejected():
    a = Fold(index=0, train=np.array([2, 3]), test=np.array([0, 1]), val=None, name="a")
    b = Fold(index=1, train=np.array([3]), test=np.array([1, 2]), val=None, name="b")
    with pytest.raises(SplitError, match="disjoint"):
        _assert_test_parts_are_disjoint([a, b])


def test_disjoint_test_parts_pass():
    a = Fold(index=0, train=np.array([2, 3]), test=np.array([0, 1]), val=None, name="a")
    b = Fold(index=1, train=np.array([0, 1]), test=np.array([2, 3]), val=None, name="b")
    _assert_test_parts_are_disjoint([a, b])


# NOTE: `Fold.__post_init__` (eval/contract.py) already rejects a fold whose own
# train and test overlap. Every fold constructed here must be internally valid, or
# the test fails in the constructor and never reaches the function under test.


def test_large_fold_set_is_fast():
    folds = [
        Fold(
            index=i,
            train=np.array([0]),
            test=np.arange(i * 200_000, (i + 1) * 200_000),
            val=None,
            name=f"f{i}",
        )
        for i in range(10)
    ]
    _assert_test_parts_are_disjoint(folds)
```

- [ ] **Step 2: Run it and note the timing**

Run:
```bash
cd lib/dsio && uv run pytest tests/eval/test_folds_from_splits.py -q --durations=5
```
Expected: PASS, but `test_large_fold_set_is_fast` takes on the order of seconds. Write the duration down.

- [ ] **Step 3: Replace the body with a vectorised check**

In `src/dsio/splits/folds.py`, replace the loop inside `_assert_test_parts_are_disjoint` with:
```python
    positions = np.concatenate([fold.test for fold in folds]) if folds else np.empty(0, dtype=np.int64)
    values, counts = np.unique(positions, return_counts=True)
    repeated = values[counts > 1]
    if repeated.size:
        owners = {
            int(position): [fold.name for fold in folds if position in set(fold.test.tolist())]
            for position in repeated[:5].tolist()
        }
        detail = "; ".join(f"{pos} in {' and '.join(names)}" for pos, names in owners.items())
        raise SplitError(
            f"{repeated.size} example(s) appear in more than one test part; "
            f"folds must test disjoint examples: {detail}"
        )
```
The per-fold name lookup only runs on the first five offending positions, so the slow path costs nothing when the check passes — which is every run except a broken one.

Keep the docstring as it is; it explains why this is checked before fitting rather than after.

- [ ] **Step 4: Run the tests again and compare**

Run:
```bash
cd lib/dsio && uv run pytest tests/eval/test_folds_from_splits.py -q --durations=5
```
Expected: PASS, and `test_large_fold_set_is_fast` now well under a second.

- [ ] **Step 5: VERIFY**

Run:
```bash
cd lib/dsio && uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Vectorise the disjointness check

It walked every test row through a Python dict before any model was fitted.
The name lookup now only runs on positions that actually collide."
```

---

### Task 13: Trim the package façades and the last duplicate implementations

41 symbols re-exported from `data/__init__.py` and 30 from `eval/__init__.py` existed so consumers of a read-only library had a stable surface. Plan 2 makes this a repo you own; the façade is maintenance with no reader. Two second implementations go with them.

**Files:**
- Modify: every `src/dsio/*/__init__.py`
- Modify: `src/dsio/data/readers.py` — delete `FileReader`
- Modify: `src/dsio/data/adapters.py` — delete `KeyedExamples`
- Test: existing suite; adjust imports where tests used the façade

**Interfaces:**
- Consumes: nothing
- Produces: modules importable by their defining path, e.g. `from dsio.data.store import SignalStore`

- [ ] **Step 1: Delete `FileReader` and `KeyedExamples`**

In `src/dsio/data/readers.py`, delete `class FileReader`. In `open_reader`, delete the branch that returns it and raise on an unknown backend:
```python
    raise ReadError(
        f"unknown backend {backend!r}; the store is memory-mapped (ADR 0005). "
        "A sharded streaming backend is the planned extension, not a fallback."
    )
```
Keep the `SignalReader` Protocol — it is the seam a shards backend would plug into.

In `src/dsio/data/adapters.py`, delete `class KeyedExamples`. It subclasses `TableExamples` to add a `keys` accessor and is not distinct from it.

- [ ] **Step 2: Run the suite to find what breaks**

Run:
```bash
cd lib/dsio && uv run pytest -q
```
Expected: failures only in tests that referenced the two deleted classes. Delete those tests — do not adapt them to the survivors, since they were testing the deleted behaviour.

- [ ] **Step 3: Reduce every `__init__.py` to its docstring**

For each of `src/dsio/data/`, `src/dsio/eval/`, `src/dsio/splits/`, `src/dsio/config/`, `src/dsio/runs/`, `src/dsio/artifacts/`, `src/dsio/nn/`, `src/dsio/train/`, `src/dsio/contracts/`: keep the module docstring, delete every `from ... import (...)` block and every `__all__`.

Leave one exception: `src/dsio/config/__init__.py` keeps `RunConfig` and `preset`, because `from dsio.config import RunConfig, preset` is the one import a project's `presets.py` writes on day one and it is part of the template's generated code.

**This breaks `src/dsio/cli/run_cmd.py:11`**, which currently reads:
```python
from dsio.config import PRESETS, RunConfig, load_preset_modules, preset_parameters, resolve
```
Repoint it at the defining modules rather than widening the façade back out:
```python
from dsio.config.presets import PRESETS, load_preset_modules, preset_parameters, resolve
from dsio.config.schema import RunConfig
```

- [ ] **Step 4: Repair the import sites**

Run:
```bash
cd lib/dsio && uv run pytest -q 2>&1 | grep ImportError
```
For each failure, change the import to name the defining module — `from dsio.data import SignalStore` becomes `from dsio.data.store import SignalStore`. Repeat until the suite is green. Do the same for `src/` files that imported through a façade.

- [ ] **Step 5: VERIFY**

Run:
```bash
cd lib/dsio && uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: all pass.

- [ ] **Step 6: Measure the result**

Run:
```bash
cd lib/dsio && find src -name '*.py' | xargs wc -l | tail -1
```
Expected: roughly `7700 total`, down from the ~13,345 recorded in Task 1. If it is far off, list the per-package counts and report the discrepancy before continuing:
```bash
cd lib/dsio/src/dsio && for d in */; do echo "$(find $d -name '*.py' | xargs cat | wc -l) $d"; done | sort -rn
```

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "Trim the façades and the last duplicate readers

Forty-one re-exported symbols were a stable surface for a library you could
not edit. Imports now name the module that defines the thing."
```

---

## Done when

- VERIFY passes on `lean/01-cut-dead-weight`.
- `src/dsio` is roughly 7,700 lines, down from 13,345.
- `src/dsio/agents`, `matrix`, `tracking`, `data/cache.py`, `data/remote.py`, `eval/multiplicity.py`, `eval/select.py`, `splits/stratify.py`, `splits/generate.py` and five CLI command modules no longer exist.
- `dsio run` is the only command, and bare `dsio run` lists presets.
- `lint-imports` enforces that `data` never imports `splits`.
- `train/tabular.py` and `splits/temporal.py` are **untouched** — both are deliberate.

## Not in this plan

Collapsing the workspace, removing Copier, the loss-contract change, dissolving `ssl/`, the directory restructure, torchmetrics, deleting `cross_validate`, and MLflow. Those are Plans 2 and 3.
