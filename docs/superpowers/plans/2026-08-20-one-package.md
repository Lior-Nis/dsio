# One Package — Implementation Plan (2a of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the Copier template and its two-distribution workspace into a single package rooted at `src/dsio/` that runs its own test suite from a fresh clone, and give it a torch setup that serves CI on CPU and a local Blackwell GPU on cu128 simultaneously.

**Architecture:** Packaging only. No module is renamed, no source package is restructured, no interface changes shape. `lib/dsio/src/dsio/` moves to `src/dsio/`, `lib/dsio/tests/` moves to `tests/`, and `lib/dsio/pyproject.toml` becomes the root `pyproject.toml`. Everything the two-distribution split *required* — the Jinja templating, the entry-point preset discovery, the `project → dsio` import contract — is deleted because the thing it bridged no longer has two sides. The Lightning-only reshape is Plan 2b.

**Tech Stack:** Python 3.12, uv, pydantic 2, numpy 2, torch/Lightning, typer, pytest, ruff, mypy, import-linter, Docker.

**Spec:** `docs/superpowers/specs/2026-08-20-dsio-lean-design.md` (decision 2, "One package, cloned and owned"), recorded as `docs/adr/0018-a-repository-not-a-template.md`.

## Global Constraints

- Python `>=3.12`. The only dependency *addition* in this plan is the `pytorch-cu128` index and the two accelerator extras; everything else is a move or a deletion.
- `ruff` line-length 100, rule set `E,F,W,I,UP,B,BLE,SIM,RUF`. `BLE` (blind except) is load-bearing — never silence it with a bare `except`. A typed re-raise (`except X as e: raise Y() from e`) is fine and is used already.
- `mypy` runs with `disallow_untyped_defs = true` over the package.
- **Do not restructure source packages.** `nn/` stays `nn/`, `ssl/` stays `ssl/`, `train/tabular.py` stays. Renaming and dissolving them is Plan 2b, and doing it here makes the move-diff unreviewable.
- **Do not change any function signature or behaviour.** If the suite needs a code change beyond an import path or a config path, stop and report — this plan is a move.
- The package is named `dsio` and stays named `dsio`.
- After the move, the full check runs from the **repository root**, referred to below as **VERIFY**:
  ```bash
  uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
  ```
  Before the move (Task 1 only) it still runs from `lib/dsio/`.
- Baseline to preserve: **363 tests**, 4 import contracts kept, ruff and mypy clean.
- Every task ends with VERIFY passing and a commit. Never carry a red suite into the next task.

---

### Task 1: Baseline and branch

**Files:** none

**Interfaces:**
- Consumes: nothing
- Produces: a recorded baseline every later task is compared against

- [ ] **Step 1: Confirm the suite is green before touching anything**

Run:
```bash
cd lib/dsio && uv sync && uv run pytest -q
```
Expected: 363 passed. If anything fails on a clean checkout, stop and report — the rest of this plan assumes a green baseline.

- [ ] **Step 2: Confirm the other gates and record the size**

Run:
```bash
cd lib/dsio && uv run ruff check . && uv run mypy && uv run lint-imports
cd lib/dsio && find src -name '*.py' | xargs wc -l | tail -1
```
Expected: all clean, 4 contracts kept, `7708 total`. Write the number down.

- [ ] **Step 3: Record the current import-contract names**

Run:
```bash
cd lib/dsio && grep -n '^name = ' pyproject.toml
```
Write down every contract name. Task 8 checks that exactly the intended one disappeared and no other was disturbed.

- [ ] **Step 4: Create the working branch**

```bash
git checkout -b lean/02a-one-package origin/main
```

---

### Task 2: Move the package to the repository root

The single mechanical move. Nothing is renamed inside the package; only its location and the config paths that point at it change.

**Files:**
- Delete: `src/{{ module_name }}/__init__.py.jinja`, `src/{{ module_name }}/presets.py.jinja`, `tests/test_presets.py.jinja`
- Move: `lib/dsio/src/dsio/` → `src/dsio/`; `lib/dsio/tests/` contents → `tests/`; `lib/dsio/uv.lock` → `uv.lock`; `lib/dsio/pyproject.toml` → `pyproject.toml`
- Delete: `pyproject.toml.jinja`, and the now-empty `lib/`
- Modify: the moved `pyproject.toml` (paths only)

**Interfaces:**
- Consumes: nothing
- Produces: `src/dsio/` importable from the repository root; VERIFY runs from the root

- [ ] **Step 1: Remove the template's project-side files first**

They occupy the destination paths, and `src/{{ module_name }}/` will otherwise collide with the move.

```bash
git rm -r 'src/{{ module_name }}' tests/test_presets.py.jinja
```

Note: `tests/test_presets.py.jinja` is already broken — it imports `PRESETS`, `load_preset_modules`, `resolve` from `dsio.config` and `check` from `dsio.train`, none of which are exported after Plan 1's façade trim. Its replacement is Task 3, not a port.

- [ ] **Step 2: Move the package, tests, lockfile and pyproject**

```bash
git mv lib/dsio/src/dsio src/dsio
git mv lib/dsio/uv.lock uv.lock
git mv lib/dsio/pyproject.toml pyproject.toml
for entry in lib/dsio/tests/*; do git mv "$entry" tests/; done
```

Then confirm nothing is left behind but the README (Task 7 handles it):
```bash
find lib -type f
```
Expected: only `lib/dsio/README.md`.

- [ ] **Step 3: Repoint the tool paths in the moved `pyproject.toml`**

Only paths change. Set:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/dsio"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.mypy]
python_version = "3.12"
files = ["src/dsio"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```
These are almost certainly already correct — `lib/dsio/pyproject.toml` used the same relative layout — so verify rather than assume, and change only what is actually wrong.

- [ ] **Step 4: Move the README and delete `lib/`**

```bash
git mv lib/dsio/README.md docs/spine-readme.md
rmdir lib/dsio lib
```
`docs/spine-readme.md` is a staging location only; Task 7 merges its content into the root README and deletes it. Parking it in `docs/` keeps the prose available to Task 7 without leaving `lib/` alive.

- [ ] **Step 5: Sync and run VERIFY from the repository root**

```bash
uv sync
uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: 363 passed, ruff clean, mypy clean, 4 contracts kept.

If `uv sync` complains the lockfile is stale, run `uv lock` and say so in your report — the lock was written against a workspace member and may need regenerating for a single distribution.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Move the package to the repository root

lib/dsio/src/dsio -> src/dsio, lib/dsio/tests -> tests, and lib/dsio's
pyproject becomes the root one. Nothing inside the package is renamed; only
its location and the paths pointing at it."
```

---

### Task 3: Replace entry-point preset discovery

`load_preset_modules()` discovers presets through the `dsio.presets` entry-point group, so that a project could advertise its presets "without dsio ever having to know the project's package name". With one package, dsio *is* the package — the indirection bridges two sides that no longer exist.

**Files:**
- Create: `src/dsio/presets.py`, `tests/config/test_preset_discovery.py`
- Modify: `src/dsio/config/presets.py`
- Test: `tests/config/test_config.py` (existing discovery coverage)

**Interfaces:**
- Consumes: `dsio.config.presets.PRESETS`, `dsio.config.presets.preset`
- Produces: `dsio.config.presets._BUILTIN_PRESET_MODULES: tuple[str, ...]`, and `load_preset_modules() -> list[str]` with an unchanged signature

- [ ] **Step 1: Write the failing test**

Create `tests/config/test_preset_discovery.py`.

**The second test must run in a subprocess, and this is not optional.** `tests/conftest.py`
defines `_spine_preset` with `autouse=True`, which registers a `spine_baseline` preset before
*every* test in the suite. An in-process assertion that "some preset is registered" therefore
passes whether discovery works or not — the same shape of unfalsifiable test that let a
totally broken `nn/__init__.py` keep the suite green in Plan 1.

```python
import importlib
import subprocess
import sys

from dsio.config.presets import _BUILTIN_PRESET_MODULES, load_preset_modules


def test_every_builtin_preset_module_is_importable():
    """A stale entry here would rot exactly as a stale runner entry did."""
    for name in _BUILTIN_PRESET_MODULES:
        importlib.import_module(name)


def test_discovery_registers_a_preset_in_a_clean_interpreter():
    """In-process this is unfalsifiable: conftest's autouse fixture has already
    registered one. Only a fresh interpreter proves discovery itself works."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from dsio.config.presets import PRESETS, load_preset_modules;"
            " load_preset_modules();"
            " assert PRESETS.names(), 'discovery registered nothing'",
        ],
        check=True,
    )


def test_discovery_is_idempotent():
    assert load_preset_modules() == load_preset_modules()
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/config/test_preset_discovery.py -q`
Expected: FAIL — `ImportError: cannot import name '_BUILTIN_PRESET_MODULES'`.

- [ ] **Step 3: Create `src/dsio/presets.py`**

```python
"""Presets: the runnable configurations this project defines.

A preset is a function returning a validated ``RunConfig``. Variants are *arguments*,
not files — `dsio run <preset> seed=7` needs nothing checked in, which is why this repo
has no tree of config files to keep in sync (ADR 0001).

This module is imported by discovery, so anything decorated with ``@preset`` here is
runnable from the CLI. Add your own below; upstream does not touch this file, so it will
not conflict when you merge.
"""

from __future__ import annotations

from dsio.config.presets import preset
from dsio.config.schema import RunConfig
from dsio.train.tabular import TabularTask


@preset
def spine_baseline(
    dataset: str = "iris",
    estimator: str = "logreg",
    seed: int = 42,
) -> RunConfig:
    """Starter baseline. Replace the task with your own once you have data staged."""
    return RunConfig(
        name=f"spine-{estimator}",
        seed=seed,
        tags=("baseline",),
        task=TabularTask(dataset=dataset, estimator=estimator),
    )
```

`spine_baseline` is deliberately the same name `tests/conftest.py` registers — see the next step, which deletes that fixture. Keep the parameters identical to the fixture's (`dataset`, `estimator`, `test_fraction`, `seed`) so no existing test changes meaning, and check `TabularTask`'s signature before writing them.

- [ ] **Step 4: Replace entry-point discovery in `src/dsio/config/presets.py`**

Add near the top, mirroring `dsio/train/__init__.py`'s `_BUILTIN_RUNNER_MODULES`:
```python
#: Modules imported to register presets. A project adds its own module here, or names it
#: in DSIO_PRESET_MODULES. Entry points are gone: they existed so the spine, shipped as a
#: separate distribution, need not know a project's package name. There is one package now.
_BUILTIN_PRESET_MODULES: tuple[str, ...] = ("dsio.presets",)
```

Rewrite `load_preset_modules()` to iterate `_BUILTIN_PRESET_MODULES` instead of `entry_points(group=PRESET_ENTRY_POINT_GROUP)`, keeping the `DSIO_PRESET_MODULES` environment escape hatch exactly as it is. **Keep the existing "a declared entry point that fails to import is an error, not a skip" behaviour** — that reasoning ("a silently empty preset list is indistinguishable from a project with no presets, and the difference matters at 2am") applies unchanged to a module list. Do not add an `except ImportError`.

Delete `PRESET_ENTRY_POINT_GROUP` and the `entry_points` import if nothing else uses them — grep first.

- [ ] **Step 5: Delete the now-redundant conftest fixture**

`tests/conftest.py`'s `_spine_preset` exists for a reason its own docstring states: *"The
spine ships no presets — those belong to a project."* That stops being true in this task. The
package now ships `dsio/presets.py`, so the fixture is a duplicate registration guarded by
`if "spine_baseline" in PRESETS: return`, which silently no-ops and hides whether discovery
ran at all.

Delete the whole `_spine_preset` fixture. Keep its `load_runners()` call alive if other tests
depend on runners being registered — check by running the suite after removing it, and if
something breaks, move that one call into a separate autouse fixture that does *only* that,
with a comment saying why. Do not keep the preset registration.

- [ ] **Step 6: Remove the entry-point declaration**

The `[project.entry-points."dsio.presets"]` table came from `pyproject.toml.jinja`, which Task 2 deleted, so the root `pyproject.toml` should not have one. Confirm with `grep -n "entry-points" pyproject.toml` — expected: no output.

- [ ] **Step 7: Run the tests, then VERIFY**

```bash
uv run pytest tests/config/test_preset_discovery.py -q
uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: the new tests pass; 366 total (363 + 3).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Discover presets by module list, not entry point

The entry-point group existed so a separately-distributed spine need not know
a project's package name. There is one package now, so discovery is a tuple of
module names — the same shape _BUILTIN_RUNNER_MODULES already uses, guarded the
same way."
```

---

### Task 4: Delete the Copier machinery

**Files:**
- Delete: `copier.yml`, `.copier-answers.yml.jinja`
- Modify: `.gitignore` (drop template-era entries if any)

**Interfaces:**
- Consumes: nothing
- Produces: nothing

- [ ] **Step 1: Confirm no `.jinja` file survives**

Run:
```bash
find . -name '*.jinja' -not -path './.git/*'
```
Expected: only `.copier-answers.yml.jinja` (Task 2 removed the rest). If `pyproject.toml.jinja` is still present, Task 2 was incomplete — delete it here and note it in your report.

- [ ] **Step 2: Delete the template definition**

```bash
git rm copier.yml .copier-answers.yml.jinja
```

- [ ] **Step 3: Check `.gitignore` for template-era paths**

Read `.gitignore`. The template excluded generated-project directories (`runs/`, `stores/`, `views/`, `cache/`, `models/`, `data/`). Those are still the right things to ignore — a run still writes `runs/` — so **keep them**, but check for any entry that only made sense with `lib/dsio/` in the path and fix it. Report what you found either way.

- [ ] **Step 4: VERIFY**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: 366 passed, all gates clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Delete the Copier machinery

Jinja existed for exactly one variable: the package name. With the name fixed
there is nothing to template, and git merge does what copier update did."
```

---

### Task 5: Torch on CPU for CI, cu128 for a local Blackwell GPU

The current `[tool.uv.sources] torch = [{ index = "pytorch-cpu" }]` forces `+cpu` wheels for everyone, so a local RTX 5070 Ti (compute capability 12.0, sm_120) resolves CPU torch and `torch.cuda.is_available()` is `False`. The version floor is irrelevant — torch already resolves to 2.13.

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_accelerator_extras.py`

**Interfaces:**
- Consumes: nothing
- Produces: extras `cpu` and `cu128`, mutually exclusive; default resolution unchanged for anyone who selects neither

- [ ] **Step 1: Declare the two indexes and the conflicting extras**

Replace the existing `[[tool.uv.index]]` / `[tool.uv.sources]` block with:

```toml
# torch ships one wheel per accelerator, and they cannot coexist in a resolution.
# CI has no GPU and wants the small CPU wheel; a local Blackwell card (sm_120) needs
# cu128 or it fails at model-build time with "no kernel image is available for
# execution on the device". Declaring both as conflicting extras lets one lockfile
# serve both: `uv sync --extra cpu` in CI, `uv sync --extra cu128` locally.
[project.optional-dependencies]
cpu = ["torch>=2.7"]
cu128 = ["torch>=2.7"]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cu128" }]]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cu128", extra = "cu128" },
]
```

Keep the existing `torch` extra (`torch = ["torch>=2.6", ...]`) but raise its floor to `>=2.7`, since sm_120 support does not exist below it. Leave `lightning` and `torchmetrics` in that extra untouched.

Remove the bare `"torch>=2.6"` and `"lightning>=2.5"` entries from `[dependency-groups].dev` **only if** they now resolve through an extra; if removing them breaks `uv sync --extra cpu`, put them back and report why. The dev group's job is to make the Lightning path testable in CI, and that must keep working.

- [ ] **Step 2: Regenerate the lockfile and confirm both resolutions work**

```bash
uv lock
uv sync --extra cpu
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Expected: a `+cpu` version and `False`.

Then:
```bash
uv sync --extra cu128
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Expected: a `+cu128` version. On a machine with an NVIDIA GPU and a recent driver this prints `True`; on a machine without one it prints `False` and that is correct — the wheel is right, the hardware is absent. Report which you observed and the driver version from `nvidia-smi` if present.

**This step downloads multi-gigabyte wheels. Expect it to be slow.**

- [ ] **Step 3: Write a test that the extras stay mutually exclusive**

Create `tests/test_accelerator_extras.py`:
```python
"""The cpu and cu128 extras must stay declared as conflicting.

Without the conflicts table uv will try to resolve both torch wheels at once and fail,
or worse, silently pick one. This asserts the declaration itself, which is the thing a
careless dependency edit removes.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_cpu_and_cu128_are_declared_conflicting():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())

    extras = data["project"]["optional-dependencies"]
    assert "cpu" in extras and "cu128" in extras

    conflicts = data["tool"]["uv"]["conflicts"]
    pairs = [{entry["extra"] for entry in group} for group in conflicts]
    assert {"cpu", "cu128"} in pairs

    sources = {entry["extra"]: entry["index"] for entry in data["tool"]["uv"]["sources"]["torch"]}
    assert sources == {"cpu": "pytorch-cpu", "cu128": "pytorch-cu128"}
```

- [ ] **Step 4: Run it, then VERIFY on the CPU extra**

```bash
uv sync --extra cpu
uv run pytest tests/test_accelerator_extras.py -q
uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: 367 passed. **Leave the environment synced on `--extra cpu`** so later tasks match CI.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "One lockfile, two accelerators

The old source pin forced +cpu wheels on everyone, so a local Blackwell card
resolved CPU torch and never saw its GPU. Conflicting extras let CI take the
small wheel and a workstation take cu128 without a second lockfile."
```

---

### Task 6: CI runs from the root, on the CPU extra

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the `cpu` extra from Task 5
- Produces: a green CI run

- [ ] **Step 1: Remove the working-directory indirection**

The `spine` job currently sets `defaults.run.working-directory: lib/dsio`, added because the root had no project. Delete that `defaults:` block and its comment — the root *is* the project now.

- [ ] **Step 2: Select the CPU extra explicitly**

Change the Install step to:
```yaml
      - name: Install
        run: uv sync --locked --extra cpu
```
Without `--extra cpu`, uv resolves the accelerator extras' `torch` from the default index rather than the CPU one, and CI downloads CUDA wheels it cannot use.

- [ ] **Step 3: Retire the `fork-drift` job**

Delete it. It warns that "this branch modifies the shared spine — send spine fixes upstream", which assumed the workflow was running in a *fork* of a template. This repository is the upstream, and after this plan there is no separate spine to drift from. Leaving a job whose advice is addressed to nobody is the stale-documentation problem in YAML.

- [ ] **Step 4: Bump the deprecated action**

`actions/checkout@v4` runs on deprecated Node 20. Change both remaining usages to `actions/checkout@v5`. Keep `fetch-depth: 0` — provenance tests need real history and a shallow clone breaks `rev-parse`.

- [ ] **Step 5: VERIFY locally, then confirm the workflow parses**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('workflow parses')"
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "CI runs from the root, on the CPU extra

The working-directory indirection existed because the root had no project.
It has one now. fork-drift goes with it: this repository is the upstream, so
its warning was addressed to nobody."
```

---

### Task 7: One README, and the upstream-remote propagation story

**Files:**
- Modify: `README.md`
- Delete: `docs/spine-readme.md` (the staged `lib/dsio/README.md` from Task 2)

**Interfaces:**
- Consumes: nothing
- Produces: documented instructions that actually run

- [ ] **Step 1: Read both files before writing**

Read `README.md` and `docs/spine-readme.md`. The root README describes a Copier template; the spine README describes a read-only shared package. Neither is true after this plan. Keep the parts that are still true — the design principles, the "store once, index many" and "never block, always reconstructible" sections — and rewrite the rest.

- [ ] **Step 2: Replace the quickstart**

It currently reads `uvx copier copy gh:<you>/dsio my-project` followed by `uv run dsio presets`. Neither works: Copier is gone as of Task 4, and `dsio presets` was deleted in Plan 1 (bare `dsio run` lists them). Write:

````markdown
## Start a project

```bash
git clone https://github.com/<you>/dsio my-project
cd my-project
git remote rename origin upstream
uv sync --extra cu128        # or --extra cpu on a machine without an NVIDIA GPU
uv run dsio run              # lists the presets
uv run dsio run spine_baseline
```

Add your components under `src/dsio/` — a backbone in `model/`, a preset in `presets.py`.
Later, pull spine improvements without losing your work:

```bash
git fetch upstream && git merge upstream/main
```

That is the whole propagation story. It works because the package name is fixed, so paths
line up between your clone and upstream and a merge conflicts only in files you both
edited. The previous Copier template existed to rename the package per project, which is
precisely what stopped a plain merge from working.
````

- [ ] **Step 3: Fold in what the spine README said that is still true**

Its "treat as read-only, send fixes upstream" framing is obsolete — you own the whole tree now. Its description of what dsio owns is not. Merge that prose into the root README's opening, then:

```bash
git rm docs/spine-readme.md
```

- [ ] **Step 4: Fix the Shape section**

It documents a two-distribution workspace (`pyproject.toml` / `src/<module>/` / `lib/dsio/`). Replace with the real single-package layout, and delete the sentence beginning "`project → dsio`, never the reverse. That direction is enforced by packaging" — Task 8 removes that contract, and ADR 0018 records the downgrade.

- [ ] **Step 5: Fix the developing section**

`cd lib/dsio` no longer exists. The commands are now run from the root.

- [ ] **Step 6: Verify every command in the README actually runs**

For each fenced command, run it (or, for `git clone`, confirm the URL form is right). A README whose first three lines fail is how this repository shipped its CI for months. Report any command you could not run and why.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "One README, and a quickstart that runs

The old one opened with copier copy and dsio presets — a tool this plan
deletes and a command Plan 1 deleted. Propagation is now git merge upstream,
which works for the reason the template obscured: the package name is fixed."
```

---

### Task 8: Retire the contract that has no second side

`"dsio never imports project code"` forbids `dsio` from importing a package named `project`. It encoded the two-distribution direction. With one package there is no project distribution to import, and the contract can never fire.

**Files:**
- Modify: `pyproject.toml`
- Modify: `docs/adr/0018-a-repository-not-a-template.md` (status line only)

**Interfaces:**
- Consumes: nothing
- Produces: 3 import contracts where there were 4

- [ ] **Step 1: Confirm the contract is inert**

Run:
```bash
grep -rn "^import project\|^from project" src/ tests/ || echo "no project package exists"
uv run lint-imports | tail -3
```
Expected: no such imports, 4 contracts kept.

- [ ] **Step 2: Delete that contract only**

Remove the whole `[[tool.importlinter.contracts]]` block whose `name = "dsio never imports project code"`, including its explanatory comment. **Leave the other three exactly as they are** — foundation independence, `eval` forbidden imports, and `data` never imports `splits`. Those enforce real leakage walls and are the reason the hidden `data ↔ splits` cycle was found.

- [ ] **Step 3: Mark ADR 0018 as implemented**

Its status block currently says `Implemented: no. This is Plan 2.` Change it to record that Plan 2a implemented it, and that the `project → dsio` direction is now enforced by nothing — there is one package, so the direction it described no longer has two sides. The ADR's Consequences section already discusses this downgrade; do not rewrite it, only the status line.

- [ ] **Step 4: VERIFY**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: 367 passed, **3 contracts kept, 0 broken**.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Retire the contract that has no second side

'dsio never imports project code' encoded the two-distribution direction. One
package, no project distribution, nothing the contract can catch. The three
leakage contracts stay."
```

---

### Task 9: Rebuild the Docker image for one package

**Files:**
- Modify: `Dockerfile`

**Interfaces:**
- Consumes: the root `pyproject.toml` and `uv.lock`
- Produces: an image that runs `dsio run`

- [ ] **Step 1: Rewrite the copy layers**

It currently copies a workspace: `COPY pyproject.toml uv.lock README.md ./`, then `COPY lib/dsio/pyproject.toml lib/dsio/README.md ./lib/dsio/`, then `COPY lib/ ./lib/` and `COPY src/ ./src/`. `lib/` no longer exists. Reduce to the root files plus `src/`, keeping the dependency layer first so source edits do not invalidate the install:

```dockerfile
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --extra cpu --no-install-project

COPY src/ ./src/
RUN uv sync --locked --extra cpu
```

Use `--extra cpu`: the image is for local/cloud parity of the *pipeline*, and baking CUDA wheels into it multiplies its size for a GPU the container may not have. If you disagree after seeing the file, say so in your report rather than switching silently.

- [ ] **Step 2: Keep the git install and say why**

Do not remove the `apt-get install git` layer. Its comment explains it: provenance capture shells out to git, and without it every run in the image records `code_hash=None` and becomes unpromotable. That is still true.

- [ ] **Step 3: Build it**

```bash
docker build -t dsio:plan2a . 2>&1 | tail -20
```
Expected: a successful build. If Docker is unavailable in this environment, say so explicitly in your report rather than marking the step done — an unbuilt Dockerfile is an untested one.

- [ ] **Step 4: Run the CLI inside the image**

```bash
docker run --rm dsio:plan2a uv run dsio run
```
Expected: a JSON envelope listing presets.

- [ ] **Step 5: VERIFY and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy && uv run lint-imports
git add -A
git commit -m "Rebuild the image for one package

The copy layers described a two-member workspace that no longer exists."
```

---

### Task 10: Prove a fresh clone works

The claim this plan exists to make true is that the repository runs itself. Assert it against a real clone, not the working tree.

**Files:** none

**Interfaces:**
- Consumes: everything above
- Produces: evidence for the plan's headline claim

- [ ] **Step 1: Clone the branch into a scratch directory**

```bash
cd /tmp && rm -rf dsio-freshclone
git clone --branch lean/02a-one-package /home/liornisimov/Projects/dsio dsio-freshclone
cd /tmp/dsio-freshclone && ls -A
```
Expected: no `copier.yml`, no `*.jinja`, no `lib/`, and a real `pyproject.toml`.

- [ ] **Step 2: Install and run the full check from scratch**

```bash
cd /tmp/dsio-freshclone && uv sync --extra cpu
cd /tmp/dsio-freshclone && uv run pytest -q
cd /tmp/dsio-freshclone && uv run ruff check . && uv run mypy && uv run lint-imports
```
Expected: 367 passed, all gates clean, 3 contracts. **This is the headline claim: `uv sync && uv run pytest` green in a fresh clone.**

- [ ] **Step 3: Run the CLI from the clone**

```bash
cd /tmp/dsio-freshclone && uv run dsio run
cd /tmp/dsio-freshclone && uv run dsio run spine_baseline
```
Expected: the first lists presets; the second completes a run and writes a run record.

- [ ] **Step 4: Measure and report**

```bash
cd /tmp/dsio-freshclone && find src -name '*.py' | xargs wc -l | tail -1
```
Report the number against Task 1's 7,708. It should be within a few dozen lines — this plan moves code rather than deleting it, so a large change either way means something went wrong. Say so plainly if it did.

- [ ] **Step 5: Clean up the scratch clone**

```bash
rm -rf /tmp/dsio-freshclone
```

---

## Done when

- `git clone && uv sync --extra cpu && uv run pytest` is green from a fresh clone.
- No `copier.yml`, no `*.jinja`, no `lib/` anywhere in the tree.
- `uv sync --extra cu128` produces a CUDA-capable torch; `--extra cpu` produces the CPU wheel; the two are declared conflicting and a test asserts it.
- `lint-imports` reports **3** contracts kept — the three leakage walls, without the two-distribution direction.
- CI runs from the root on the CPU extra, with no `fork-drift` job.
- Every command in the README has been executed.
- Line count is within a few dozen of 7,708: this plan moves code, it does not cut it.

## Not in this plan

The Lightning-only reshape — the torch fixture, deleting `train/tabular.py` and the `forecast` extra, the torchmetrics swap, the `loss(pred, target)` contract change, dissolving `ssl/` into `model/` and `train/callbacks.py`, and the `data/ dataset/ model/ train/` restructure. That is Plan 2b, and it should be planned only after this one lands: the file paths it operates on are the ones this plan creates.
