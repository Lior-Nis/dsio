# One Framework — Implementation Plan (2b of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Lightning the only training path, move the paradigm into the dataset so there is exactly one `LightningModule` and no paradigm-named directories, and land the `data/ dataset/ model/ train/` layout.

**Architecture:** Unlike Plan 2a, this plan changes what the code *means*. Three moves, in dependency order. First the scaffolding that lets non-Lightning code go (a torch fixture, then `train/tabular.py` and the dead extras). Then the semantic change: masking moves out of the model's augmentor slots and into the dataset, which makes `SslModule`'s step override unnecessary — and with `encoder_state` extracted as a free function and predict-mode chosen by config, `SslModule` disappears entirely. Only then the renames, because renaming before the semantics settle makes every diff unreviewable.

**Tech Stack:** Python 3.12, torch + Lightning, torchmetrics, pydantic 2, numpy 2, uv, pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-08-20-dsio-lean-design.md` (decisions 1, 4 and 5), recorded as `docs/adr/0015-lightning-is-the-only-training-path.md`.

## Global Constraints

- Python `>=3.12`. `torch` and `lightning` live only in the `cpu` and `gpu` extras — **never** in the dev group. That is deliberate: a bare `uv sync` installs no torch and fails loudly rather than silently pulling gigabytes.
- **VERIFY**, from the repository root, and every command needs the extra:
  ```bash
  uv run --extra cpu pytest && uv run --extra cpu ruff check . && uv run --extra cpu mypy && uv run --extra cpu lint-imports
  ```
- `ruff` line-length 100, rules `E,F,W,I,UP,B,BLE,SIM,RUF`. `BLE` is load-bearing — never a bare `except`; a typed re-raise (`except X as e: raise Y() from e`) is fine and used already.
- `mypy` runs with `disallow_untyped_defs = true`. Everything you write needs annotations.
- Baseline: **371 tests**, **3 import contracts**, ruff and mypy clean.
- Every task ends with VERIFY passing and a commit. Never carry a red suite into the next task.
- **Do not rename anything until Task 7.** Renaming while semantics are still moving produces a diff nobody can review, and this repository has already proven that a reviewable diff is what catches its defects.

### The failure mode this repository keeps producing

Seven times across Plans 1 and 2a, a check appeared to pass while proving nothing: a test satisfied by an `autouse` fixture; a registry populated by an unrelated module's import; a ruff exclusion bypassed by how ruff was invoked; a "torch-free" proof run against an already-populated venv; a green suite hiding an emptied component registry; `lint-imports` exiting 0 with zero contracts declared.

**Therefore, in this plan: any test you add or rely on must be verified by breaking the thing it guards and observing the failure, then restoring and confirming `git diff` is clean.** A step that says "confirm it passes" is not verification. Every task below that adds a guard says this explicitly; do it even where it does not.

---

### Task 1: Baseline and branch

**Files:** none

- [ ] **Step 1: Confirm green and record the numbers**

```bash
uv sync --extra cpu
uv run --extra cpu pytest 2>&1 | tail -2
uv run --extra cpu ruff check . && uv run --extra cpu mypy && uv run --extra cpu lint-imports
find src -name '*.py' | xargs wc -l | tail -1
```
Expected: `371 passed`, all gates clean, 3 contracts, ~7800 lines. Write the numbers down. If anything fails on a clean checkout, stop and report.

- [ ] **Step 2: Record what depends on what you are about to delete**

```bash
grep -rln "TabularTask\|train.tabular" src/ tests/
grep -rln "sklearn\|scikit" src/ tests/
grep -rln "SslModule\|SslMethod\|encoder_state\|dsio.ssl" src/ tests/
```
Write the three lists down. Tasks 2, 3 and 6 each check their own list against yours; a file appearing that you did not record means the tree moved under you.

- [ ] **Step 3: Branch**

```bash
git checkout -b lean/02b-one-framework origin/main
```

---

### Task 2: A torch fixture, so the tabular one can go

`tests/conftest.py`'s `config` fixture builds a `RunConfig` from `TabularTask`. Every test that takes `config` therefore depends on the runner this plan deletes. The fixture must move to torch **before** Task 3, or the suite dies wholesale.

**Files:**
- Modify: `tests/conftest.py`
- Test: the existing suite is the test — every consumer of `config` must still pass

**Interfaces:**
- Consumes: `dsio.train.torch_task.TorchTask`, `dsio.config.schema.RunConfig`
- Produces: a `config` fixture whose task is a `TorchTask`

- [ ] **Step 1: Find every consumer**

```bash
grep -rn "def test_.*\bconfig\b\|(config" tests/ | grep -v conftest | head -30
```
List them. These are the tests whose meaning could change.

- [ ] **Step 2: Read `TorchTask`'s required fields**

```bash
grep -n "class TorchTask" -A 40 src/dsio/train/torch_task.py
```
It needs more than `TabularTask` did — a store, a window spec, components. `tests/train/test_torch_runner.py` already constructs one; read how it does it and reuse that shape rather than inventing one.

- [ ] **Step 3: Rewrite the fixture**

Replace the `TabularTask` body with a `TorchTask` built the same way `test_torch_runner.py` builds its own, using the smallest corpus that trains in under a second. Keep the fixture's *name* and return type identical so no consumer changes.

If a consumer genuinely depends on tabular-specific behaviour, **do not** weaken it to pass — report it, because that test is telling you something about what Task 3 is deleting.

- [ ] **Step 4: VERIFY**

Expected: 371 passed. A changed count means a consumer's meaning changed — investigate and report rather than accepting it.

- [ ] **Step 5: Clear the residual carried in from Plan 2a**

`docs/superpowers/specs/2026-08-20-dsio-lean-design.md`'s `## Verification` section still
contains a bare `uv sync && uv run pytest`. That command errors now — an accelerator extra is
mandatory. Plan 2a fixed the identical sentence in §2 but the finding was scoped to that
section, so this one survived. Change it to `uv sync --extra cpu && uv run pytest`. This rides
in Task 2's commit because Task 1 creates none.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Build the shared config fixture from a torch task

Every test taking `config` depended on the tabular runner. Moving the
fixture first is what makes deleting that runner a deletion rather than a
suite-wide rewrite."
```

---

### Task 3: Delete the tabular runner and the dead extras

Lightning is the only first-class training path (ADR 0015). A tabular baseline remains reachable as a plain script; it does not need a registered runner inside the package.

**Files:**
- Delete: `src/dsio/train/tabular.py`, and the tabular tests that cover only it
- Modify: `src/dsio/train/__init__.py` (`_BUILTIN_RUNNER_MODULES`), `src/dsio/presets.py`, `pyproject.toml`
- Test: `tests/train/test_runner_registry.py`, `tests/train/test_runner_bootstrap.py`

**Interfaces:**
- Consumes: nothing
- Produces: `_BUILTIN_RUNNER_MODULES` without `dsio.train.tabular`

- [ ] **Step 1: Repoint the built-in preset first**

`src/dsio/presets.py`'s `spine_baseline` imports `TabularTask` inside its body. Rewrite it to return a `TorchTask` — reuse Task 2's shape. **Keep the function-local import and its comment**: enumeration must not pay for importing a task, and bare `dsio run` must still list presets without importing torch. Prove that still holds:

```bash
uv sync                     # no extra: torch absent
uv run dsio run             # must list presets
uv pip show torch           # must report not found
uv sync --extra cpu         # restore
```

- [ ] **Step 2: Delete the runner and its tests**

```bash
git rm src/dsio/train/tabular.py
```
Then delete tests that cover only the tabular runner. **A test that covers something surviving must be repointed, not deleted** — check each against Task 1's list before removing it.

- [ ] **Step 3: Remove it from the registry**

Delete `"dsio.train.tabular"` from `_BUILTIN_RUNNER_MODULES`. **Then prove the registry guards still work**: empty the tuple entirely, confirm `tests/train/test_runner_registry.py` AND `tests/train/test_runner_bootstrap.py` both fail, restore, confirm both pass, and check `git diff` is clean. Those two guards exist because this exact registry silently rotted once already.

- [ ] **Step 4: Drop the dead extras**

In `pyproject.toml` remove the `forecast` extra (`statsforecast`, `mlforecast`) and the `legacy` extra (`zarr`) — nothing in `src/` imports any of them. For `tabular` (`scikit-learn`, `xgboost`, `polars`, `pyarrow`): check what still imports scikit-learn (`tests/eval/test_metrics.py` pins metrics against it, and `src/dsio/ssl/probe.py` may use it). **Keep scikit-learn wherever it is genuinely used and say where; drop the rest.** Run `uv lock` and report what left the lockfile.

- [ ] **Step 5: VERIFY and commit**

```bash
uv sync --extra cpu
uv run --extra cpu pytest && uv run --extra cpu ruff check . && uv run --extra cpu mypy && uv run --extra cpu lint-imports
git add -A
git commit -m "Delete the tabular runner and the extras nothing imports

Lightning is the only first-class training path. A tabular baseline is a
plain script now, not a registered runner."
```

---

### Task 4: Metrics through torchmetrics

`src/dsio/eval/metrics.py` implements fourteen metrics in numpy because scikit-learn used to be an optional extra. Torch is a hard dependency now, so torchmetrics is always present — and it brings distributed reduction, which a hand-rolled version gets wrong the first time it sees two GPUs.

**Files:**
- Modify: `src/dsio/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: `torchmetrics.functional`
- Produces: `METRICS` registry with the same names and the same `(y_true, y_pred, y_score) -> float` signature

- [ ] **Step 1: Keep the registry, replace the bodies**

The registry, the names, and the call signature all stay — only the implementations change. Callers pass numpy arrays; convert with `torch.from_numpy` inside each adapter. Aim for ~60 lines total.

- [ ] **Step 2: Keep the correctness pin**

`tests/eval/test_metrics.py` pins `average_precision` against scikit-learn to 1e-12, because the trapezoid interpolation in `auc(recall, precision)` is optimistically biased and this is the metric most often computed wrong. **Keep that test and keep it at 1e-12.** If torchmetrics disagrees with scikit-learn beyond that tolerance, stop and report the numbers — do not loosen the tolerance to make it pass. A loosened tolerance is a silenced finding.

- [ ] **Step 3: VERIFY and commit**

Expected: the test count may fall if hand-rolled edge-case tests covered implementation details rather than behaviour. Report the delta and name each deleted test.

```bash
git add -A
git commit -m "Metrics through torchmetrics

Fourteen numpy implementations existed because sklearn was optional. Torch
is a hard dependency now, and torchmetrics reduces correctly across devices."
```

---

### Task 5: The dataset owns the paradigm

This is the semantic change the plan exists for. Masking moves out of the model's stochastic slots and into the dataset, so `(x_masked, x_orig)` arrives as `(x, target)` and the existing `loss(pred, target)` contract does the rest.

**Files:**
- Modify: `src/dsio/nn/module.py` (delete the augmentor slots), `src/dsio/nn/registry.py` (delete `AUGMENTORS`), `src/dsio/nn/data.py` (dataset applies masking)
- Test: `tests/nn/test_module.py`, `tests/nn/test_data.py`

**Interfaces:**
- Consumes: `dsio.ssl.masking` (moves in Task 6; import it where it lives today)
- Produces: a dataset that can emit `(x, target)` for a pretext objective; `DsioModule` with no stochastic slots

- [ ] **Step 1: Understand what the guard was for before removing it**

`DsioModule` skips its two stochastic slots unless `self.training`, and its docstring explains why: *"a chain that applies whatever is configured whenever it is called will augment during validation, which makes the metric noisy **and** irreproducible while looking entirely normal."*

That property must survive. It survives **structurally**: the DataModule supplies a different dataset per stage, so a masking dataset is used for training and a plain one for validation. **Write the test that proves it before you delete the guard** — assert that a validation batch is unmasked. Then delete the guard and confirm the test still passes. If it does not, stop: the property did not survive and the design is wrong.

- [ ] **Step 2: Move masking into the dataset**

The dataset gains an optional pretext transform. When present, `__getitem__` returns the masked signal as `x` and the original as the target key; when absent, it returns `(x, label)` as today. Keep `row` in the batch either way — predictions are aligned to folds by row identity, not DataLoader ordering.

- [ ] **Step 3: Delete `AUGMENTORS` and the slots**

Remove the registry, the two slots, and the `self.training` branch. Update every component that registered an augmentor.

- [ ] **Step 4: Prove the whole chain still trains**

Run `tests/train/test_torch_runner.py` and `tests/train/test_ssl_runner.py` directly. Both must pass. If SSL cannot yet train this way, **stop and report** — Task 6 depends on this working.

- [ ] **Step 5: VERIFY and commit**

```bash
git add -A
git commit -m "The dataset owns the paradigm

Masking moves from a model slot to the dataset, so (x_masked, x_orig)
arrives as (x, target) and the loss contract needs no special case. The
never-augment-during-validation property now holds by construction: the
DataModule gives a different dataset per stage."
```

---

### Task 6: Dissolve `ssl/`

A directory named after a paradigm is a category error in a tree organised by technical kind, and it is how a junk drawer regrows. `ssl/` is 870 lines, of which only the module was ever structurally SSL — and after Task 5, not even that.

**Files:**
- Delete: `src/dsio/ssl/` entirely
- Create: `src/dsio/train/callbacks.py`
- Modify: `src/dsio/nn/components.py` (or new `heads.py`/`losses.py`), `src/dsio/nn/module.py`, `src/dsio/train/ssl_task.py`
- Test: `tests/ssl/*` — repoint, do not delete wholesale

**Interfaces:**
- Produces: `dsio.nn.export_encoder(module) -> dict[str, torch.Tensor]`; a `predict` mode on the task config

- [ ] **Step 1: Move the components**

`ssl/methods.py` (214) is head+loss pairs → register them as heads and losses. `ssl/masking.py` (157) is a transform → move to where Task 5 put the dataset's pretext transform. Neither is SSL-specific machinery; both are components.

- [ ] **Step 2: `encoder_state` becomes a free function**

It is an *export* concern, not a method. Move it to `src/dsio/nn/` as `export_encoder(module) -> dict[str, torch.Tensor]`, taking any module. **Keep its exclusion of the objective head and keep the reason in the docstring** — *"a decoder trained to reconstruct masked spans has no meaning outside the pretext task, and shipping it invites someone to load it as though it were part of the model."* That sentence is why the function exists.

- [ ] **Step 3: `predict_step` becomes a config choice**

`SslModule.predict_step` returns embeddings rather than predictions. Add a field to the task config — `predict: Literal["prediction", "embedding"] = "prediction"` — and branch on it in `DsioModule.predict_step`. One class, no subclass, and the choice is visible in the recorded config rather than implied by a type.

- [ ] **Step 4: Delete `SslModule` and the directory**

With the step override gone (Task 5), `export_encoder` extracted, and predict mode configured, nothing remains. `git rm -r src/dsio/ssl`.

`ssl/probe.py` (194) → `src/dsio/train/callbacks.py`. A linear probe on frozen features is a general representation-quality tool, not an SSL one; say so in its docstring. `ssl/budget.py` (191) is a sweep over one config axis — **cut it**, and record in the commit message that it is a per-project experiment protocol.

- [ ] **Step 5: Repoint the tests, do not delete them**

`tests/ssl/` covers masking, methods, and the probe — all of which survive under new homes. Move those tests to match. Delete only what covered `budget.py` and the deleted module. **Say which you deleted and why each covered only deleted behaviour.**

- [ ] **Step 6: Prove the export still excludes the head**

Write a test asserting `export_encoder` returns no key from the objective's head. **Then break it**: temporarily include the head, confirm the test fails, restore, confirm `git diff` is clean. This is the property most likely to rot silently — a wrong export produces a model that loads and is wrong.

- [ ] **Step 7: VERIFY and commit**

```bash
git add -A
git commit -m "Dissolve ssl/: a paradigm is not a directory

Methods were head+loss pairs, masking was a transform, the probe is a
general representation tool, and the module's last reason to exist went
when the dataset took over the paradigm. encoder_state is an export
concern, so it is a function now; predict mode is config, not a subclass."
```

---

### Task 7: The layout

Only now, with semantics settled, do things move.

**Files:**
- Move: `src/dsio/nn/` → `src/dsio/model/`; `src/dsio/nn/data.py` → `src/dsio/dataset/`
- Modify: every importer, `pyproject.toml` import contracts, `README.md`

- [ ] **Step 1: Move, then repair imports package by package**

```bash
git mv src/dsio/nn src/dsio/model
mkdir -p src/dsio/dataset && git mv src/dsio/model/data.py src/dsio/dataset/dataset.py
```
Then fix imports one package at a time, running `uv run --extra cpu pytest 2>&1 | grep -E "ImportError|ModuleNotFoundError"` after each, so a failure names the package you just touched.

- [ ] **Step 2: Update the import contracts**

`pyproject.toml`'s contracts name `dsio.nn` and friends. Update the names, and update `tests/test_import_contracts.py` to match — that test asserts the three contract names are present, so it will fail loudly if you rename a contract and forget it. **Confirm it does**: rename one contract without updating the test, observe the failure, restore.

- [ ] **Step 3: Fix the README**

`README.md` says "a backbone in `nn/`". After this task it is `model/`. This is the sentence a reader follows first.

- [ ] **Step 4: VERIFY and commit**

Expected: 3 contracts kept, all gates green.

```bash
git add -A
git commit -m "data/ dataset/ model/ train/

Renames only. The semantics moved in Tasks 5 and 6, deliberately, so this
diff is reviewable as the mechanical change it is."
```

---

### Task 8: Merge `WindowView` into the Dataset

`data/views.py`'s `WindowView` and the torch `Dataset` take the same constructor arguments, carry near-identical store-name guards, and both read windows on demand. The spec lists `WindowView` as cut; Plan 2a kept it deliberately for this task.

**Files:**
- Modify: `src/dsio/data/views.py`, `src/dsio/dataset/dataset.py`
- Test: `tests/data/test_store.py` (its only references)

- [ ] **Step 1: Confirm the overlap before removing anything**

Read both. Report what each does that the other does not. **If they differ in a way that matters, say so and stop** — the spec's claim that one duplicates the other is a claim, and this task is where it gets tested.

- [ ] **Step 2: Merge and repoint**

Keep the store-name guard (it catches an index built against a different store) and its message. Repoint `tests/data/test_store.py`.

- [ ] **Step 3: VERIFY and commit**

---

### Task 9: Prove it from a fresh clone

**Files:** none. No commit.

- [ ] **Step 1: Clone the branch to a scratch directory and run everything**

```bash
git clone --branch lean/02b-one-framework /home/liornisimov/Projects/dsio /tmp/dsio-2b
cd /tmp/dsio-2b && uv sync --extra cpu
cd /tmp/dsio-2b && uv run --extra cpu pytest 2>&1 | tail -2
cd /tmp/dsio-2b && uv run --extra cpu ruff check . && uv run --extra cpu mypy && uv run --extra cpu lint-imports
```

- [ ] **Step 2: Confirm the layout and that no paradigm directory survives**

```bash
ls /tmp/dsio-2b/src/dsio
```
Expect `data/ dataset/ model/ train/` plus `config/ runs/ artifacts/ eval/ splits/ cli/ contracts.py presets.py`. **No `nn/`, no `ssl/`.**

- [ ] **Step 3: Run both paradigms end to end**

```bash
cd /tmp/dsio-2b && uv run --extra cpu dsio run
cd /tmp/dsio-2b && uv run --extra cpu dsio run spine_baseline
```
Then run an SSL preset if one is registered, or construct one, and confirm pretraining produces an encoder export with no objective-head keys.

- [ ] **Step 4: Measure and report honestly**

```bash
cd /tmp/dsio-2b && find src -name '*.py' | xargs wc -l | tail -1
```
Report against Task 1's number and against the spec's ~6,200 target for this stage. **If it is far off, say so plainly rather than rounding toward the prediction.**

- [ ] **Step 5: Clean up**

```bash
rm -rf /tmp/dsio-2b
```

---

## Done when

- No `src/dsio/nn/`, no `src/dsio/ssl/`, no `src/dsio/train/tabular.py`.
- `src/dsio/` holds `data/ dataset/ model/ train/` alongside the unchanged spine packages.
- Exactly one `LightningModule`. No paradigm subclass.
- Masking lives in the dataset; a validation batch is provably unmasked.
- `export_encoder` is a free function that provably excludes the objective head.
- Metrics come from torchmetrics, with the `average_precision` pin still at 1e-12 against scikit-learn.
- 3 import contracts, renamed to match the new packages, with the contract-names test updated.
- A fresh clone installs, passes, and trains both paradigms.

## Not in this plan

Fold-as-process (deleting `cross_validate`), MLflow as the source of truth, the Postgres compose stack and the backup job. Those are Plan 3, and they change what a *run* means rather than what a *model* is.

**Carried in from Plan 2a:** the spec's `## Verification` section still contains a bare `uv sync && uv run pytest`, which errors now that an accelerator extra is mandatory. It is cleared in Task 2, Step 5 — the first step in this plan that produces a commit.
