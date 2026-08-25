# Plan 3b — MLflow is the source of truth

Implements locked decisions 7 and 8 of `docs/superpowers/specs/2026-08-20-dsio-lean-design.md`,
plus the Backup section. Plan 3a deliberately excluded these so it needed no running
infrastructure. **This plan introduces the first one**, which is the fact that shapes every
task below.

## What is true today

`runs/record.py` (319 lines) is a file-based ledger: `RunLedger.start/load/list_runs/iter_runs`
allocates a directory per run, and `Run` writes `run.json`, `metrics.jsonl`, `config.resolved.yaml`
and `reproduce.sh` into it. `runs/provenance.py` (154) stamps git rev and captures the dirty
diff. `runs/seeding.py` (86) is the seed recorder and worker-seeding pair. `artifacts/store.py`
(284) is a local model registry with digest-on-save and fail-closed load.

## What decisions 7 and 8 require

**MLflow becomes the source of truth, and a run fails without it.** This supersedes ADR 0002
("the ledger is authoritative; trackers are sinks") on the grounds that *a component a run
cannot write to is not a sink — it is a broken dependency*.

`runs/` stops being a ledger and becomes **the provenance stamper**: git rev, dirty-diff
capture, the reproduce script, the seed recorder. `artifacts/` becomes **a thin policy layer
over MLflow's Model Registry**: digest on save, verify on load, fail closed on mismatch —
which MLflow does not do — plus `promotion_blockers` for ADR 0003's clean-tree gate.

Everything ADR 0002 said MLflow could not hold is logged as an **artifact** rather than a
param: the dirty diff, `reproduce.sh`, the full nested config. Flat params exist for search
only; `config_hash` is a tag.

## The constraint that shapes this plan: the suite must not need Docker

Today `uv run --extra cpu pytest` passes on a laptop with nothing running. **That must stay
true**, or every future task in this repository acquires an infrastructure dependency and the
fresh-clone proof stops meaning anything.

So each task states which of three kinds its tests are:

- **Unit** — no MLflow at all. The default. Test the policy, not the transport.
- **Fake-backed** — MLflow's own `file:` tracking URI, which needs no server and no Docker.
  This is the honest middle: real MLflow client code, no infrastructure.
- **Live** — needs `docker compose up`. Marked, and **skipped by default**. A live test that
  silently skips in CI and is never run anywhere is worse than no test, so Task 8's
  fresh-clone proof must run them explicitly and report their output.

**Do not let "it needs a server" become a reason a behaviour is untested.** Fail-fast in
particular is testable with no server at all: point at a dead URI and assert the run fails
before training starts.

## Tasks

### Task 1 — the compose stack
`compose.yaml` (Postgres 16 + MLflow) and `docker/mlflow/Dockerfile`, per decision 8. Postgres
rather than SQLite because decision 6 made concurrent runs normal and SQLite is single-writer.
Named volumes, so artifacts never land in the source tree. `psycopg2` is baked into the image
because the official one ships without it (mlflow#9513).
**Verify:** `docker compose up -d` brings both up healthy; the MLflow UI answers on 5000. No
Python changes. Report the actual `docker compose ps` output.

### Task 2 — fail-fast, and the logger
`MLFlowLogger(tracking_uri=...)` wired into both runners, with `log_model=False` (the spec's
default, and deliberate). **A run must fail immediately when MLflow is unreachable — before
building a store, a module or a trainer**, not after training finishes and the log call fails.
**Verify (unit, no server):** point at a dead URI; assert the failure happens before any
expensive work, with a message naming the tracking URI and how to start it. This is the
"broken dependency" claim in decision 7 — prove it fires early, not just that it fires.

### Task 3 — `runs/` becomes the provenance stamper
Delete `RunLedger` and the per-run directory allocation. Keep provenance, the dirty-diff
capture (ADR 0003's reproducibility guarantee), the reproduce script and the seed recorder;
they now write **into MLflow as artifacts**. `Run`'s remaining job is whatever the runners
genuinely need — decide what that is and say so.
**Guard to rehome, named because Plan 3a taught this lesson twice:** the ledger currently owns
run-id allocation and `config_hash` identity. Whatever replaces it must still make two runs of
the same config distinguishable and a run's config recoverable. Name where each lands.
**Verify (fake-backed):** a run logs its diff, config and reproduce script as artifacts and
they read back byte-identical.

### Task 4 — `artifacts/` becomes a policy layer
Keep digest-on-save, verify-on-load and **fail-closed on mismatch** — MLflow does not do this,
which is the whole reason this layer survives at all. Keep `promotion_blockers`. Delete the
local storage and listing that MLflow's registry replaces.
**Verify (fake-backed):** spec Verification item 6 — corrupt a stored model, confirm load
raises on digest mismatch rather than returning weights. This test exists today and must keep
passing against the new backing.

### Task 5 — backup, push-only
The nightly script and systemd timer from the spec's Backup section. **Both halves** — the
Postgres dump *and* the `mlartifacts` volume; restoring the database alone yields an index
pointing at files that no longer exist.
**`rclone copy`, never `rclone sync`.** `sync` propagates a wiped volume into the archive at
exactly the moment the backup matters. Scope the remote to `drive.file` so the one-way
property is enforced by the credential rather than by discipline.
**Verify:** dry-run the script against a local rclone remote; confirm a file deleted locally
still exists in the archive afterwards. That is the property, and it is testable without
Google Drive.

### Task 6 — ADRs
ADR 0002 is **superseded** by decision 7 — mark it, do not rewrite its body. ADR 0016 says
`Implemented: no. This is Plan 3.`; it is now implemented. ADR 0003's clean-tree gate must
still be described accurately against wherever `promotion_blockers` ended up.
**Every path, symbol and filename must be verified to exist.** This repository has shipped
documentation pointing at deleted code four times; the sweep must cover `docs/` as well as
`src/`, which is the omission that caused the fourth.

### Task 7 — fresh-clone proof
Clone, install, full check, then **spec Verification items 2, 4 and 8** end to end:
- `docker compose up -d`, `dsio run <preset>` records a run, and **killing MLflow makes the
  next run fail immediately rather than after training**;
- a shell loop over 5 folds produces 5 runs in one experiment, grouped natively;
- the `reproduce.sh` logged as an MLflow artifact reruns to identical metrics, **including
  from a run made with a dirty working tree**.
Run the live-marked tests explicitly and report their output. Report the code-line count via
`scripts/count_code.py` against **4,658**.

## Global constraints

Python >=3.12; torch/lightning only in the `cpu`/`gpu` extras. Every uv command needs
`--extra cpu`.
**VERIFY:** `uv run --extra cpu pytest && uv run --extra cpu ruff check . && uv run --extra cpu mypy && uv run --extra cpu lint-imports`
ruff line-length 100, rules `E,F,W,I,UP,B,BLE,SIM,RUF`; no bare `except`, a typed re-raise is
fine. mypy `disallow_untyped_defs`. Baseline at plan start: **436 tests**, **3 contracts**,
**4,658 code lines**.

**This repository has produced twenty-two checks that could not fail.** Every guard added or
touched must be verified by breaking what it guards, observing the failure, restoring, and
confirming `git diff` is clean. **Clear `__pycache__` after restoring** — a same-size mutation
restored within one mtime tick leaves stale bytecode, and the dangerous direction is mutated
code running as restored, which makes a guard-bites proof pass while proving nothing.

**A refusal-only test suite proves half a guard.** For every check that rejects something, add
the case that must still be *accepted*. Plan 3a shipped a guard that would have refused every
legitimate cross-validation, and no refusal test could have caught it.
