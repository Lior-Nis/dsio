# 16. MLflow is the source of truth, and a run fails without it

Status: accepted (2026-08-20)
Supersedes: ADR 0002 ("The run ledger is authoritative; trackers are sinks").
Implemented: yes, by Plan 3b. `compose.yaml` (Postgres 16 + MLflow, per the backend-store
decision below) and `docker/mlflow/Dockerfile` (psycopg2 baked in, since the official
image ships without it) stand up the stack. `dsio.train.tracking.require_mlflow` is the
"a run fails without it" guard: it is called from both the CLI preflight (`check_torch`/
`check_ssl` in `train/torch_task.py`/`train/ssl_task.py`) and again at the top of the
runner itself (`run_torch`/`run_ssl_pretrain`), before the store, a module or a `Trainer`
exist. `build_mlflow_logger` (`dsio.train.tracking`) wires `MLFlowLogger(log_model=False)`
into both runners — model logging stays `dsio.artifacts.store.ModelRegistry`'s job, which
has the fail-closed policy MLflow's own model logging does not. `runs/` (`dsio.runs.
record`) is reduced to the provenance stamper; `artifacts/` (`dsio.artifacts.store`) is
reduced to digest-on-save, verify-on-load and fail-closed-on-mismatch over MLflow's Model
Registry. The nightly backup (`ops/backup-mlflow.sh`, `ops/mlflow-backup.service`, `ops/
mlflow-backup.timer`) is push-only (`rclone copy`, never `sync`) and captures both halves —
the Postgres dump and the `mlartifacts` volume — since Postgres holds artifact URIs and
restoring the database alone yields an index pointing at files that no longer exist.

## Context

ADR 0002 made the local run ledger authoritative and demoted MLflow and W&B to sinks that
mirror it. The reasoning was sound at the time: a directory of JSON survives anything that
can read a filesystem, and a run should be able to happen on a plane or inside a Kaggle
kernel.

Two things changed.

**The `runs/` directory mixes generated data with source.** A ledger living inside the repo
tree means every experiment writes into the thing git is versioning. Gitignoring it does not
fix the category error, it hides it.

**A component a run cannot write to is not a sink — it is a broken dependency.** The sink
model says a tracker failure should never kill a run. But if the tracker holds the only
queryable index of what was run, a silent mirroring failure means discovering at the end of a
week that nothing was recorded. ADR 0002 chose "log loudly and continue", which is the
correct answer *for a mirror* and the wrong answer for the place results actually live.

The objection that killed this idea in the original design was that MLflow cannot hold what
the ledger holds. That objection does not survive contact with the artifact API — every gap
is closed by logging the thing as an *artifact* rather than as a param:

| ADR 0002's gap | Resolution |
|---|---|
| Uncommitted working-tree diff | `log_artifact(diff.patch)` |
| `reproduce.sh` | `log_artifact(reproduce.sh)` |
| Full nested config (params cap at 250 chars) | `log_artifact(config.resolved.yaml)`; flat params for search only |
| Config-hash run identity | `set_tag("config_hash", …)`, searchable |

## Decision

MLflow is the source of truth for everything a run produces: metrics, params, tags, the
resolved config, the working-tree diff, the reproduce script, and the out-of-fold predictions.
**If MLflow is unavailable, the run fails.** There is no degraded mode, because a run whose
results are not recorded has not happened.

`runs/` stops being a ledger and becomes the provenance stamper: git revision, dirty-diff
capture, the reproduce script, and the seed record. These are the things MLflow does not
compute — the diff in particular is what keeps ADR 0003's "never block a run" promise honest,
since without it a dirty run is unreproducible.

`artifacts/` becomes a thin policy layer over MLflow's Model Registry rather than a parallel
one: compute a digest on save, verify it on load, **fail closed on mismatch** — which MLflow
does not do — plus `promotion_blockers()` for ADR 0003's clean-tree gate. MLflow's aliases are
*designed* to be moving targets, which is the opposite of a pinned reference, so pinning stays
ours.

The backend store is Postgres, not SQLite, because the fold-as-process decision (ADR 0017)
makes concurrent runs normal and the intended workload is several agents launching experiments
in parallel. SQLite is single-writer and the MLflow server's own workers contend on it.

## What building this surfaced

**"The run fails" is only meaningful because the probe is bounded.** MLflow's own client
defaults to a 120-second timeout and 7 retries with exponential backoff
(`mlflow.environment_variables`), which turns "the stack is down" into a multi-minute hang
before a run ever reports the failure it exists to report early. `require_mlflow`
(`dsio.train.tracking`) shrinks that budget to 5 seconds and zero retries for the probe
call only, then restores it immediately so a run that does reach MLflow still gets the
generous retry behaviour for real logging. Measured against a closed port, the probe now
fails in roughly 0.004 seconds — `test_an_unreachable_server_fails_in_bounded_time`
(`tests/train/test_tracking.py`) pins this at a generous `< 15s` rather than an exact
figure, to tolerate a loaded CI box without hiding a regression back toward the unbounded
default. Someone will eventually look at `_bounded_probe_timeout` and be tempted to
simplify it away; without it, this ADR's whole "fails without it" claim degrades back into
a several-minute hang.

**Lightning marks the MLflow run FINISHED before the runner's own failure guards run.**
`trainer.predict()` finalizes Lightning's `MLFlowLogger` run the instant it returns, which
is *before* `run_torch`/`run_ssl_pretrain` (`dsio.train.torch_task`/`dsio.train.ssl_task`)
run their own post-predict guards — ADR 0017's guards 1 and 4, carried over from the
deleted `cross_validate`: predictions that do not line up with the fold they came from,
and a fold whose metric fails to compute. Left alone, a run that failed one of those checks
would show as FINISHED
in MLflow: the exact "the record lies" failure decision 7 exists to prevent, just moved one
layer down. Both runners now wrap the code after provenance is stamped in `try`/`except BaseException`
and call `mlflow_logger.experiment.set_terminated(mlflow_logger.run_id, "FAILED")` on any
failure before re-raising, so a late failure is recorded as one.

## Consequences

Results stop living in the source tree, and there is exactly one place to look for them.

The accepted risk is concentration: the Postgres volume becomes a single point of failure for
all experimental history. A directory of JSON survives anything that can read a filesystem; a
database needs a working database. The mitigation is not optional and belongs on day one — a
nightly `pg_dump` **and** an artifact-store copy, pushed one-way to offsite storage. Both
halves must be captured together: Postgres holds artifact *URIs*, so restoring the database
alone yields an index pointing at files that no longer exist.

The archive is push-only and append-only. `rclone copy`, never `rclone sync` — `sync` makes
the destination match the source, which propagates a wiped volume into the backup at exactly
the moment the backup matters.

Fail-fast has a cost that will arrive with cloud training: a GPU box that cannot reach MLflow
cannot train at all. That is deferred deliberately rather than solved speculatively, and it is
the one part of this decision that will need revisiting rather than extending.
