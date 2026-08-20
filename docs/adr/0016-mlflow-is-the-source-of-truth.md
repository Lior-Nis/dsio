# 16. MLflow is the source of truth, and a run fails without it

Status: accepted (2026-08-20)
Supersedes: ADR 0002 ("The run ledger is authoritative; trackers are sinks").
Implemented: no. This is Plan 3.

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
