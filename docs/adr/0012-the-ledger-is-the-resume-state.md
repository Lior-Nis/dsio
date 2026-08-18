# 12. The run ledger is the resume state, and a cell's identity is its config hash

Status: accepted (2026-08-18)
Implements: the plan's Phase 6, and its "matrix resumability" verification test.

## Context

FORGE orchestrated its sweeps in bash: 70+ shell scripts, including `probe_loop.sh` — a
polling daemon that watched a checkpoint directory every fifteen minutes and tracked what it
had already done in a `.probed_checkpoints` file. That is a job scheduler written in bash,
and dsio's plan named replacing it as Phase 6.

## Decisions

**A cell's identity is the sha256 of its resolved config**, not its position in the product.
Three consequences:

1. re-running a completed cell is a no-op;
2. adding an axis value does not renumber the cells that already ran;
3. two cells that resolve to the same config *are* the same job, however they were reached
   — a duplicate is impossible rather than merely unlikely.

Identity therefore does not depend on the order the axes were given, which matters because
reordering a command line must not silently re-run a finished sweep.

**The resume state is the run ledger, not a sidecar.** A cell is done when the ledger holds
a *completed* run whose config hash matches. Nothing else is written and nothing else is
consulted.

This is a deliberate correction of the `.probed_checkpoints` shape, which is the standard
solution and is wrong in a specific way: a sidecar records an *intention* and can disagree
with what happened. It says "done" for a run that crashed after the line was appended, or
omits one that finished before the write. Deriving resume state from the ledger means it
cannot drift, because the ledger is the same record the result is later read from. A crashed
run is simply absent from the completed set, so it is retried — no reconciliation logic
exists because there is nothing to reconcile.

`test_killing_a_sweep_mid_flight_resumes_exactly_where_it_stopped` is the plan's headline
check for this phase: a sweep dies on cell 2 of 4, and the second invocation runs one cell
and skips three.

**An axis is written like an override.** `task.lr=1e-3,3e-4`, and a matrix is the cross
product. The syntax someone already knows for one run extends to two hundred without a
second grammar. `glob:` axes are resolved and sorted **at parse time**, because a matrix
whose size depends on when it was expanded cannot be resumed — the second invocation would
be a different matrix.

**Every cell is resolved before any cell runs.** Two hundred cells that each fail after
loading a corpus is two hundred wasted loads; resolving first turns a typo in one axis into
an immediate parse error, and produces the identities the resume logic needs anyway.

**A failed cell does not stop the sweep, and is not swallowed.** It is recorded as a failed
run with its traceback, the sweep continues, and the summary reports every failure with a
non-zero exit. One bad cell out of two hundred should cost one cell — but a sweep that
quietly reports success while a fifth of it failed is worse than one that stops.
`--fail-fast` gives the other behaviour. Failed cells are retried on the next invocation,
because a failed cell is not a finished one and the usual reason it failed is a bug that has
since been fixed.

**A sweep above `--max-cells` is refused before it is materialised.** A sweep that would
take a week should be refusable before it starts, not after.

## Search

Optuna is driven directly. The `hydra-optuna-sweeper` plugin is unusable regardless of the
Hydra decision: stable 1.2.0 is from 2022 and pins `optuna<3.0`.

**A trial is an ordinary Run** — same ledger, same config hash, same artifact contract — so
a searched result and a hand-run one are compared by the same commands, and a trial that
turns out to matter is promoted like any other run.

**The distribution is named explicitly**, never inferred from the bounds. A learning rate
searched uniformly over [1e-5, 1e-3] spends 90% of its trials above 1e-4, which is a silent
and expensive mistake that nothing in the numbers reveals.

**A failed trial is pruned, not fatal.** A search whose fifth trial hits an unusable
configuration should record it and carry on; one that dies there has wasted the four before
it. A trial that produced no metric at all is a failure rather than a silent zero — scoring
it would optimise against a default.

**Search shares the matrix's resume mechanism.** A trial whose config hash already has a
completed run reads that run's recorded metric instead of retraining an identical
configuration. So a search after a sweep costs only the points the sweep did not cover, a
sampler that revisits a point pays nothing, and a re-invoked search does not repeat finished
work. This falls out of content-addressing rather than being separate machinery — it is the
same identity doing the same job. Optuna's sqlite storage is the complementary half: the
study remembers which points were tried, the ledger remembers which configs completed.

## What building this surfaced

`reuse_completed=False` still populated the reuse cache as the search ran, so within-search
reuse happened regardless and the flag was a lie. Caught by the test that asserted the flag
does what its name says — worth noting because the "off" path of a feature flag is the one
that habitually goes untested.

## Consequences

Optuna joins the dev group while remaining an optional extra, for the third time with the
same reasoning: a path CI cannot execute is a path whose correctness is a claim.

Label-budget curves from ADR 0011 are now expressible as a matrix — a budget axis crossed
with an arm axis — though no runner drives one end to end yet.

What is deliberately not built: no parallel execution, no distributed workers, no queue.
Cells run sequentially. Content-addressed identity is exactly what makes parallelism safe to
add later (two workers cannot collide on a cell, because the ledger's `mkdir` claim decides
the winner), but there is no cluster to add it for.

The ranking this produces makes Phase 7 concrete: on a small sweep, the top four runs sat
within 0.0003 of each other against a fold spread of ~0.006. `dsio eval rank` already warns
that a ranking is an ordering rather than a claim; selection under multiplicity is what turns
one into the other.
