# 7. The stage cache hashes code and cuts off early

Status: accepted (2026-08-17)
Implements: ADR 0004 items 1, 5, 12 and 13.

## Context

ADR 0004 originally specified explicit `version=` bumps with a source-drift *warning*, then
reversed on the grounds that warn-and-serve-stale is Hamilton's silent-staleness bug with a
log line attached. The survey of how other systems derive keys settled the rest:

| Tool | Hashes code? | Fails |
|---|---|---|
| Snakemake | yes, `code` is a default rerun trigger | safe |
| Hamilton | yes, source ignoring docstrings/comments | **unsafe** — misses nested calls |
| joblib | yes; warns then clears | safe |
| HF `datasets` | yes; random fingerprint when unhashable | safe (recomputes) |
| Flyte | no — human bumps `cache_version` | unsafe if forgotten |
| DVC | only if you list the script in `deps` | unsafe if forgotten |
| Kedro | no caching at all | n/a |

## Decision

Key derivation is a **composable list of policies**, following Flyte's `CachePolicy` shape
rather than one fixed rule. The default is `CodePolicy + ConfigPolicy + EnvPolicy`.

**Code is in the key by default, so drift cannot occur.** A warning is unnecessary because
different code is a different key. Source is AST-normalised — comments, docstrings,
formatting and the function's own name are discarded — so reformatting or renaming does not
throw away a forty-minute stage.

**Upstream contributes its output digest, not its key.** This is early cutoff, and it falls
out of the key definition rather than needing separate machinery: rewrite a parsing stage
and it recomputes, but if the bytes are unchanged every downstream key is unchanged too.
Nix's content-addressed derivations and Bazel's action outputs work the same way. Early
cutoff is what makes aggressive code hashing affordable.

**`ExplicitVersionPolicy` is available per stage** for logic you want pinned by hand — the
Flyte/DVC posture, chosen deliberately rather than by default.

## What is not in the key

Anything that changes only *how fast* you got the answer: `num_workers`, `device`, `gpus`,
`log_level`, `pin_memory`, `prefetch_factor` and friends, stripped recursively. Anything
that changes the output bytes is in. Flyte exposes the same idea as `ignored_inputs`.

Getting this wrong in one direction thrashes the cache on every machine; in the other it
silently reuses artifacts across incompatible settings.

## The limit, stated rather than discovered

`CodePolicy` hashes the stage function's own body. **A helper it calls is invisible**, so
editing the helper yields a hit on stale logic. Hamilton documents the identical gap; most
systems with this feature do not mention it. Two remedies, both explicit: pass helpers as
`extra_code`, or bump `version`. `test_helper_edits_are_invisible_without_extra_code`
asserts both halves so the limit cannot regress into a surprise.

## Integrity

`CacheEntry.load()` re-hashes the artifact and fails closed on mismatch. Metadata present
with the artifact missing — an interrupted write — is also a hard failure rather than a
miss. Existence is not integrity: Kedro's `--only-missing-outputs` treats a truncated file
as valid forever.

## Two bugs this found

Both were caught by tests rather than review, and both would have been near-invisible.

The fallback for un-sourceable callables used `repr(fn)`, which embeds a memory address —
so a callable *object* got a different key in every process, and the cache would simply
never have hit. It now falls back to module and qualname, and for a callable object hashes
its class source.

The AST dump included the function's own name, so renaming a stage function silently
discarded its cached results.
