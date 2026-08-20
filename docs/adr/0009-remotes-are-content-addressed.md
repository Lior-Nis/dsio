# 9. The committed manifest is the remote index, and objects are addressed by content

Status: accepted (2026-08-18)
Suspended (2026-08-20): the implementation is removed with `data/remote.py` because cloud
training is deferred. The decision is not overturned — revisit it on its merits when the
cloud design has real requirements to answer to.
Implements: the plan's Phase 2 remote, and the fresh-clone verification test.

## Context

The plan names one headline property for the data layer:

> clone into a clean directory, `uv sync --locked`, `dsio data pull`,
> `dsio reproduce <run_id>` — must work with no manual steps.

This is the test FORGE failed. Its classification checkpoints silently reloaded an MAE
encoder from a hardcoded path, so a fresh clone could not reproduce anything; the path
existed on exactly one machine.

DVC was rejected in ADR 0004 (maintenance mode under lakeFS since the 2025-11-18
acquisition, and its pipeline half would sit unused beside the stage cache). That leaves
the question of what replaces it, and the answer turns out to be much smaller than DVC.

## Decision

**The committed `manifest.yaml` is the entire index.** It is already in git and already
names every payload file with its sha256. Pull reads the local committed manifest, fetches
each object by hash, and verifies. There is no remote catalogue, no lock file, and no
`.dvc` pointer files — and therefore nothing that can drift out of step with what git says.

**Objects are stored by content hash, never by path**, at
`<prefix>/objects/<first two hex>/<sha256>`. The fan-out exists because several backends
degrade badly on a directory with a million entries.

Three consequences, and the third is the reason:

1. re-pushing an unchanged store is a few existence checks and no transfer;
2. two stores sharing a file store it once;
3. **old manifests never break.** A `SplitFile` binds itself to `store_manifest_sha256` and
   a run record pins its data. Under path addressing, re-ingesting a corpus overwrites
   `signal.bin`, and every split and every result citing the old digest becomes
   unreproducible *in silence* — the files are all still there, they simply mean something
   else now. Under content addressing the old bytes keep their own name, and a twelve-month
   old run pulls exactly what it was trained on.
   `test_an_old_manifest_still_resolves_after_reingest` is that property.

**The remote mapping is committed, at `stores/remotes.yaml`.** Resolution order is
`--remote` → `DSIO_DATA_REMOTE` → the committed mapping. The committed file is what makes
the fresh-clone test pass without manual steps: an environment variable cannot be cloned.
The other two exist so a one-off push elsewhere does not require editing a tracked file.

**Transport is fsspec**, so `s3://`, `gs://`, `az://`, `hf://datasets/user/repo` and a plain
local directory all work with no dsio-side code. HuggingFace Hub remains the interesting
backend for large corpora, because Xet's content-defined chunking dedupes *within* a file —
which the scheme here cannot do on its own, since appending one row changes the whole
digest.

## What is verified, and where

Pull writes each object to a `.partial` name, hashes it, and only then `os.replace`s it into
position; on mismatch it unlinks and raises. A partially-written store that verifies as
complete is precisely the failure the manifest exists to prevent. After all objects land,
`SignalStore.verify()` runs over the assembled store — belt and braces, because the two
checks can fail differently: one catches a bad object, the other catches a store assembled
from individually-valid objects that do not belong together.

Push verifies locally *before* uploading. Publishing bytes that contradict the manifest
would put a corrupt store behind a digest that promises otherwise, which is worse than
failing.

`RemoteIntegrityError` is separate from `RemoteError` because the two demand different
responses: a missing object means push it; wrong bytes mean something is seriously wrong
with the remote and nothing fetched from it should be trusted until that is understood. The
CLI maps them to `integrity` and `remote` respectively.

## The gitignore is load-bearing

```
/stores/*/*
!/stores/*/manifest.yaml
!/stores/*/entities.jsonl
```

`manifest.yaml` must be committed or pull has no index. `entities.jsonl` is committed too:
it is small, and it is what makes a split file reviewable, because group membership lives
there. The two large binaries are ignored.

Pull therefore skips `entities.jsonl` — git already provided it — and that falls out of
digest checking rather than needing a rule of its own. `stores/remotes.yaml` sits one level
up and so is not caught by the `*/*` pattern.

## Consequences

fsspec moves into the dev dependency group while remaining an optional extra for users. The
remote path must be exercised in CI against a local filesystem, or the fresh-clone
reproduction claim is only a claim. The tests run against a real filesystem through fsspec
rather than a mock: a mock would prove the calls happened in the right order, where what
matters is that the bytes arrive and verify.

What is deliberately not built: no garbage collection of unreferenced objects, no remote
listing command, no mirroring between remotes. Each is easy to add against this layout and
none is needed before there is a second machine.
