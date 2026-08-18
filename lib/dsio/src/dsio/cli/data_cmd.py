"""``dsio data`` — inspect, verify and index canonical stores.

Every listing has a projected form. A store with a few hundred entities produces more JSON
than any caller wants by default, so the projection ships from the start and detail is
opt-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from dsio.cli.envelope import json_command
from dsio.data import SignalStore, StageCache, WindowSpec, build_index, load_or_build
from dsio.data.remote import Transfer, pull, push, status
from dsio.data.store import SIGNAL_FILE
from dsio.data.views import index_path

app = typer.Typer(help="Inspect and verify canonical stores.", no_args_is_help=True)

STORES_ROOT = Path("stores")


@app.command("ls")
@json_command
def list_stores(
    root: Annotated[Path, typer.Option(help="Directory holding the stores.")] = STORES_ROOT,
) -> dict[str, Any]:
    """List every store under the root, with its size and entity count."""
    if not root.is_dir():
        return {"root": str(root), "count": 0, "stores": []}
    stores = []
    for path in sorted(root.iterdir()):
        if not (path / "manifest.yaml").is_file():
            continue
        store = SignalStore(path)
        stores.append(
            {
                "name": path.name,
                "entities": len(store.entities),
                "groups": len(store.groups),
                "rows": store.n_rows,
                "channels": store.channels,
                "bytes": (path / SIGNAL_FILE).stat().st_size,
            }
        )
    return {"root": str(root), "count": len(stores), "stores": stores}


@app.command("show")
@json_command
def show(
    store: Annotated[Path, typer.Argument(help="Path to a store directory.")],
    entities: Annotated[
        bool, typer.Option("--entities", help="Include the full entity listing.")
    ] = False,
    limit: Annotated[int, typer.Option(help="Entities to show when listing them.")] = 20,
) -> dict[str, Any]:
    """Describe one store: shape, groups, and where its bytes live."""
    signal = SignalStore(store)
    manifest = signal.manifest()
    payload: dict[str, Any] = {
        "name": signal.path.name,
        "rows": signal.n_rows,
        "channels": signal.channels,
        "dtype": str(signal.header.dtype),
        "entities": len(signal.entities),
        "groups": sorted(signal.groups),
        "signal_sha256": manifest.signal_sha256[:16],
        "bytes": (signal.path / SIGNAL_FILE).stat().st_size,
    }
    if entities:
        payload["entity_list"] = [
            {
                "id": entity.entity_id,
                "group": entity.group,
                "start_row": entity.start_row,
                "n_rows": entity.n_rows,
                "attrs": entity.attrs,
            }
            for entity in signal.entities[:limit]
        ]
        payload["entity_list_truncated"] = len(signal.entities) > limit
    return payload


@app.command("verify")
@json_command
def verify(
    store: Annotated[Path, typer.Argument(help="Path to a store directory.")],
) -> dict[str, Any]:
    """Re-hash a store's bytes and check them against its committed manifest.

    Fails closed. A store that has silently changed under a committed split file is the
    situation this exists to catch, and reporting "probably fine" would defeat it.
    """
    signal = SignalStore(store)
    signal.verify()
    return {
        "name": signal.path.name,
        "verified": True,
        "rows": signal.n_rows,
        "signal_sha256": signal.manifest().signal_sha256,
    }


@app.command("index")
@json_command
def index(
    store: Annotated[Path, typer.Argument(help="Path to a store directory.")],
    length: Annotated[int, typer.Option(help="Window length in rows.")] = 500,
    stride: Annotated[int, typer.Option(help="Rows between window starts.")] = 250,
    build: Annotated[
        bool, typer.Option("--build/--dry-run", help="Write the index, or only report it.")
    ] = True,
) -> dict[str, Any]:
    """Build (or describe) a windowed view over a store.

    A window-config change costs seconds and megabytes here rather than hours and tens of
    gigabytes: the index is offsets, and the signal is never copied. Materialising one
    corpus per (length, stride, policy) is how a store grows to hundreds of gigabytes.
    """
    signal = SignalStore(store)
    spec = WindowSpec(length=length, stride=stride)
    destination = index_path(signal, spec)
    windows = load_or_build(signal, spec) if build else build_index(signal, spec)
    return {
        "store": signal.path.name,
        "spec": spec.model_dump(mode="json"),
        "digest": spec.digest,
        "windows": len(windows),
        "groups": len(set(windows.entity_groups)),
        "path": str(destination),
        "written": build,
    }


@app.command("push")
@json_command
def push_cmd(
    store: Annotated[Path, typer.Argument(help="Path to a store directory.")],
    remote: Annotated[
        str | None, typer.Option(help="fsspec URL. Defaults to the committed mapping.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would transfer, send nothing.")
    ] = False,
) -> dict[str, Any]:
    """Upload a store's payload to its remote, skipping objects already there.

    Verifies locally first. Publishing bytes that do not match the manifest would put a
    corrupt store behind a digest that promises otherwise, which is worse than failing.
    """
    transfers = push(store, remote=remote, dry_run=dry_run)
    return _transfer_payload(Path(store).name, transfers, dry_run=dry_run)


@app.command("pull")
@json_command
def pull_cmd(
    name: Annotated[str, typer.Argument(help="Store name under the data root.")],
    remote: Annotated[
        str | None, typer.Option(help="fsspec URL. Defaults to the committed mapping.")
    ] = None,
    root: Annotated[Path | None, typer.Option(help="Data root. Defaults to stores/.")] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-fetch even when local bytes already match.")
    ] = False,
) -> dict[str, Any]:
    """Fetch a store's payload using its committed manifest, verifying every object.

    The manifest is the index, so this needs no remote listing and cannot be handed
    something other than the bytes git already names. Fails closed on a digest mismatch
    and leaves nothing behind.
    """
    transfers = pull(name, remote=remote, root=root, force=force)
    return _transfer_payload(name, transfers, dry_run=False)


@app.command("status")
@json_command
def status_cmd(
    name: Annotated[str, typer.Argument(help="Store name under the data root.")],
    remote: Annotated[str | None, typer.Option(help="fsspec URL.")] = None,
    root: Annotated[Path | None, typer.Option(help="Data root. Defaults to stores/.")] = None,
) -> dict[str, Any]:
    """Report, per file, whether the bytes are here, on the remote, both or neither."""
    return status(name, remote=remote, root=root)


def _transfer_payload(name: str, transfers: list[Transfer], *, dry_run: bool) -> dict[str, Any]:
    moved = [item for item in transfers if not item.skipped]
    return {
        "store": name,
        "dry_run": dry_run,
        "transferred": len(moved),
        "skipped": len(transfers) - len(moved),
        "bytes": sum(item.bytes for item in moved),
        "files": [
            {
                "file": item.filename,
                "digest": item.digest[:16],
                "bytes": item.bytes,
                "skipped": item.skipped,
            }
            for item in transfers
        ],
    }


cache_app = typer.Typer(help="Inspect the stage cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


@cache_app.command("ls")
@json_command
def cache_ls(
    root: Annotated[Path, typer.Option(help="Cache directory.")] = Path("cache"),
    stage: Annotated[str | None, typer.Option(help="Only entries for this stage.")] = None,
) -> dict[str, Any]:
    """List cached stage outputs, newest first."""
    cache = StageCache(root)
    entries = cache.entries(stage)
    return {
        "root": str(root),
        "count": len(entries),
        "bytes": cache.size_bytes(),
        "entries": [
            {
                "stage": entry.stage,
                "key": entry.key,
                "version": entry.version,
                "bytes": entry.output_bytes,
                "seconds": entry.seconds,
                "created_at": entry.created_at,
            }
            for entry in entries
        ],
    }
