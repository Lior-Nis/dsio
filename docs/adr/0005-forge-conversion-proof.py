"""Convert a real FORGE Zarr store into a dsio canonical store, and prove it is lossless.

This is the test that decides whether "store once, index many" actually works, so it is
deliberately adversarial: rebuild the continuous signal, re-derive the windows, and assert
they are *byte-identical* to the ones FORGE materialised.

Reconstruction is possible because FORGE records `start_frame` per window within its
session. Overlapping windows must agree on the rows they share — checking that is a free
consistency test on the reconstruction itself, and it would catch a wrong assumption about
the layout immediately rather than silently.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import zarr

SRC = "/home/liornisimov/Projects/acc_base/data/processed/len500_stride200_fogstride100_anyfog_kaggle_defog.zarr"
DST = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/forge_dsio/kaggle_defog")
N_SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
LENGTH = 500
STRIDE = 200


def main() -> int:
    from dsio.data import SignalStore, WindowSpec, WindowView, build_index

    z = zarr.open(SRC, mode="r")
    meta = z["metadata"]
    session_id = meta["session_id"][:]
    patient_id = meta["patient_id"][:]
    start_frame = meta["start_frame"][:].astype(np.int64)

    by_session: dict[str, list[int]] = defaultdict(list)
    for i, sid in enumerate(session_id):
        by_session[str(sid)].append(i)

    chosen = sorted(by_session)[:N_SESSIONS]
    print(f"{len(by_session)} sessions total; converting {len(chosen)}", flush=True)

    accs = z["accs"]
    sessions: dict[str, tuple[np.ndarray, str]] = {}
    overlap_checked = 0

    for sid in chosen:
        rows = sorted(by_session[sid], key=lambda i: start_frame[i])
        total = int(start_frame[rows[-1]]) + LENGTH
        signal = np.full((total, 3), np.nan, dtype=np.float32)

        for i in rows:
            s = int(start_frame[i])
            window = accs[i]
            existing = signal[s : s + LENGTH]
            seen = ~np.isnan(existing[:, 0])
            if seen.any():
                # Overlapping windows must agree; if they do not, the reconstruction
                # assumption is wrong and every downstream comparison is meaningless.
                if not np.allclose(existing[seen], window[seen], rtol=0, atol=0):
                    raise SystemExit(f"session {sid}: overlapping windows disagree at {s}")
                overlap_checked += int(seen.sum())
            signal[s : s + LENGTH] = window

        if np.isnan(signal[:, 0]).any():
            gaps = int(np.isnan(signal[:, 0]).sum())
            print(f"  {sid}: {gaps} uncovered rows (sparse coverage), trimming", flush=True)
            keep = ~np.isnan(signal[:, 0])
            signal = signal[keep]
        sessions[sid] = (signal, str(patient_id[rows[0]]))

    print(f"overlap rows verified identical: {overlap_checked:,}", flush=True)

    if DST.exists():
        import shutil

        shutil.rmtree(DST)
    with SignalStore.builder(DST, channels=3, dtype="float32", source=SRC) as b:
        for sid, (signal, pid) in sessions.items():
            b.add(sid, signal, group=pid, attrs={"protocol": "defog"})

    store = SignalStore(DST)
    store.verify()
    print(f"\n{store}\nverify: ok", flush=True)

    index = build_index(store, WindowSpec(length=LENGTH, stride=STRIDE))
    view = WindowView(store, index)
    print(index, flush=True)

    # --- the proof: every stride-aligned FORGE window must appear, byte-identical ------
    lookup = {
        (str(index.entity_ids[k]), int(index.starts[k]) - store.entity(str(index.entity_ids[k])).start_row): k
        for k in range(len(index))
    }
    compared = missing = 0
    for sid in chosen:
        for i in by_session[sid]:
            s = int(start_frame[i])
            if s % STRIDE:
                continue  # dense-stride extra, not on the base grid
            key = (sid, s)
            if key not in lookup:
                missing += 1
                continue
            got = view[lookup[key]]
            want = accs[i]
            if not np.array_equal(got, want):
                raise SystemExit(f"MISMATCH session={sid} start={s}")
            compared += 1

    print(f"\nwindows compared byte-identical: {compared:,}")
    print(f"windows missing from dsio index : {missing}")

    src_bytes = sum(f.stat().st_size for f in Path(SRC).rglob("*") if f.is_file())
    dst_bytes = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
    frac = len(chosen) / len(by_session)
    print(
        f"\nsize: FORGE {src_bytes / 1e9:.2f} GB total "
        f"(~{src_bytes * frac / 1e9:.2f} GB for this subset) "
        f"-> dsio {dst_bytes / 1e9:.2f} GB"
    )
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
