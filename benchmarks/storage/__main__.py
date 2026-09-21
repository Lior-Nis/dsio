"""Command-line entry point for the reproducible storage benchmark."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.storage.benchmark import PROFILES, run_benchmark, write_report
from benchmarks.storage.candidates import CANDIDATES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--reads", type=int, default=2_000)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-zarr", type=Path)
    parser.add_argument("--real-key", default="accs")
    parser.add_argument("--real-name", default="real-corpus")
    parser.add_argument("--real-row-limit", type=int)
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.rows = 4096
        args.window = 64
        args.reads = 16
        args.workers = [1, 2]

    real_sources: tuple[tuple[str, np.ndarray, dict[str, object]], ...] = ()
    if args.real_zarr is not None:
        import zarr

        source = zarr.open(str(args.real_zarr), mode="r")[args.real_key]
        original_shape = list(source.shape)
        if args.real_row_limit is not None and source.ndim > 2:
            rows_per_item = int(np.prod(source.shape[1:-1]))
            item_limit = (args.real_row_limit + rows_per_item - 1) // rows_per_item
            array = np.asarray(source[:item_limit])
        else:
            array = np.asarray(source)
        array = array.reshape(-1, array.shape[-1])
        if args.real_row_limit is not None:
            array = array[: args.real_row_limit]
        real_sources = (
            (
                args.real_name,
                array,
                {
                    "format": "zarr",
                    "path": str(args.real_zarr),
                    "key": args.real_key,
                    "original_shape": original_shape,
                    "row_limit": args.real_row_limit,
                },
            ),
        )
    if args.real_only and not real_sources:
        parser.error("--real-only requires --real-zarr")
    profiles = () if args.real_only else tuple(args.profiles)

    if args.work_root is not None:
        report = run_benchmark(
            args.work_root,
            profiles=profiles,
            candidates=tuple(args.candidates),
            rows=args.rows,
            channels=args.channels,
            window=args.window,
            reads=args.reads,
            workers=tuple(args.workers),
            seed=args.seed,
            real_sources=real_sources,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="dsio-store-benchmark-") as directory:
            report = run_benchmark(
                Path(directory),
                profiles=profiles,
                candidates=tuple(args.candidates),
                rows=args.rows,
                channels=args.channels,
                window=args.window,
                reads=args.reads,
                workers=tuple(args.workers),
                seed=args.seed,
                real_sources=real_sources,
            )
    write_report(report, args.output)


if __name__ == "__main__":
    main()
