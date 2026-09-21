# Storage benchmark

Run the complete deterministic synthetic benchmark from the repository root:

```bash
uv run --locked --group benchmark python -m benchmarks.storage \
  --output benchmarks/storage/results/local-synthetic.json
```

Use `--quick` for a smoke run. Use `--real-zarr PATH --real-key KEY` to add an explicit
project-owned corpus, or add `--real-only` to omit the synthetic profiles. Run
`python -m benchmarks.storage --help` for workload controls. An explicit `--work-root`
must be empty; the benchmark refuses to overwrite or mix prior measurements.

The JSON report contains raw measurements and the exact environment. It is evidence, not a
portable speed guarantee. The decision and checked-in result interpretation live in
[`docs/adr/0005-canonical-store-is-flat-binary.md`](../../docs/adr/0005-canonical-store-is-flat-binary.md).
