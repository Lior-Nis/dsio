"""The cpu and gpu extras must stay declared as conflicting.

Without the conflicts table uv will try to resolve both torch wheels at once and fail,
or worse, silently pick one. This asserts the declaration itself, which is the thing a
careless dependency edit removes.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_cpu_and_gpu_are_declared_conflicting():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())

    extras = data["project"]["optional-dependencies"]
    assert "cpu" in extras and "gpu" in extras

    conflicts = data["tool"]["uv"]["conflicts"]
    pairs = [{entry["extra"] for entry in group} for group in conflicts]
    assert {"cpu", "gpu"} in pairs

    # Only cpu is index-pinned: the small CPU-only wheel is the whole point of that
    # index. gpu deliberately carries no source override — the default PyPI torch
    # wheel for this platform is already CUDA-enabled, and pinning it to a specific
    # CUDA index would only make it resolve an older build than the unpinned default.
    sources = {entry["extra"]: entry["index"] for entry in data["tool"]["uv"]["sources"]["torch"]}
    assert sources == {"cpu": "pytorch-cpu"}
