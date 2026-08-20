"""The cpu and cu128 extras must stay declared as conflicting.

Without the conflicts table uv will try to resolve both torch wheels at once and fail,
or worse, silently pick one. This asserts the declaration itself, which is the thing a
careless dependency edit removes.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_cpu_and_cu128_are_declared_conflicting():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())

    extras = data["project"]["optional-dependencies"]
    assert "cpu" in extras and "cu128" in extras

    conflicts = data["tool"]["uv"]["conflicts"]
    pairs = [{entry["extra"] for entry in group} for group in conflicts]
    assert {"cpu", "cu128"} in pairs

    sources = {entry["extra"]: entry["index"] for entry in data["tool"]["uv"]["sources"]["torch"]}
    assert sources == {"cpu": "pytorch-cpu", "cu128": "pytorch-cu128"}
