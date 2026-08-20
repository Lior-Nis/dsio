"""The declared import-linter contracts must stay declared.

``lint-imports`` exits 0 when zero contracts are declared — an empty
``[tool.importlinter]`` section is a silent pass, not a warning. Nothing else in the
suite reads ``[tool.importlinter]``, so a careless edit that deletes a contract block
(or renames one enough to break a downstream reference) makes CI's "Import contracts"
step keep passing while the leakage wall it was guarding is gone. This asserts the
three contracts by name, which is the thing a careless edit removes.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_all_declared_contracts_are_present():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())

    contracts = data["tool"]["importlinter"]["contracts"]
    names = {contract["name"] for contract in contracts}

    assert names == {
        "Foundation modules import no other dsio module",
        "Evaluation depends on no pipeline layer",
        "The data layer never imports splits",
    }
