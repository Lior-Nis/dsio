"""Guards the component-registry bootstrap against a silently-neutered import.

``dsio.nn.__init__`` imports :mod:`dsio.nn.components` purely for its side effect of
populating ``BACKBONES``/``HEADS``/``LOSSES`` via their decorators. Nothing in that
module is otherwise used, so it is one accidental docstring-only rewrite away from
silently emptying every registry: ``run_cmd._bootstrap`` would still succeed, but
every torch/ssl_pretrain preset would fail at model-build time with "unknown
backbone/head/loss".

This can't be checked in-process: sibling test modules (``tests/nn/test_module.py``,
``tests/ssl/test_methods.py``, ``tests/ssl/test_probe.py``) import
``dsio.nn.components`` at collection time, which populates the registries before any
test body runs — regardless of whether ``dsio.nn.__init__`` still does the import
itself. A subprocess that imports only ``dsio.nn.registry`` is the only way to see
what a virgin process actually gets.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")


def test_importing_nn_package_populates_registries() -> None:
    """Importing ``dsio.nn`` alone (not ``dsio.nn.components``) must register the
    built-in backbones, heads, and losses. This is what every preset relies on."""
    probe = (
        "import dsio.nn as nn_pkg\n"
        "import dsio.nn.registry as r\n"
        "msg = 'empty: nn/__init__.py stopped importing components'\n"
        "assert r.BACKBONES.names(), 'BACKBONES ' + msg\n"
        "assert r.HEADS.names(), 'HEADS ' + msg\n"
        "assert r.LOSSES.names(), 'LOSSES ' + msg\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"registry bootstrap failed in a virgin process\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
