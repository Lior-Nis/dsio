"""Guards the builtin runner module list against stale entries.

``load_runners`` imports each name in ``_BUILTIN_RUNNER_MODULES`` while tolerating an
absent runner extra. A stale module entry is not an absent extra and must fail loudly.
These tests pin both sides of that boundary.

An empty tuple would make ``test_every_builtin_runner_module_is_importable`` pass
vacuously (zero iterations) while leaving no task registered at all: ``dsio run``
still works, but reading back any recorded config dies with ``UnknownComponentError:
unknown task 'torch'``. ``test_builtin_runner_modules_is_not_empty`` below, and its
twin in ``tests/config/test_preset_discovery.py``, exist so that failure mode cannot
hide behind a vacuously-true loop.
"""

import importlib

import pytest

import dsio.train as train
from dsio.train import _BUILTIN_RUNNER_MODULES


def test_builtin_runner_modules_is_not_empty() -> None:
    """A stale-entry check over an empty tuple passes for the wrong reason. Without
    this, setting ``_BUILTIN_RUNNER_MODULES = ()`` makes every runner-registration
    guard here pass while no runner is ever registered."""
    assert _BUILTIN_RUNNER_MODULES, "no runner modules declared"


def test_every_builtin_runner_module_is_importable() -> None:
    """The declared module names must remain importable in the full test environment."""
    for name in _BUILTIN_RUNNER_MODULES:
        importlib.import_module(name)


@pytest.mark.parametrize("missing", ["torch", "lightning", "mlflow"])
def test_load_runners_skips_only_an_absent_supported_extra(
    missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_import(_: str) -> None:
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    monkeypatch.setattr(importlib, "import_module", fail_import)

    assert train.load_runners() == []


@pytest.mark.parametrize(
    "failure",
    [
        ModuleNotFoundError("missing internal module", name="dsio.train.missing"),
        ModuleNotFoundError("broken optional package", name="lightning.pytorch"),
        ImportError("runner import failed"),
    ],
)
def test_load_runners_propagates_broken_imports(
    failure: ImportError, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_import(_: str) -> None:
        raise failure

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(type(failure)) as caught:
        train.load_runners()
    assert caught.value is failure
