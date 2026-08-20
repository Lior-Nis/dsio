"""Guards the builtin runner module list against stale entries.

``load_runners`` imports each name in ``_BUILTIN_RUNNER_MODULES`` and swallows
``ImportError`` so a clone without an optional dependency (e.g. scikit-learn) can
still list presets and inspect runs. That same ``except ImportError: continue`` also
swallows a ``ModuleNotFoundError`` for an entry that points at code which no longer
exists at all, so a deleted package can leave a dead lookup behind that never fails
loudly. This test asserts every entry names a module that is actually importable.
"""

import importlib

from dsio.train import _BUILTIN_RUNNER_MODULES


def test_every_builtin_runner_module_is_importable() -> None:
    """A stale entry is swallowed by load_runners' except ImportError, so it can
    rot silently. This asserts the list names only modules that actually exist."""
    for name in _BUILTIN_RUNNER_MODULES:
        importlib.import_module(name)
