"""Task runners. One entrypoint; the task kind selects the runner."""

from dsio.train.runner import PREFLIGHTS, RUNNERS, check, execute, preflight, runner

__all__ = [
    "PREFLIGHTS",
    "RUNNERS",
    "check",
    "execute",
    "load_runners",
    "preflight",
    "runner",
]

_BUILTIN_RUNNER_MODULES = ("dsio.train.tabular",)


def load_runners() -> list[str]:
    """Import built-in runner modules so their registrations happen.

    Runners whose optional dependencies are absent are skipped: a clone without
    scikit-learn should still be able to list presets and inspect runs.
    """
    import importlib

    loaded: list[str] = []
    for module in _BUILTIN_RUNNER_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            continue
        loaded.append(module)
    return loaded
