"""Task runners. One entrypoint; the task kind selects the runner."""

_BUILTIN_RUNNER_MODULES = (
    "dsio.train.torch_task",
    "dsio.train.ssl_task",
)
_OPTIONAL_RUNNER_PACKAGES = frozenset({"torch", "lightning", "mlflow"})


def load_runners() -> list[str]:
    """Import built-in runner modules so their registrations happen.

    Runners whose optional dependencies are absent are skipped: a clone without
    torch and lightning should still be able to list presets and inspect runs.
    """
    import importlib

    loaded: list[str] = []
    for module in _BUILTIN_RUNNER_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name not in _OPTIONAL_RUNNER_PACKAGES:
                raise
            continue
        loaded.append(module)
    return loaded
