"""Task runners. One entrypoint; the task kind selects the runner."""

_BUILTIN_RUNNER_MODULES = (
    "dsio.train.torch_task",
    "dsio.train.ssl_task",
)
_OPTIONAL_RUNNER_PACKAGES = frozenset({"torch", "lightning", "mlflow"})


def load_runners() -> list[str]:
    """Import built-in runner modules so their registrations happen.

    Missing framework dependencies are tolerated here for compatibility with legacy
    recorded-run inspection. New installations include the complete training stack.
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
