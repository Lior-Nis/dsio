"""A missing optional extra must be reported as actionable, not as "dsio has a bug".

``ErrorCode.INTERNAL`` is documented (envelope.py) as meaning dsio itself is broken, as
distinct from an actionable failure the caller can fix. Before this test existed, a bare
``uv sync`` (no ``torch``, deliberately -- see README) made ``dsio run ... --dry-run``
report ``{"code": "internal", "error": "No module named 'torch'"}``: exactly the wrong
signal for the one failure mode every fresh checkout without an extra selected will hit.
"""

from __future__ import annotations

from dsio.cli.envelope import ErrorCode, classify, failure


def test_a_missing_optional_import_is_actionable_not_internal() -> None:
    exc = ModuleNotFoundError("No module named 'torch'")
    exc.name = "torch"
    code, retryable = classify(exc)
    assert code is ErrorCode.MISSING_DEPENDENCY
    assert code is not ErrorCode.INTERNAL
    assert retryable is False


def test_the_failure_message_names_the_extra_to_install() -> None:
    """The whole point: a caller reading the message learns the fix, not just the
    symptom. Watch this fail: revert to ``str(exc) or type(exc).__name__`` in
    ``failure()`` and the message drops back to the bare ``ModuleNotFoundError`` text,
    with no ``uv sync --extra ...`` in it anywhere."""
    exc = ModuleNotFoundError("No module named 'torch'")
    exc.name = "torch"
    payload = failure(exc)
    assert payload["code"] == "missing_dependency"
    assert "uv sync --extra cpu" in payload["error"]


def test_a_submodule_import_still_names_the_top_level_extra() -> None:
    """``import lightning.pytorch`` fails with ``exc.name == "lightning.pytorch"``, not
    the bare top-level package -- the extra lookup has to strip the submodule path or
    every submodule import falls through to the generic fallback message."""
    exc = ModuleNotFoundError("No module named 'lightning.pytorch'")
    exc.name = "lightning.pytorch"
    payload = failure(exc)
    assert "uv sync --extra cpu" in payload["error"]


def test_mlflow_unreachable_is_missing_dependency_not_internal() -> None:
    """MLflow being unreachable used to fall through `classify`'s catch-all to
    `ErrorCode.INTERNAL, False` -- indistinguishable from an actual dsio bug, and marked
    not-retryable even though `docker compose up -d` (or pointing MLFLOW_TRACKING_URI
    elsewhere) fixes it without touching dsio at all. It is the same "an external
    dependency is not available yet" shape `MISSING_DEPENDENCY` already exists for a
    missing Python package -- just an infrastructure dependency instead, and genuinely
    retryable once the stack is up, unlike a missing import (fixed by reinstalling, not
    by retrying)."""
    from dsio.train.tracking import MlflowUnavailableError

    exc = MlflowUnavailableError("MLflow at 'http://localhost:59999' is unreachable")
    code, retryable = classify(exc)
    assert code is ErrorCode.MISSING_DEPENDENCY
    assert code is not ErrorCode.INTERNAL
    assert retryable is True


def test_classify_never_needs_torch_lightning_importable() -> None:
    """The MLflow-unreachable mapping above is looked up through a lazy import
    specifically so this module never has to require torch/lightning at its own module
    scope -- a bare `dsio run` lists presets without either
    (`dsio.train.load_runners`'s own docstring), and this is exactly the module that has
    to turn a missing-torch `ModuleNotFoundError` into a helpful message rather than
    crash trying to report it (this file's own module docstring).

    A regular in-process `monkeypatch` on `sys.modules` cannot test this: pytest has
    already imported `dsio.cli.envelope` (and everything it imports) once, successfully,
    well before this test runs, so re-blocking `lightning` afterward proves nothing about
    whether *importing the module itself* needed it. A fresh subprocess, with `lightning`
    blocked before `dsio.cli.envelope` is ever touched, is the only way to actually
    exercise "this module's own import line never needs it".
    """
    import subprocess
    import sys

    script = (
        "import sys\n"
        "for name in ('lightning', 'lightning.pytorch', 'lightning.pytorch.loggers'):\n"
        "    sys.modules[name] = None\n"
        "import dsio.cli.envelope as envelope\n"
        "code, retryable = envelope.classify(ValueError('something unrelated'))\n"
        "assert code.value == 'invalid_config', code\n"
        "assert retryable is False\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_an_unmapped_missing_module_still_gets_an_actionable_code_and_a_useful_fallback() -> None:
    """A module this table does not know about must not silently fall through to
    ``internal`` -- the code alone already tells an automated caller "this is fixable",
    and the message still points at the general fix even without a specific extra name."""
    exc = ModuleNotFoundError("No module named 'something_unmapped'")
    exc.name = "something_unmapped"
    code, _retryable = classify(exc)
    assert code is ErrorCode.MISSING_DEPENDENCY
    payload = failure(exc)
    assert "extra" in payload["error"] and "pyproject.toml" in payload["error"]
