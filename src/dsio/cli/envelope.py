"""A single JSON envelope for every command, success or failure.

One shape — ``{ok, error, code, retryable}`` — means a human and an agent drive the tool
identically, and a caller can branch on ``code`` without parsing prose. ``retryable``
distinguishes "your input was wrong" from "try again", which is what an automated caller
actually needs to decide.

Errors are rendered here rather than allowed to reach a traceback, because a traceback on
stdout is not a contract.
"""

from __future__ import annotations

import functools
import json
import sys
from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import ValidationError

from dsio.artifacts.store import RegistryIntegrityError
from dsio.config.overrides import OverrideError
from dsio.config.registry import DuplicateComponentError, UnknownComponentError
from dsio.data.staging import StagingError
from dsio.data.store import StoreError
from dsio.eval.contract import EvalError
from dsio.splits.models import SplitError

T = TypeVar("T")


class ErrorCode(StrEnum):
    UNKNOWN_COMPONENT = "unknown_component"
    BAD_OVERRIDE = "bad_override"
    INVALID_CONFIG = "invalid_config"
    DUPLICATE_COMPONENT = "duplicate_component"
    NOT_FOUND = "not_found"
    INTEGRITY = "integrity"
    LEAKAGE = "leakage"
    BLOCKED = "blocked"
    MISSING_DEPENDENCY = "missing_dependency"
    INTERNAL = "internal"


class BlockedError(RuntimeError):
    """Raised when an action is refused by a gate, such as promotion of a dirty run."""


# Which optional-dependency extra (see pyproject.toml) provides each import a bare
# `uv sync` leaves missing. Keyed on the top-level module name a ModuleNotFoundError
# names, not the distribution name -- ``import sklearn`` fails on ``sklearn``, not
# ``scikit-learn``.
_EXTRA_FOR_MODULE: dict[str, str] = {
    "torch": "cpu",
    "lightning": "cpu",
    "torchmetrics": "cpu",
    "sklearn": "tabular",
    "pyarrow": "data",
}


def _missing_dependency_message(exc: ModuleNotFoundError) -> str:
    """A missing optional import is an actionable setup gap, not a dsio bug.

    ``uv sync`` with no extra selected installs no torch at all, on purpose (README) --
    so the first thing a fresh checkout does without one is fail somewhere deep inside a
    runner with a bare ``ModuleNotFoundError``. Naming the extra that fixes it is the
    difference between "dsio is broken" and "you forgot a setup step".
    """
    module = (exc.name or "").split(".")[0]
    extra = _EXTRA_FOR_MODULE.get(module)
    hint = (
        f"install it with `uv sync --extra {extra}`"
        if extra is not None
        else "install the optional-dependency extra that provides it (see pyproject.toml)"
    )
    return f"{exc}; {hint}"


# Order matters: the first match wins, so specific types precede the ValueError and
# OSError catch-alls they inherit from.
_CODES: list[tuple[type[BaseException], ErrorCode, bool]] = [
    (UnknownComponentError, ErrorCode.UNKNOWN_COMPONENT, False),
    (DuplicateComponentError, ErrorCode.DUPLICATE_COMPONENT, False),
    (OverrideError, ErrorCode.BAD_OVERRIDE, False),
    (ValidationError, ErrorCode.INVALID_CONFIG, False),
    # Integrity is its own code because a caller must be able to tell "your bytes changed"
    # from "dsio has a bug". The first is actionable — restore the store, regenerate the
    # split — and the second is a bug report.
    (RegistryIntegrityError, ErrorCode.INTEGRITY, False),
    (StoreError, ErrorCode.INTEGRITY, False),
    (StagingError, ErrorCode.INTEGRITY, False),
    # Leakage is separated from ordinary invalid input because it is the one failure class
    # that must never be retried around or suppressed by an automated caller.
    (SplitError, ErrorCode.LEAKAGE, False),
    (EvalError, ErrorCode.LEAKAGE, False),
    (BlockedError, ErrorCode.BLOCKED, False),
    (FileNotFoundError, ErrorCode.NOT_FOUND, False),
    # An optional extra not being installed (torch, sklearn, ...) is a setup gap the
    # caller can fix, not a dsio bug -- must precede ValueError/OSError below since
    # ModuleNotFoundError is neither.
    (ModuleNotFoundError, ErrorCode.MISSING_DEPENDENCY, False),
    (ValueError, ErrorCode.INVALID_CONFIG, False),
    (OSError, ErrorCode.INTERNAL, True),
]


def _mlflow_unavailable_error_type() -> type[BaseException] | None:
    """``MlflowUnavailableError``'s type, or ``None`` if it cannot even be imported.

    Deliberately not a module-level import: ``dsio.train.tracking`` imports Lightning at
    its own module scope, and this module (``dsio.cli.envelope``) must stay importable
    without torch/lightning installed at all -- a bare ``dsio run`` lists presets without
    either (``dsio.train.load_runners``'s own docstring), and this is exactly the module
    that has to turn a missing-torch ``ModuleNotFoundError`` into a helpful message
    rather than crash trying to report it. If ``exc`` really is an
    ``MlflowUnavailableError``, that module necessarily already imported successfully to
    raise it, so this import is a cache hit off ``sys.modules``, never the first (and
    possibly failing) import of Lightning.
    """
    try:
        from dsio.train.tracking import MlflowUnavailableError
    except ImportError:
        return None
    return MlflowUnavailableError


def classify(exc: BaseException) -> tuple[ErrorCode, bool]:
    """Map an exception to an error code and whether retrying could help."""
    mlflow_unavailable = _mlflow_unavailable_error_type()
    if mlflow_unavailable is not None and isinstance(exc, mlflow_unavailable):
        # I5's sibling fix, on the CLI side: unreachable MLflow used to fall through to
        # the generic `ErrorCode.INTERNAL, False` below, indistinguishable from an actual
        # dsio bug. It is retryable -- `docker compose up -d`, or pointing
        # `MLFLOW_TRACKING_URI` elsewhere, fixes it without touching dsio -- the same
        # "an external dependency is not available yet" shape `MISSING_DEPENDENCY`
        # already exists for, just an infrastructure dependency rather than a Python one.
        return ErrorCode.MISSING_DEPENDENCY, True
    for exc_type, code, retryable in _CODES:
        if isinstance(exc, exc_type):
            return code, retryable
    return ErrorCode.INTERNAL, False


def emit(payload: dict[str, Any]) -> None:
    """Write one JSON object to stdout."""
    json.dump(payload, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


def ok(**data: Any) -> dict[str, Any]:
    return {"ok": True, "error": None, "code": None, "retryable": False, **data}


def failure(exc: BaseException) -> dict[str, Any]:
    code, retryable = classify(exc)
    if isinstance(exc, ModuleNotFoundError):
        message = _missing_dependency_message(exc)
    else:
        message = str(exc) or type(exc).__name__
    return {
        "ok": False,
        "error": message,
        "code": str(code),
        "retryable": retryable,
    }


def json_command(fn: Callable[..., dict[str, Any]]) -> Callable[..., None]:
    """Wrap a command so it always emits one envelope and exits with a meaningful code.

    ``BaseException`` is deliberately not caught: Ctrl-C and SystemExit must still
    terminate the process rather than being reported as a tidy JSON failure.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            emit(failure(exc))
            raise SystemExit(1) from exc
        emit(ok(**result))

    return wrapper
