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

T = TypeVar("T")


class ErrorCode(StrEnum):
    UNKNOWN_COMPONENT = "unknown_component"
    BAD_OVERRIDE = "bad_override"
    INVALID_CONFIG = "invalid_config"
    DUPLICATE_COMPONENT = "duplicate_component"
    NOT_FOUND = "not_found"
    INTEGRITY = "integrity"
    BLOCKED = "blocked"
    INTERNAL = "internal"


class BlockedError(RuntimeError):
    """Raised when an action is refused by a gate, such as promotion of a dirty run."""


_CODES: list[tuple[type[BaseException], ErrorCode, bool]] = [
    (UnknownComponentError, ErrorCode.UNKNOWN_COMPONENT, False),
    (DuplicateComponentError, ErrorCode.DUPLICATE_COMPONENT, False),
    (OverrideError, ErrorCode.BAD_OVERRIDE, False),
    (ValidationError, ErrorCode.INVALID_CONFIG, False),
    (RegistryIntegrityError, ErrorCode.INTEGRITY, False),
    (BlockedError, ErrorCode.BLOCKED, False),
    (FileNotFoundError, ErrorCode.NOT_FOUND, False),
    (ValueError, ErrorCode.INVALID_CONFIG, False),
    (OSError, ErrorCode.INTERNAL, True),
]


def classify(exc: BaseException) -> tuple[ErrorCode, bool]:
    """Map an exception to an error code and whether retrying could help."""
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
    return {
        "ok": False,
        "error": str(exc) or type(exc).__name__,
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
