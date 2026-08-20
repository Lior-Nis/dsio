"""Dotted ``key.path=value`` overrides applied to a config mapping.

Overrides may only set values at paths that already exist. Creating a path on demand
would let ``optim.learning_rate=3e-4`` silently add a key the model ignores; requiring
the path to exist turns that typo into an error naming the valid siblings.

Scalar parsing is explicit and ordered rather than delegated to a YAML loader, because
PyYAML's 1.1 resolver reads ``3e-4`` as the *string* ``"3e-4"`` — and a learning rate
silently becoming a string is exactly the class of failure this system exists to prevent.
"""

from __future__ import annotations

import json
from typing import Any

_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}
_NULL = {"null", "none", "~"}


class OverrideError(ValueError):
    """Raised when an override is malformed or targets a path that does not exist."""


def parse_scalar(text: str) -> Any:
    """Parse an override's right-hand side into a Python value.

    Order is significant and deliberate:
    JSON literals, then int, then float, then bool/null words, then plain string.
    """
    stripped = text.strip()
    if stripped == "":
        return ""

    if stripped[0] in "[{\"" or (stripped[0] == "'" and stripped[-1] == "'"):
        try:
            return json.loads(stripped.replace("'", '"') if stripped[0] == "'" else stripped)
        except json.JSONDecodeError:
            return stripped

    try:
        return int(stripped)
    except ValueError:
        pass

    try:
        return float(stripped)
    except ValueError:
        pass

    lowered = stripped.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    if lowered in _NULL:
        return None

    return stripped


def split_override(token: str) -> tuple[str, Any]:
    """Split ``a.b=c`` into a dotted path and a parsed value."""
    cleaned = token.lstrip("-")
    if "=" not in cleaned:
        raise OverrideError(
            f"malformed override {token!r}; expected the form key.path=value"
        )
    path, _, raw_value = cleaned.partition("=")
    path = path.strip()
    if not path:
        raise OverrideError(f"malformed override {token!r}; the key is empty")
    return path, parse_scalar(raw_value)


def apply_override(data: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """Return a copy of ``data`` with ``path`` set to ``value``.

    Every segment of ``path`` must already exist. Raises :class:`OverrideError` naming
    the available keys otherwise.
    """
    segments = path.split(".")
    result = _deep_copy_mapping(data)
    cursor: Any = result
    walked: list[str] = []

    for segment in segments[:-1]:
        if not isinstance(cursor, dict) or segment not in cursor:
            raise OverrideError(_no_such_path(cursor, walked, segment))
        walked.append(segment)
        cursor = cursor[segment]

    leaf = segments[-1]
    if not isinstance(cursor, dict) or leaf not in cursor:
        raise OverrideError(_no_such_path(cursor, walked, leaf))
    cursor[leaf] = value
    return result


def apply_overrides(data: dict[str, Any], tokens: list[str]) -> dict[str, Any]:
    """Apply a sequence of ``key.path=value`` tokens in order."""
    result = data
    for token in tokens:
        path, value = split_override(token)
        result = apply_override(result, path, value)
    return result


def _deep_copy_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Copy nested dicts so an override never mutates the caller's structure."""
    return {
        key: _deep_copy_mapping(value) if isinstance(value, dict) else value
        for key, value in data.items()
    }


def _no_such_path(cursor: Any, walked: list[str], segment: str) -> str:
    prefix = ".".join([*walked, segment])
    if isinstance(cursor, dict):
        available = ", ".join(sorted(cursor)) or "(no keys)"
        parent = ".".join(walked) or "the config root"
        return f"no such config path {prefix!r}; {parent} has: {available}"
    parent = ".".join(walked) or "the config root"
    return f"no such config path {prefix!r}; {parent} is not a mapping"
