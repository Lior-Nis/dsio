"""Name-to-value registries populated by decorators.

This replaces Hydra's ``_target_`` string plus reflective import. The difference that
matters: an unknown name fails here at parse time against a known key set, with a
suggestion, instead of failing inside ``instantiate()`` after data loading has begun.
Renaming a registered class is also caught by mypy at every use site, which a string
target never is.

Registries hold values, not types, so the same machinery serves component classes
(``Registry[type[TaskConfig]]``) and preset functions (``Registry[PresetFn]``).
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Any

_ALL_REGISTRIES: dict[str, Registry[Any]] = {}


class UnknownComponentError(KeyError):
    """Raised when a name is not registered in a namespace.

    Carries a rendered message rather than a bare key so the CLI can surface it directly;
    ``KeyError.__str__`` would otherwise wrap the message in its own quotes.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class DuplicateComponentError(ValueError):
    """Raised when two entries claim the same name in one namespace.

    Never a warning. A silent overwrite makes one entry unreachable, and which one wins
    depends on import order — a bug that surfaces months later as a config that quietly
    does the wrong thing.
    """


class Registry[T]:
    """A namespaced mapping from short names to registered values."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._items: dict[str, T] = {}
        _ALL_REGISTRIES[namespace] = self

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator registering a value under ``name`` in this namespace."""

        def decorate(item: T) -> T:
            self.add(name, item)
            return item

        return decorate

    def add(self, name: str, item: T) -> None:
        """Register ``item`` under ``name``, refusing to shadow an existing entry."""
        if name in self._items:
            raise DuplicateComponentError(
                f"{self.namespace!r} already has {name!r} registered to "
                f"{_describe(self._items[name])}; cannot also register {_describe(item)}"
            )
        self._items[name] = item

    def get(self, name: str) -> T:
        """Return the value registered under ``name``, or raise with a suggestion."""
        try:
            return self._items[name]
        except KeyError:
            raise UnknownComponentError(self._unknown_message(name)) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def items(self) -> dict[str, T]:
        return dict(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def _unknown_message(self, name: str) -> str:
        known = self.names()
        message = f"unknown {self.namespace} {name!r}"
        close = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
        if close:
            message += f"; did you mean {', '.join(repr(item) for item in close)}?"
        elif known:
            message += f"; known {self.namespace}s: {', '.join(known)}"
        else:
            message += f"; no {self.namespace}s are registered"
        return message


def _describe(item: Any) -> str:
    module = getattr(item, "__module__", "?")
    qualname = getattr(item, "__qualname__", repr(item))
    return f"{module}.{qualname}"


def all_registries() -> dict[str, Registry[Any]]:
    """Return every registry created so far, keyed by namespace."""
    return dict(_ALL_REGISTRIES)
