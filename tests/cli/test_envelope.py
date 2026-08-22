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
