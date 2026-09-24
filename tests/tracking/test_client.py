from __future__ import annotations

import os

import pytest

from dsio.tracking.client import (
    DEFAULT_TRACKING_URI,
    TRACKING_URI_ENV,
    resolve_tracking_uri,
    temporary_mlflow_environment,
)


def test_tracking_uri_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRACKING_URI_ENV, raising=False)
    assert resolve_tracking_uri() == DEFAULT_TRACKING_URI == "http://localhost:5000"

    monkeypatch.setenv(TRACKING_URI_ENV, "http://environment.example:5000")
    assert resolve_tracking_uri() == "http://environment.example:5000"
    assert resolve_tracking_uri("http://explicit.example:5000") == ("http://explicit.example:5000")


@pytest.mark.parametrize("existing", [None, "before"])
@pytest.mark.parametrize("raises", [False, True])
def test_temporary_mlflow_environment_restores_values(
    monkeypatch: pytest.MonkeyPatch,
    existing: str | None,
    raises: bool,
) -> None:
    name = "DSIO_TEST_MLFLOW_ENVIRONMENT"
    if existing is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, existing)

    def exercise() -> None:
        with temporary_mlflow_environment({name: "during"}):
            assert os.environ[name] == "during"
            if raises:
                raise RuntimeError("operation failed")

    if raises:
        with pytest.raises(RuntimeError, match="operation failed"):
            exercise()
    else:
        exercise()

    assert os.environ.get(name) == existing
