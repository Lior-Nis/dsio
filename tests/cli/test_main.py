"""The CLI composition root mutates process policy only when it actually runs."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pytest

import dsio.cli.main as cli_main

_SUPPRESS_MLFLOW_BANNER = "MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"


@pytest.mark.parametrize("initial", [None, "false"])
def test_importing_cli_main_does_not_change_mlflow_suppression(initial: str | None) -> None:
    setup = (
        f"os.environ[{_SUPPRESS_MLFLOW_BANNER!r}] = {initial!r}"
        if initial is not None
        else f"os.environ.pop({_SUPPRESS_MLFLOW_BANNER!r}, None)"
    )
    probe = (
        "import os\n"
        f"{setup}\n"
        f"before = os.environ.get({_SUPPRESS_MLFLOW_BANNER!r})\n"
        "import dsio.cli.main\n"
        f"assert os.environ.get({_SUPPRESS_MLFLOW_BANNER!r}) == before\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_main_overwrites_mlflow_suppression_before_invoking_typer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def command(**kwargs: Any) -> None:
        assert os.environ[_SUPPRESS_MLFLOW_BANNER] == "true"
        calls.append(kwargs)

    monkeypatch.setenv(_SUPPRESS_MLFLOW_BANNER, "false")
    monkeypatch.setattr(cli_main.typer.main, "get_command", lambda app: command)

    cli_main.main()

    assert calls == [{"standalone_mode": False}]
