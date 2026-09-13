"""The composition root: mounts the one command the CLI has.

A command exists only if it is needed before there is a Python session — everything else
happens where Python is already available, in a repo you own. That leaves ``run``.
"""

from __future__ import annotations

import os

import typer

from dsio.cli import run_cmd
from dsio.cli.envelope import emit, failure

app = typer.Typer(
    name="dsio",
    help="Reproducible ML/DL experimentation.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _dsio() -> None:
    """Reproducible ML/DL experimentation."""


# Register through Typer's public API. The callback above keeps the one-command app as a
# group, so ``dsio run`` remains an explicit subcommand rather than collapsing into the
# command and swallowing ``run`` as its preset argument.
app.command("run")(run_cmd.run)


def main() -> None:
    """Entry point that renders even argument-parse failures as JSON.

    Click runs with ``standalone_mode=False`` so a usage error becomes an envelope rather
    than Click's own text on stderr. A caller parsing stdout should never have to handle
    two output formats.

    Usage errors are detected structurally rather than by exception class, because Typer
    vendors its own fork of Click's internals — ``typer._click.exceptions.MissingParameter``
    is not a ``click.ClickException``, so an isinstance check silently misses it and the
    user gets a traceback. Duck-typing on the interface survives that.
    """
    command = typer.main.get_command(app)
    # MLflow prints a run URL to stdout when a REST-backed run is created. Set this only
    # at actual CLI execution so importing the CLI remains side-effect free while the
    # command's single-JSON-envelope contract stays intact. Overwrite ambient values: a
    # false value would allow the banner to corrupt stdout.
    os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
    try:
        command(standalone_mode=False)
    except Exception as exc:  # noqa: BLE001 - the boundary that renders every error
        if _is_abort(exc):
            emit(failure(KeyboardInterrupt("aborted")))
            raise SystemExit(130) from None
        if _is_usage_error(exc):
            emit(failure(ValueError(exc.format_message())))  # type: ignore[attr-defined]
            raise SystemExit(getattr(exc, "exit_code", 2)) from None
        emit(failure(exc))
        raise SystemExit(1) from None


def _is_usage_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a Click-style user-facing error, from any Click fork."""
    return callable(getattr(exc, "format_message", None))


def _is_abort(exc: BaseException) -> bool:
    return type(exc).__name__ == "Abort" or isinstance(exc, KeyboardInterrupt)


if __name__ == "__main__":
    main()
