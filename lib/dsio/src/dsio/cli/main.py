"""The composition root.

Sub-apps are mounted here and nowhere else, so no command module imports a sibling. The
rule is a linter contract because command modules that reach into each other's private
helpers are how gating logic drifts apart between subcommands.
"""

from __future__ import annotations

import typer

from dsio.cli import (
    artifacts_cmd,
    data_cmd,
    eval_cmd,
    run_cmd,
    runs_cmd,
    splits_cmd,
)
from dsio.cli.envelope import emit, failure

app = typer.Typer(
    name="dsio",
    help="Reproducible ML/DL experimentation.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(runs_cmd.app, name="runs")
app.add_typer(artifacts_cmd.app, name="models")
app.add_typer(data_cmd.app, name="data")
app.add_typer(splits_cmd.app, name="splits")
app.add_typer(eval_cmd.app, name="eval")
# run_cmd's commands sit at the top level: `dsio run`, `dsio presets`.
app.registered_commands.extend(run_cmd.app.registered_commands)


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
