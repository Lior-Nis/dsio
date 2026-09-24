"""Compose process, repository, environment, and installed-package identity."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

from dsio.tracking.execution.environment import capture_environment
from dsio.tracking.execution.git import capture_git, working_tree_patch

_GIT_TIMEOUT_SECONDS = 15
_REDACTED = "<redacted>"


def capture_execution(*, secrets: Collection[str] = ()) -> tuple[dict[str, Any], bytes | None]:
    """Return identity-bearing execution facts and an optional reconstructible patch."""
    cwd = Path.cwd()
    project_root = _repository_root(cwd) or cwd
    git = capture_git(cwd=project_root)
    environment = capture_environment(lock_path=project_root / "uv.lock")
    patch = working_tree_patch(cwd=project_root) if git.dirty else None
    return (
        {
            "command": _redact_command(tuple(sys.orig_argv), secrets),
            "environment": environment.model_dump(mode="json"),
            "git": git.model_dump(mode="json"),
            "package_sha256": _package_digest(),
        },
        patch,
    )


def _repository_root(cwd: Path) -> Path | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    rendered = completed.stdout.strip()
    return Path(rendered) if completed.returncode == 0 and rendered else None


def _package_digest() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _redact_command(command: Sequence[str], secrets: Collection[str]) -> list[str]:
    names = {name.replace("_", "-") for name in secrets}
    redacted: list[str] = []
    hide_next = False
    for token in command:
        if hide_next:
            redacted.append(_REDACTED)
            hide_next = False
            continue
        if token.startswith("--"):
            option, separator, _ = token[2:].partition("=")
            if option in names:
                redacted.append(f"--{option}={_REDACTED}" if separator else token)
                hide_next = not separator
                continue
        redacted.append(token)
    return redacted
