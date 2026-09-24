"""Capture process, consumer-repository, dependency, and DSIO package identity."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

from dsio.runs.provenance import capture_env, capture_git, working_tree_patch

_GIT_TIMEOUT_SECONDS = 15
_REDACTED = "<redacted>"


def capture_execution(*, secrets: Collection[str] = ()) -> tuple[dict[str, Any], bytes | None]:
    """Return identity-bearing execution facts and an optional reconstructible patch."""
    cwd = Path.cwd()
    repo_root = _repository_root(cwd)
    project_root = repo_root or cwd
    git = capture_git(cwd=project_root)
    lock = project_root / "uv.lock"
    environment = capture_env(lock_path=lock)
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
    if completed.returncode != 0:
        return None
    rendered = completed.stdout.strip()
    return Path(rendered) if rendered else None


def _package_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
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
                if separator:
                    redacted.append(f"--{option}={_REDACTED}")
                else:
                    redacted.append(token)
                    hide_next = True
                continue
        redacted.append(token)
    return redacted
