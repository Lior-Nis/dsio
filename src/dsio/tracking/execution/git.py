"""Capture reconstructible consumer-repository identity."""

from __future__ import annotations

import subprocess
from pathlib import Path

from dsio.contracts import DsioModel, sha256_of_bytes, sha256_of_file

_GIT_TIMEOUT_SECONDS = 15


class GitState(DsioModel):
    """Git identity of the working tree at run start."""

    sha: str | None = None
    branch: str | None = None
    dirty: bool = False
    code_hash: str | None = None
    patch_sha256: str | None = None

    @property
    def available(self) -> bool:
        return self.sha is not None


def capture_git(cwd: Path | None = None) -> GitState:
    """Capture the commit plus the exact dirty working-tree identity."""
    sha = _git("rev-parse", "HEAD", cwd=cwd)
    if sha is None:
        return GitState()

    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
    status = _git("status", "--porcelain", cwd=cwd) or ""
    if not status:
        return GitState(sha=sha, branch=branch, dirty=False, code_hash=sha)

    patch = working_tree_patch(cwd=cwd)
    digest = sha256_of_bytes(status.encode("utf-8") + b"\0" + patch)
    return GitState(
        sha=sha,
        branch=branch,
        dirty=True,
        code_hash=f"{sha}-dirty-{digest}",
        patch_sha256=sha256_of_bytes(patch),
    )


def working_tree_patch(cwd: Path | None = None) -> bytes:
    """Return a patch containing tracked changes and reconstructible untracked text."""
    parts: list[str] = []
    tracked = _git_raw("diff", "HEAD", cwd=cwd)
    if tracked:
        parts.append(tracked)

    untracked = _git("ls-files", "--others", "--exclude-standard", cwd=cwd)
    root = Path(cwd) if cwd else Path.cwd()
    for name in (untracked or "").splitlines():
        if not name:
            continue
        rendered = _git_raw(
            "diff",
            "--no-index",
            "--",
            "/dev/null",
            name,
            cwd=cwd,
            ok_returncodes=frozenset({0, 1}),
        )
        if rendered and "Binary files" not in rendered:
            parts.append(rendered)
        elif (root / name).is_file():
            digest = sha256_of_file(str(root / name))
            parts.append(f"# untracked binary {name} sha256={digest}\n")
    return "".join(parts).encode("utf-8")


def _git(*args: str, cwd: Path | None = None) -> str | None:
    raw = _git_raw(*args, cwd=cwd)
    return raw.strip() if raw is not None else None


def _git_raw(
    *args: str,
    cwd: Path | None = None,
    ok_returncodes: frozenset[int] = frozenset({0}),
) -> str | None:
    """Return unstripped output so a unified diff remains byte-for-byte applicable."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode in ok_returncodes else None
