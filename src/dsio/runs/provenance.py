"""Capture of everything outside the config that determines a run's result.

The governing rule: never block a run, but capture enough that any run can be
reconstructed. A dirty working tree is allowed and tagged; because the diff itself is
stored as an artifact, a dirty run is still exactly reproducible.

The one thing this module refuses to do is guess. When git is unavailable, ``code_hash``
returns ``None`` rather than a confident-but-wrong stamp — a wrong provenance value is
worse than a missing one, because it looks trustworthy.
"""

from __future__ import annotations

import platform
import socket
import subprocess
import sys
from pathlib import Path

from dsio.contracts import DsioModel, sha256_of_bytes, sha256_of_file

_GIT_TIMEOUT_SECONDS = 15


def _git(*args: str, cwd: Path | None = None) -> str | None:
    """Run a git command, returning stripped stdout or ``None`` if git cannot answer."""
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
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


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


def working_tree_patch(cwd: Path | None = None) -> bytes:
    """Return a patch capturing every uncommitted change, including untracked files.

    ``git diff HEAD`` covers tracked modifications; untracked files are appended as
    ``/dev/null`` diffs so the patch alone is sufficient to rebuild the tree state.
    """
    parts: list[str] = []
    tracked = _git("diff", "HEAD", cwd=cwd)
    if tracked:
        parts.append(tracked)

    untracked = _git("ls-files", "--others", "--exclude-standard", cwd=cwd)
    root = Path(cwd) if cwd else Path.cwd()
    for name in (untracked or "").splitlines():
        if not name:
            continue
        rendered = _git("diff", "--no-index", "--", "/dev/null", name, cwd=cwd)
        if rendered:
            parts.append(rendered)
        elif (root / name).is_file():
            # --no-index returns non-zero for binary or unreadable paths; record its
            # digest so the file is at least identified rather than silently dropped.
            digest = sha256_of_file(str(root / name))
            parts.append(f"# untracked binary {name} sha256={digest}")
    return ("\n".join(parts) + "\n").encode("utf-8") if parts else b""


def capture_git(cwd: Path | None = None) -> GitState:
    """Capture the git state of the working tree."""
    sha = _git("rev-parse", "HEAD", cwd=cwd)
    if sha is None:
        return GitState()

    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
    status = _git("status", "--porcelain", cwd=cwd) or ""
    dirty = bool(status)

    if not dirty:
        return GitState(sha=sha, branch=branch, dirty=False, code_hash=sha)

    patch = working_tree_patch(cwd=cwd)
    # Hash the status *and* the patch: status alone misses content edits, and the patch
    # alone misses staged-vs-unstaged distinctions and empty untracked files.
    digest = sha256_of_bytes(status.encode("utf-8") + b"\0" + patch)
    return GitState(
        sha=sha,
        branch=branch,
        dirty=True,
        code_hash=f"{sha}-dirty-{digest}",
        patch_sha256=sha256_of_bytes(patch),
    )


class EnvState(DsioModel):
    """The execution environment, to the depth that affects numerical results."""

    python: str
    platform: str
    hostname: str
    lock_sha256: str | None = None
    torch: str | None = None
    cuda: str | None = None
    cudnn: str | None = None
    gpu: str | None = None


def _torch_versions() -> tuple[str | None, str | None, str | None, str | None]:
    """Return torch, CUDA, cuDNN and GPU strings when torch is installed."""
    try:
        import torch
    except ImportError:
        return None, None, None, None

    cuda_version = getattr(torch.version, "cuda", None)
    cudnn_version: str | None = None
    gpu: str | None = None
    if torch.cuda.is_available():
        raw_cudnn = torch.backends.cudnn.version()
        cudnn_version = str(raw_cudnn) if raw_cudnn is not None else None
        gpu = torch.cuda.get_device_name(0)
    return torch.__version__, cuda_version, cudnn_version, gpu


def capture_env(lock_path: Path | None = None) -> EnvState:
    """Capture the interpreter, platform, and dependency lock identity."""
    lock = lock_path if lock_path is not None else Path("uv.lock")
    lock_sha = sha256_of_file(str(lock)) if lock.is_file() else None
    torch_version, cuda, cudnn, gpu = _torch_versions()
    return EnvState(
        python=sys.version.split()[0],
        platform=platform.platform(),
        hostname=socket.gethostname(),
        lock_sha256=lock_sha,
        torch=torch_version,
        cuda=cuda,
        cudnn=cudnn,
        gpu=gpu,
    )
