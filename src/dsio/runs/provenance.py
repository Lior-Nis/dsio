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
    raw = _git_raw(*args, cwd=cwd)
    return raw.strip() if raw is not None else None


def _git_raw(
    *args: str, cwd: Path | None = None, ok_returncodes: frozenset[int] = frozenset({0})
) -> str | None:
    """Run a git command, returning *unstripped* stdout, or ``None`` if git cannot answer.

    Diff output must never be passed through ``str.strip()``: a unified diff's blank
    context line is a line containing a single space, and stripping the whole output
    eats that trailing space, silently shortening the final hunk by one line relative to
    its own ``@@ -a,b +c,d @@`` header. ``git apply`` then rejects the patch with
    "corrupt patch" -- which is exactly bug C1, caught because most Python edits end on a
    blank line (the gap between two ``def``s). Callers that build patches
    (:func:`working_tree_patch`) must use this, not :func:`_git`.

    ``ok_returncodes`` widens what counts as success: ``git diff --no-index`` exits ``1``
    whenever the compared paths differ, which for an untracked file (diffed against
    ``/dev/null``) is *always* -- so callers that want that diff's content must accept
    ``1`` as success too, not just ``0``.
    """
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
    if completed.returncode not in ok_returncodes:
        return None
    return completed.stdout


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

    Built entirely from :func:`_git_raw`, never :func:`_git`: stripping a diff's stdout
    corrupts it (see :func:`_git_raw`'s docstring), and this is the one value in the
    module that ``git apply`` later has to parse back byte-for-byte.
    """
    parts: list[str] = []
    tracked = _git_raw("diff", "HEAD", cwd=cwd)
    if tracked:
        parts.append(tracked)

    untracked = _git("ls-files", "--others", "--exclude-standard", cwd=cwd)
    root = Path(cwd) if cwd else Path.cwd()
    for name in (untracked or "").splitlines():
        if not name:
            continue
        # `--no-index` exits 1 whenever the two sides differ, which for a brand-new file
        # diffed against `/dev/null` is *always* -- so `1` must count as success here, or
        # every untracked file's content silently falls through to the sha256 stand-in
        # below instead of actually being captured.
        rendered = _git_raw(
            "diff", "--no-index", "--", "/dev/null", name, cwd=cwd, ok_returncodes=frozenset({0, 1})
        )
        if rendered and "Binary files" not in rendered:
            parts.append(rendered)
        elif (root / name).is_file():
            # Binary content, or a path `--no-index` could not read at all: record its
            # digest so the file is at least identified rather than silently dropped.
            digest = sha256_of_file(str(root / name))
            parts.append(f"# untracked binary {name} sha256={digest}\n")
    return "".join(parts).encode("utf-8")


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
    #: Which of `pyproject.toml`'s conflicting `cpu`/`gpu` extras (`dsio.runs.record`'s
    #: module docstring, "torch and lightning only in the cpu/gpu extras") this run's
    #: environment was installed with -- ``None`` when that cannot be determined (no
    #: torch, or a torch build this heuristic does not recognize). Captured here, at run
    #: time, rather than re-derived later by whatever `reproduce.sh` finds installed on
    #: a possibly different machine: this is the one machine that is *known* to have
    #: produced the run's metrics, so this is the only trustworthy place to read it from.
    extra: str | None = None


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


def _detect_extra(torch_version: str | None) -> str | None:
    """Which `cpu`/`gpu` extra installed this ``torch``, from its local version suffix.

    `pyproject.toml`'s `[tool.uv.sources]` pins the `cpu` extra to the `pytorch-cpu`
    index, whose wheels carry a `+cpu` local version (e.g. `2.13.0+cpu`); the `gpu`
    extra takes the default PyPI wheel instead, which is CUDA-enabled and carries a
    `+cu...` local version (e.g. `2.13.0+cu130`) -- see that file's "No index is
    declared for the gpu extra" comment. `cpu` and `gpu` are declared `conflicts` in
    `pyproject.toml`, so at most one is ever installed; anything else (no local version
    segment at all, as on a platform with no separate accelerator wheel) is honestly
    unrecognized rather than guessed at.
    """
    if torch_version is None:
        return None
    local = torch_version.split("+", 1)[1] if "+" in torch_version else ""
    if local == "cpu":
        return "cpu"
    if local.startswith("cu"):
        return "gpu"
    return None


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
        extra=_detect_extra(torch_version),
    )
