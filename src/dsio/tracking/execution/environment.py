"""Capture the dependency and hardware environment that can affect numerical results."""

from __future__ import annotations

import platform
import socket
import sys
from pathlib import Path

from dsio.contracts import DsioModel, sha256_of_file


class EnvironmentState(DsioModel):
    """The execution environment recorded with experiment evidence."""

    python: str
    platform: str
    hostname: str
    lock_sha256: str | None = None
    torch: str | None = None
    cuda: str | None = None
    cudnn: str | None = None
    gpu: str | None = None
    #: Retained so previously recorded provenance documents remain readable.
    extra: str | None = None


def capture_environment(lock_path: Path | None = None) -> EnvironmentState:
    """Capture the interpreter, platform, hardware, and dependency-lock identity."""
    lock = lock_path if lock_path is not None else Path("uv.lock")
    lock_sha = sha256_of_file(str(lock)) if lock.is_file() else None
    torch_version, cuda, cudnn, gpu = _torch_versions()
    return EnvironmentState(
        python=sys.version.split()[0],
        platform=platform.platform(),
        hostname=socket.gethostname(),
        lock_sha256=lock_sha,
        torch=torch_version,
        cuda=cuda,
        cudnn=cudnn,
        gpu=gpu,
    )


def _torch_versions() -> tuple[str | None, str | None, str | None, str | None]:
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
