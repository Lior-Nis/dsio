"""Seeding for every RNG that can affect a result, with a record of what was set.

Returning the seeds actually applied — rather than just setting them — is what makes the
determinism test meaningful. A seed that is recorded but never wired through produces
identical run records and different numbers, which is the failure this guards against.
"""

from __future__ import annotations

import os
import random

_MAX_SEED = 2**32 - 1


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, int]:
    """Seed Python, NumPy and torch, returning the seeds applied per library.

    ``deterministic`` also pins cuDNN into deterministic mode. That costs throughput and
    is the right default for a system whose headline promise is reproduction; a runner
    that knowingly trades it away can pass ``False``.
    """
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}], got {seed}")

    applied: dict[str, int] = {"python": seed}
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
        applied["numpy"] = seed

    try:
        import torch
    except ImportError:
        return applied

    torch.manual_seed(seed)
    applied["torch"] = seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        applied["cuda"] = seed
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cuDNN flags alone are not enough: several ATen kernels have nondeterministic
        # implementations that ignore them entirely. warn_only keeps a run alive when an
        # op has no deterministic variant, but the warning is the signal that this run's
        # numbers will not reproduce exactly.
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Required for deterministic cuBLAS reductions on CUDA >= 10.2, and it must be
        # set before the first CUDA context is created to have any effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return applied


def dataloader_kwargs(seed: int) -> dict[str, object]:
    """Return the DataLoader arguments needed for a deterministic epoch.

    Seeding the process is not sufficient. Each worker gets its own RNG state, so without
    an explicit generator and ``worker_init_fn`` the shuffle order and any augmentation
    randomness vary with worker count — which makes a result depend on a performance knob.
    """
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return {"generator": generator, "worker_init_fn": _seed_worker}


def _seed_worker(worker_id: int) -> None:
    """Derive each worker's seed from torch's, so worker count does not change results."""
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(worker_seed)
