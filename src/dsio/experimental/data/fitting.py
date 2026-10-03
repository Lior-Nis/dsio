"""Statistics fitted on one split role, recorded as evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dsio.data.splits.models import SplitError, SplitFile
from dsio.data.store import SignalStore


def fit_standardization(
    store: SignalStore,
    manifest: SplitFile,
    *,
    partition: int,
    observed: bool = False,
) -> dict[str, Any]:
    """Fit per-feature mean and scale over the samples assigned to training.

    The manifest's store binding and training assignment are validated before any sample
    is read. The training role is the first role declared by the manifest and cannot be
    overridden by the caller.

    Consumes:
        A ``SignalStore`` of ``[rows, features]`` samples and the replayed split manifest.

    Produces:
        ``{"mean", "scale", "role", "partition", "sample_ids", "observed"}``: float64
        statistics over every row of the training samples (``scale`` is the population standard
        deviation, with zero-variance features mapped to 1) and the exact identities used.

    Parameters:
        ``partition``: split partition index; ``observed`` (default ``false``): ignore NaN
        values per feature instead of propagating them.

    Devices:
        CPU (NumPy); fitted once before training.

    Limitations:
        Loads the role's samples into memory together; a feature with no observed value
        raises rather than inventing statistics.

    Example:
        >>> from pathlib import Path
        >>> from tempfile import mkdtemp
        >>> import numpy as np
        >>> from dsio.data.adapters import entity_examples
        >>> from dsio.data.splits import generate
        >>> path = Path(mkdtemp()) / "store"
        >>> with SignalStore.builder(path, channels=1) as builder:
        ...     for index in range(4):
        ...         _ = builder.add(f"s{index}", np.array([[float(index)]]), group=f"g{index}")
        >>> store = SignalStore(path)
        >>> examples = entity_examples(store)
        >>> manifest = generate(
        ...     examples, "group_shuffle", name="demo", seed=1,
        ...     parameters={"test_size": 0.25},
        ... )
        >>> fitted = fit_standardization(store, manifest, partition=0)
        >>> role = manifest.required_roles[0]
        >>> sorted(fitted["sample_ids"]) == sorted(manifest.fold(0).assignments[role])
        True
    """
    store_name = str(store.path.name)
    if manifest.store != store_name:
        raise SplitError(
            f"split {manifest.name!r} was built for dataset {manifest.store!r}, not {store_name!r}"
        )
    store_digest = store.identity
    if manifest.store_manifest_sha256 != store_digest:
        raise SplitError(
            f"split {manifest.name!r} was built for dataset digest "
            f"{manifest.store_manifest_sha256!r}, not {store_digest!r}"
        )
    if not manifest.required_roles:
        raise SplitError(f"split {manifest.name!r} declares no training role")
    role = manifest.required_roles[0]
    assignments = manifest.fold(partition).assignments
    try:
        sample_ids = list(assignments[role])
    except KeyError:
        raise SplitError(
            f"split {manifest.name!r} partition {partition} is missing declared training "
            f"role {role!r}"
        ) from None
    if not sample_ids:
        raise ValueError(
            f"split training role {role!r} in partition {partition} assigns no samples"
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise SplitError(
            f"split training role {role!r} in partition {partition} repeats a sample identity"
        )
    store_ids = {entity.entity_id for entity in store.entities}
    unknown = sorted(set(sample_ids) - store_ids)
    if unknown:
        raise SplitError(
            f"split training role {role!r} in partition {partition} names unknown sample "
            f"identity {unknown[0]!r}"
        )
    values = np.concatenate(
        [store.read_sample(sample_id)["data"] for sample_id in sample_ids], axis=0
    ).astype(np.float64)
    if observed:
        if bool(np.isnan(values).all(axis=0).any()):
            missing = [int(index) for index in np.flatnonzero(np.isnan(values).all(axis=0))]
            raise ValueError(f"features {missing} have no observed value in role {role!r}")
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
    else:
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "role": role,
        "partition": partition,
        "sample_ids": sample_ids,
        "observed": observed,
    }


def record_fitted(run_id: str, name: str, fitted: Mapping[str, Any]) -> str:
    """Log fitted values as an MLflow evidence artifact and return its run-relative path.

    The path is the same for every replay, so it can enter execution identity; the
    values themselves live in the artifact on the attempt's own run.

    Consumes:
        An active MLflow run ID and a JSON-serializable fitted mapping, e.g. from
        :func:`fit_standardization`.

    Produces:
        ``fitted/<name>.json`` on the run; returns that path.

    Parameters:
        ``run_id``; ``name`` (a plain file stem); ``fitted``.

    Devices:
        Device-independent.

    Limitations:
        Requires a writable MLflow run.

    Example:
        >>> from mlflow import MlflowClient
        >>> client = MlflowClient()
        >>> run = client.create_run(client.create_experiment("fitted-example"))
        >>> record_fitted(run.info.run_id, "standardization", {"mean": [0.0], "scale": [1.0]})
        'fitted/standardization.json'
    """
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"invalid fitted artifact name {name!r}")
    from mlflow import MlflowClient

    path = f"fitted/{name}.json"
    MlflowClient().log_dict(run_id, dict(fitted), path)
    return path


__all__ = ["fit_standardization", "record_fitted"]
