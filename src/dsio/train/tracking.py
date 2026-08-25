"""MLflow as a run's hard dependency (decision 7 of the lean design).

*"A component a run cannot write to is not a sink — it is a broken dependency."* If MLflow
is unreachable, the run fails, and it fails **before** any expensive work: before the store
is opened, the window index built, the module constructed or a ``Trainer`` created.

``require_mlflow`` is called in two places per task kind, the same way ``require_fold``
(``dsio.splits.folds``) is: once from the preflight (``check_torch``/``check_ssl`` in
``torch_task.py``/``ssl_task.py``), which is all the CLI's ``_preflight`` reaches, and again
at the very top of the runner itself (``run_torch``/``run_ssl_pretrain``), which is what
every test and any non-CLI caller of ``execute()`` reaches. ``SslPretrainTask``'s fold check
used to run only after the store had loaded; this guard must not reproduce that bug for
MLflow, so it fires before ``task = config.task`` is even read where that costs nothing.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from lightning.pytorch.loggers import MLFlowLogger
from mlflow.environment_variables import (
    MLFLOW_HTTP_REQUEST_MAX_RETRIES,
    MLFLOW_HTTP_REQUEST_TIMEOUT,
)
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

if TYPE_CHECKING:
    from dsio.config.schema import RunConfig
    from dsio.runs.record import Run

# The same environment variable MLflow's own client and Lightning's `MLFlowLogger` already
# read (`MLFlowLogger.__init__`'s own default is `os.getenv("MLFLOW_TRACKING_URI")`), so
# this is never a second source of truth that could disagree with them.
TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"

# compose.yaml (decision 8) publishes the server on this address; a fresh clone with the
# stack running needs no configuration at all.
DEFAULT_TRACKING_URI = "http://localhost:5000"

# The probe must fail in seconds, not minutes. MLflow's own client defaults to a
# 120-second timeout and 7 retries with exponential backoff
# (`mlflow.environment_variables`), which turns "the stack is down" into a multi-minute
# hang before the run ever reports the failure it exists to report early. Bounded here,
# for the probe only, and restored immediately after -- a run that does reach MLflow still
# gets MLflow's own (generous) retry behaviour for transient blips during real logging.
_PROBE_TIMEOUT_SECONDS = "5"
_PROBE_MAX_RETRIES = "0"


class MlflowUnavailableError(RuntimeError):
    """MLflow's tracking server could not be reached; the run has not started."""


def resolve_tracking_uri() -> str:
    """The tracking URI a run logs to: ``MLFLOW_TRACKING_URI``, or the local default."""
    return os.environ.get(TRACKING_URI_ENV) or DEFAULT_TRACKING_URI


@contextmanager
def _bounded_probe_timeout() -> Iterator[None]:
    """Shrink MLflow's HTTP timeout/retry budget for the probe call only."""
    names = (MLFLOW_HTTP_REQUEST_TIMEOUT.name, MLFLOW_HTTP_REQUEST_MAX_RETRIES.name)
    saved = {name: os.environ.get(name) for name in names}
    os.environ[MLFLOW_HTTP_REQUEST_TIMEOUT.name] = _PROBE_TIMEOUT_SECONDS
    os.environ[MLFLOW_HTTP_REQUEST_MAX_RETRIES.name] = _PROBE_MAX_RETRIES
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def require_mlflow(tracking_uri: str | None = None) -> str:
    """Fail now, with a clear message, if ``tracking_uri`` cannot be reached.

    Only ``http(s)://`` URIs are probed over the network: a local backend (``file:``, a
    bare path) has no server to be down, and MLflow's own client rejects any other scheme
    outright the moment it is used, so there is nothing this check would add for those.
    Local backends are the escape hatch the "fake-backed" tests use to stay Docker-free,
    not what decision 8's reachability guarantee is about.

    The probe calls ``search_experiments`` -- the same request ``MLFlowLogger.experiment``
    issues to resolve or create an experiment before the first real write -- rather than a
    liveness route like ``/health``, which a server can answer while its backend store
    (Postgres, per decision 8) is unreachable. A component a run cannot *write* to is not a
    sink, so the check has to reach the store, not merely the process in front of it.
    """
    uri = tracking_uri if tracking_uri is not None else resolve_tracking_uri()
    if not (uri.startswith("http://") or uri.startswith("https://")):
        return uri

    with _bounded_probe_timeout():
        try:
            MlflowClient(uri).search_experiments(max_results=1)
        except MlflowException as error:
            raise MlflowUnavailableError(
                f"MLflow at {uri!r} is unreachable: {error}. Decision 7 of the lean "
                "design: a run that cannot write to MLflow does not start. Start the "
                "local stack with `docker compose up -d` (compose.yaml), or point "
                f"{TRACKING_URI_ENV} at a reachable tracking server."
            ) from error
    return uri


#: MLflow's own cap on a param value's length (`mlflow.utils.validation.
#: MAX_PARAM_VAL_LENGTH` is more generous in recent servers, but 250 is the number the
#: spec's own table names, and truncating consistently is what "for search only" means:
#: the config artifact, not the params table, is what recovers a value exactly).
_MAX_PARAM_VALUE_LENGTH = 250


def build_mlflow_logger(config: RunConfig, run: Run, tracking_uri: str) -> MLFlowLogger:
    """The Lightning logger a runner's ``Trainer`` streams ``self.log(...)`` metrics through.

    ``log_model=False`` per the spec's deliberate default: MLflow's own model logging would
    duplicate ``dsio.artifacts.store.ModelRegistry``'s digest-on-save / fail-closed-on-load
    policy layer with a mechanism that has neither. ``experiment_name=config.name`` is what
    makes decision 8's "a shell loop over 5 folds produces 5 runs in one experiment" true
    natively: ``name`` already labels a run and groups seeds (see ``RunConfig.name``), and an
    MLflow experiment is the grouping unit that mirrors it. ``run_name=run.run_id`` ties the
    MLflow run back to this process's own run id, so the two are cross-referenced without a
    lookup table.
    """
    return MLFlowLogger(
        experiment_name=config.name,
        run_name=run.run_id,
        tracking_uri=tracking_uri,
        log_model=False,
    )


def finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Drop non-finite values before they reach MLflow.

    ``RunLedger``'s old ``Run.log_metrics`` filtered these for the same reason this does:
    a NaN or inf in a metric stream breaks every downstream comparison, and silently
    poisoning a run's record is worse than a gap. Lightning's own per-step ``self.log(...)``
    calls never went through that filter (they flow straight from the training loop to
    ``mlflow_logger.log_metrics``, not through ``Run``), so this only ever needs to guard
    the held-out metrics a runner computes explicitly and logs itself
    (``dsio.train.torch_task.run_torch``, ``dsio.train.ssl_task.run_ssl_pretrain``) --
    which is exactly where callers of this function are.
    """
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float)
        and float(value) == float(value)
        and float(value) not in (float("inf"), float("-inf"))
    }


def stamp_provenance(run: Run, mlflow_logger: MLFlowLogger) -> None:
    """Log everything ADR 0002 said MLflow could not hold, as decision 7's own table says
    to: the dirty diff and the reproduce script as artifacts, the resolved config as an
    artifact (params cap at 250 characters; the file is what recovers it exactly), and
    ``config_hash`` as a tag so a run's identity is searchable, not just recoverable.

    Called first thing inside a runner, right after ``build_mlflow_logger`` -- before the
    store, a module or a ``Trainer`` exist -- so a run's provenance lands in MLflow before
    any expensive work happens, the same guarantee ``RunLedger.start`` used to give by
    writing ``run.json`` to disk first. Accessing ``mlflow_logger.run_id`` here is also
    what actually creates the MLflow run (it is created lazily, on first access): calling
    this early is what makes that happen early too, not merely what happens to log early.
    """
    client = mlflow_logger.experiment
    mlflow_run_id = mlflow_logger.run_id
    run.mlflow_run_id = mlflow_run_id

    client.set_tag(mlflow_run_id, "config_hash", run.record.config_hash)
    if run.record.fold is not None:
        client.set_tag(mlflow_run_id, "fold", str(run.record.fold))

    for path in run.provenance_files:
        client.log_artifact(mlflow_run_id, str(path))

    for key, value in _flatten_params(run.record.config).items():
        client.log_param(mlflow_run_id, key, value)


def log_run_artifacts(run: Run, mlflow_logger: MLFlowLogger) -> None:
    """Upload whatever a runner wrote into ``run.artifacts_dir`` -- predictions,
    checkpoints, an exported encoder -- to the same MLflow run its metrics landed in.

    ``run.artifacts_dir`` is local scratch space (see ``dsio.runs.record``'s module
    docstring): this is the point at which those files actually become MLflow artifacts,
    called once a runner has finished writing them, at the end of its work.
    """
    directory = run.artifacts_dir
    if any(directory.iterdir()):
        mlflow_logger.experiment.log_artifacts(mlflow_logger.run_id, str(directory))


def _flatten_params(config: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten scalar leaves of a nested config into dotted keys for MLflow's params table.

    Params exist for *search* only (decision 7's table) -- filtering runs by
    ``task.lr`` in the UI -- not for reconstructing the config, which is what
    ``config.resolved.yaml`` (logged as an artifact, byte-exact) is for. A sequence has
    no single scalar representation worth filtering on, so it is skipped rather than
    mangled into a lossy string.
    """
    flat: dict[str, str] = {}
    for key, value in config.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(value, dotted))
        elif isinstance(value, list | tuple):
            continue
        else:
            flat[dotted] = str(value)[:_MAX_PARAM_VALUE_LENGTH]
    return flat
