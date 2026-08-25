"""The MLflow reachability guard: decision 7's "a run fails without MLflow", tested in
isolation from any actual training.

Most tests below are **unit** (no MLflow at all -- a closed port or a bogus host fails
deterministically without a server on the other end); one is **fake-backed**
(``build_mlflow_logger`` against a `file:` URI, per ``tests/conftest.py``'s isolation);
one is explicitly marked ``live`` (needs `docker compose up -d`; compose.yaml -- excluded
by default, see ``pyproject.toml``'s ``addopts``). Proving the logger actually *streams
metrics* into a backend belongs with the runner tests that produce something to log
(``tests/train/test_torch_runner.py``, ``tests/train/test_ssl_runner.py``); this file only
proves the logger is *built* the way decision 7 and 8 say it must be.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from dsio.train.tracking import (  # noqa: E402
    DEFAULT_TRACKING_URI,
    TRACKING_URI_ENV,
    MlflowUnavailableError,
    build_mlflow_logger,
    finite_metrics,
    require_mlflow,
    resolve_tracking_uri,
)

# A closed local port: nothing binds this by construction (the compose stack, when it
# runs, uses 5000), so a connection to it is refused immediately rather than merely
# unanswered -- the same "closed port" case the plan's Task 2 verification names.
DEAD_PORT_URI = "http://localhost:59999"
BOGUS_HOST_URI = "http://mlflow.invalid.example.test:5000"

# Generous relative to the probe's own 5-second bound (`dsio.train.tracking.
# _PROBE_TIMEOUT_SECONDS`), tight relative to MLflow's un-bounded default (120s timeout,
# 7 retries) -- enough slack for a loaded CI box, not enough to hide a regression back to
# the unbounded default.
BOUNDED_SECONDS = 15.0


# --- resolving the URI -----------------------------------------------------------------


def test_resolve_tracking_uri_defaults_to_the_compose_stacks_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TRACKING_URI_ENV, raising=False)
    assert resolve_tracking_uri() == DEFAULT_TRACKING_URI == "http://localhost:5000"


def test_resolve_tracking_uri_prefers_the_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TRACKING_URI_ENV, "http://tracking.example:9000")
    assert resolve_tracking_uri() == "http://tracking.example:9000"


# --- local backends are never probed ----------------------------------------------------


def test_a_local_backend_is_accepted_without_touching_the_network() -> None:
    """`file:` URIs are the fake-backed tests' escape hatch, and have no server to be
    down. An unreachable-looking path is deliberate: if this were probed like an
    http(s) URI, it would have to fail (there is nothing listening at a filesystem
    path), so the only way this passes is if no probe happens at all."""
    uri = "file:/no/such/directory/mlruns"
    assert require_mlflow(uri) == uri


# --- the logger is built the way decisions 7 and 8 say it must be -----------------------


def test_build_mlflow_logger_disables_model_logging(tmp_path: Path) -> None:
    """``log_model=False`` is the spec's deliberate default (decision 7's table): MLflow's
    own model logging would duplicate ``dsio.artifacts.store.ModelRegistry``'s
    digest-on-save / fail-closed-on-load policy layer with a mechanism that has neither.
    Fake-backed: a real ``MLFlowLogger`` against a `file:` URI, no server."""
    config = SimpleNamespace(name="probe-experiment")
    run = SimpleNamespace(run_id="run-123")
    uri = f"file:{tmp_path}/mlruns"

    logger = build_mlflow_logger(config, run, uri)  # type: ignore[arg-type]

    assert logger._log_model is False
    assert logger._experiment_name == "probe-experiment"
    assert logger._run_name == "run-123"
    assert logger._tracking_uri == uri


# --- metrics filtering: rehomed from the deleted `Run.log_metrics` ----------------------


def test_finite_metrics_drops_non_finite_values() -> None:
    """The property ``RunLedger``'s old ``Run.log_metrics`` used to guard: a NaN or inf
    in a metric stream breaks every downstream comparison, so it is dropped rather than
    logged. Rehomed here because metrics are logged straight to MLflow now
    (``dsio.train.torch_task.run_torch``, ``dsio.train.ssl_task.run_ssl_pretrain``), not
    through ``Run`` -- see ``tests/runs/test_runs.py``'s note on the deleted test."""
    finite = finite_metrics({"good": 1.0, "bad": float("nan"), "worse": float("inf")})
    assert finite == {"good": 1.0}


def test_finite_metrics_keeps_every_finite_value() -> None:
    """The acceptance half: a guard that drops non-finite values must not also drop
    ordinary ones, including a legitimate zero."""
    finite = finite_metrics({"accuracy": 0.875, "loss": 0.0, "auc": 1.0})
    assert finite == {"accuracy": 0.875, "loss": 0.0, "auc": 1.0}


# --- unreachable http(s) fails, fast, with a useful message -----------------------------


def test_a_closed_port_is_refused() -> None:
    with pytest.raises(MlflowUnavailableError, match=r"http://localhost:59999"):
        require_mlflow(DEAD_PORT_URI)


def test_a_bogus_host_is_refused() -> None:
    with pytest.raises(MlflowUnavailableError):
        require_mlflow(BOGUS_HOST_URI)


def test_the_message_names_the_uri_and_how_to_start_the_stack() -> None:
    """A fresh clone with no stack running has to be told what to do next, not just that
    something failed."""
    with pytest.raises(MlflowUnavailableError, match=r"docker compose up -d") as excinfo:
        require_mlflow(DEAD_PORT_URI)
    assert DEAD_PORT_URI in str(excinfo.value)
    assert "MLFLOW_TRACKING_URI" in str(excinfo.value)


def test_an_unreachable_server_fails_in_bounded_time() -> None:
    """MLflow's own client defaults to a 120-second timeout and 7 retries
    (`mlflow.environment_variables`); unbounded, "the stack is down" would take minutes
    to report. `require_mlflow` must not inherit that default."""
    start = time.monotonic()
    with pytest.raises(MlflowUnavailableError):
        require_mlflow(DEAD_PORT_URI)
    assert time.monotonic() - start < BOUNDED_SECONDS


def test_the_probes_timeout_override_is_restored_after_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded probe timeout/retry budget is a global environment mutation for the
    duration of one call; a real run's later logging calls must get MLflow's normal
    (generous) retry behaviour back, not the probe's tight one."""
    import os

    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "77")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")
    with pytest.raises(MlflowUnavailableError):
        require_mlflow(DEAD_PORT_URI)
    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "77"
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "3"


# --- live: the real compose stack -------------------------------------------------------


@pytest.mark.live
def test_require_mlflow_succeeds_against_the_running_compose_stack() -> None:
    """Needs `docker compose up -d` (compose.yaml). Excluded by default; run with
    `uv run --extra cpu pytest -m live`."""
    assert require_mlflow("http://localhost:5000") == "http://localhost:5000"


@pytest.mark.live
def test_the_live_stack_actually_stores_and_serves_back_an_artifact(tmp_path: Path) -> None:
    """Bug 1's own regression test, at the only level that is honest about what broke:
    a health check (`docker compose ps`, `curl /health`) and even
    `test_require_mlflow_succeeds_against_the_running_compose_stack` above both pass
    against a stack that cannot store a single artifact -- neither exercises an artifact
    write at all. `compose.yaml` used to pass `--default-artifact-root /artifacts`, a
    plain filesystem path; MLflow only proxies artifact writes over HTTP (the
    `mlflow-artifacts:/` scheme) when that flag is left unset with `--serve-artifacts`
    enabled (see `compose.yaml`'s own comment for the full argument, and MLflow's
    `mlflow server --help`), so a bare path made every client resolve it *locally* and
    try to write to a directory that only exists inside the `mlflow` container. This logs
    a real artifact to a freshly-created experiment against the running server (a new
    name every run, via `tmp_path`, so it always gets the proxy scheme MLflow assigns to
    experiments created *after* the fix -- see `compose.yaml`'s comment on why
    experiments created under the old flag are not retroactively repaired), reads it back
    over HTTP, and checks the bytes actually round-tripped -- not merely that the call
    didn't raise.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient("http://localhost:5000")
    experiment_name = f"live-artifact-roundtrip-{time.time_ns()}"
    experiment_id = client.create_experiment(experiment_name)
    experiment = client.get_experiment(experiment_id)
    # The proxy scheme, not a bare filesystem path -- this is the assertion that would
    # have caught Bug 1 directly, before ever attempting a write.
    assert experiment.artifact_location.startswith("mlflow-artifacts:/"), (
        f"experiment artifact_location {experiment.artifact_location!r} is not proxied; "
        "a client would try to write it directly, which is exactly Bug 1"
    )

    run = client.create_run(experiment_id)
    run_id = run.info.run_id
    payload = b"dsio live artifact round-trip probe\n"
    local_file = tmp_path / "probe.txt"
    local_file.write_bytes(payload)
    try:
        client.log_artifact(run_id, str(local_file))
        downloaded = Path(client.download_artifacts(run_id, "probe.txt"))
        assert downloaded.read_bytes() == payload
    finally:
        client.set_terminated(run_id, "FINISHED")
