"""
The shared, calibration-specific MLflow adapter (item #5, Phase C).

Domain objects (CalibrationReport-style data) -> this adapter -> MLflow.
If MLflow is ever replaced, only this file changes - callers never touch
the MLflow API directly.

## Explicit invariant (Phase A4)
MLflow is OBSERVATIONAL, never OPERATIONAL. If logging fails for any
reason - MLflow missing, a bad value, a filesystem issue - the calling
script prints a warning and continues exactly as if this function had
never been called. A real, multi-hour calibration run must never be
interrupted by a logging problem.

## Explicit tracking URI, every call (Phase C0's real, concrete finding)
Phase C0's manual spike confirmed a real, live discrepancy: `mlflow ui`
and the client library can each independently fall back to a DIFFERENT
default backing store (SQLite `mlflow.db` vs. file-based `mlruns/`) when
neither is given an explicit URI - the data silently ends up in two
different places depending on which command happened to run first. This
adapter ALWAYS sets the tracking URI explicitly, every call, rather than
relying on either side's version-dependent default.

## C0.5 addition: log_fleet_run() (see conversation)
A SEPARATE function from log_calibration_run(), not new optional kwargs
bolted onto it -- log_calibration_run() is explicitly calibration-shaped
(its required vehicle_spec_path, its RMSE-shaped metric examples), and
none of Track C's actual multi-agent scripts call it at all. log_fleet_run()
mirrors its exact discipline (observational-not-operational, explicit
tracking URI every call, same caught-exception safety) for the genuinely
different run type C0.1-C0.4 produced. Both functions share the SAME
tracking URI/experiment name, so fleet and calibration runs live in one
searchable MLflow store, distinguished by their own schema fields
(mlflow_schema.FLEET_REQUIRED_FIELDS vs. REQUIRED_FIELDS) rather than by
being split into separate experiments.
"""
from __future__ import annotations

import platform
import socket
import sys
from importlib.metadata import version as _pkg_version
from typing import Optional

import mlflow_schema as schema

# Phase A2's original design (file:./mlruns) was WRONG - real evidence
# from Phase C's actual test run showed MLflow 3.15.1 has deprecated the
# plain file-store backend (raises a real exception, 'in maintenance
# mode' unless explicitly opted into). Phase C0's spike run succeeded
# specifically BECAUSE it never explicitly requested the file store -
# MLflow's own real, current default is already SQLite. Corrected here
# to match that confirmed-working reality, not the original assumption.
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "tryton_calibration"


def _safe_pkg_version(pkg_name: str) -> str:
    """Never raises - returns 'unknown' rather than crash the whole
    logging call over one missing/oddly-named optional dependency."""
    try:
        return _pkg_version(pkg_name)
    except Exception:
        return "unknown"


def log_calibration_run(
    *,
    script: str,
    run_name: str,
    status: str,
    status_reason: str,
    git_commit: Optional[str],
    vehicle_spec_path: str,
    params: Optional[dict] = None,
    metrics: Optional[dict] = None,
    artifacts: Optional[list] = None,
    tags: Optional[dict] = None,
) -> None:
    """
    Logs one real calibration run to MLflow. NEVER raises - any real
    exception is caught, printed as a warning, and execution returns
    normally, per the observational-not-operational invariant (A4).

    Required (C1 - named parameters, so a missing one is an immediate,
    real TypeError at the call site, not a silent gap):
        script, run_name, status, status_reason, git_commit, vehicle_spec_path

    Optional (dicts - callers include only what's relevant to them):
        params: extra parameters beyond what this function logs automatically
        metrics: numeric results (e.g. rmse_total, rmse_u, runtime_seconds)
        artifacts: list of real file paths to attach to the run
        tags: extra tags beyond script/git_commit/hostname/run_type/status/status_reason
    """
    try:
        schema.validate_status(status)  # raises ValueError on a typo'd status -
        # caught by the outer try/except below, same as any other logging
        # failure - a bad status value should never crash the real script.

        import mlflow

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=run_name):
            # --- Always-logged parameters (C3): schema version + real,
            # live-queried package versions - callers never supply these.
            mlflow.log_param("mlflow_schema_version", schema.SCHEMA_VERSION)
            mlflow.log_param("python_version", sys.version.split()[0])
            mlflow.log_param("numpy_version", _safe_pkg_version("numpy"))
            mlflow.log_param("scipy_version", _safe_pkg_version("scipy"))
            mlflow.log_param("stable_baselines3_version", _safe_pkg_version("stable-baselines3"))
            mlflow.log_param("mlflow_version", _safe_pkg_version("mlflow"))
            mlflow.log_param("vehicle_spec_path", vehicle_spec_path)

            # --- Caller-supplied parameters
            if params:
                for key, value in params.items():
                    mlflow.log_param(key, value)

            # --- Caller-supplied metrics
            if metrics:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

            # --- Always-logged tags
            mlflow.set_tag("script", script)
            mlflow.set_tag("git_commit", git_commit or "unknown")
            mlflow.set_tag("hostname", socket.gethostname())
            mlflow.set_tag("status", status)
            mlflow.set_tag("status_reason", status_reason)

            # --- Caller-supplied tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)

            # --- Artifacts
            if artifacts:
                for path in artifacts:
                    try:
                        mlflow.log_artifact(path)
                    except Exception as e:
                        print(f"[mlflow_logging] WARNING - could not log artifact "
                              f"{path!r}: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"[mlflow_logging] WARNING - experiment tracking failed, "
              f"continuing without it: {type(e).__name__}: {e}")


def log_fleet_run(
    *,
    script: str,
    run_name: str,
    status: str,
    status_reason: str,
    git_commit: Optional[str],
    vehicle_spec_path: str,
    comm_severity_level: str,
    dynamics_backend: str,
    coordination_success_rate: float,
    reward_generic: float,
    reward_calibrated: float,
    params: Optional[dict] = None,
    metrics: Optional[dict] = None,
    artifacts: Optional[list] = None,
    tags: Optional[dict] = None,
) -> None:
    """
    C0.5 addition (see conversation) - logs one real fleet/multi-agent
    run to MLflow. Same NEVER-raises, observational-not-operational
    invariant as log_calibration_run(): any real exception is caught,
    printed as a warning, execution returns normally. A real 3200-episode
    sweep must never be interrupted by a logging problem, same as a
    multi-hour calibration run.

    Required (mirrors log_calibration_run()'s own C1 discipline - named
    parameters, so a missing one is an immediate, real TypeError at the
    call site, not a silent gap; see mlflow_schema.FLEET_REQUIRED_FIELDS):
        script, run_name, status, status_reason, git_commit, vehicle_spec_path,
        comm_severity_level, dynamics_backend, coordination_success_rate,
        reward_generic, reward_calibrated

    comm_severity_level/dynamics_backend are validated against this
    project's own real, closed vocabularies (mlflow_schema.
    VALID_COMM_SEVERITY_LEVELS / VALID_DYNAMICS_BACKENDS) before anything
    is logged - a typo'd value fails loudly here rather than silently
    creating an unfilterable one-off tag downstream.

    Optional (dicts - callers include only what's relevant to them, e.g.
    comm_packet_loss_prob_generic_to_calibrated / comm_bandwidth_bytes_per_tick
    / comm_seed / n_agents belong in params; collision_rate or any other
    extra metric this schema doesn't yet name as a dedicated parameter
    belongs in metrics):
        params: extra parameters beyond what this function logs automatically
        metrics: extra numeric results beyond the five required above
        artifacts: list of real file paths to attach to the run (e.g. the
            per-episode CSV safe_csv_path() already writes)
        tags: extra tags beyond the always-logged ones below
    """
    try:
        schema.validate_status(status)
        schema.validate_comm_severity_level(comm_severity_level)
        schema.validate_dynamics_backend(dynamics_backend)
        # All three raised ValueErrors are caught by the outer try/except
        # below, same as log_calibration_run()'s own status check - a bad
        # value should never crash the real script.

        import mlflow

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=run_name):
            # --- Always-logged parameters
            mlflow.log_param("mlflow_schema_version", schema.SCHEMA_VERSION)
            mlflow.log_param("python_version", sys.version.split()[0])
            mlflow.log_param("numpy_version", _safe_pkg_version("numpy"))
            mlflow.log_param("scipy_version", _safe_pkg_version("scipy"))
            mlflow.log_param("stable_baselines3_version", _safe_pkg_version("stable-baselines3"))
            mlflow.log_param("mlflow_version", _safe_pkg_version("mlflow"))
            mlflow.log_param("vehicle_spec_path", vehicle_spec_path)

            # --- Caller-supplied parameters
            if params:
                for key, value in params.items():
                    mlflow.log_param(key, value)

            # --- Always-logged fleet metrics (required, per FLEET_REQUIRED_FIELDS)
            mlflow.log_metric("coordination_success_rate", coordination_success_rate)
            mlflow.log_metric("reward_generic", reward_generic)
            mlflow.log_metric("reward_calibrated", reward_calibrated)

            # --- Caller-supplied metrics
            if metrics:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

            # --- Always-logged tags
            mlflow.set_tag("script", script)
            mlflow.set_tag("git_commit", git_commit or "unknown")
            mlflow.set_tag("hostname", socket.gethostname())
            mlflow.set_tag("status", status)
            mlflow.set_tag("status_reason", status_reason)
            mlflow.set_tag("comm_severity_level", comm_severity_level)
            mlflow.set_tag("dynamics_backend", dynamics_backend)

            # --- Caller-supplied tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)

            # --- Artifacts
            if artifacts:
                for path in artifacts:
                    try:
                        mlflow.log_artifact(path)
                    except Exception as e:
                        print(f"[mlflow_logging] WARNING - could not log artifact "
                              f"{path!r}: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"[mlflow_logging] WARNING - fleet experiment tracking failed, "
              f"continuing without it: {type(e).__name__}: {e}")