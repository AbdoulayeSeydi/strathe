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