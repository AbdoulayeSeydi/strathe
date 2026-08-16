"""
Frozen MLflow logging schema, v1 (item #5 - experiment tracking).

This is the single source of truth for what gets logged, and how -
log_calibration_run() (mlflow_logging.py, Phase C) imports directly from
this file rather than re-typing field names, so the schema can genuinely
never drift out of sync with what the code actually logs.

Changing this schema after real runs exist means bumping SCHEMA_VERSION,
not silently redefining what an existing field means - old runs stay
correctly labeled with whichever version they were actually logged under.
"""

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Status vocabulary - stable, closed set. Detail belongs in status_reason
# (a free-text string), never by inventing a new status value.
# ---------------------------------------------------------------------------

STATUS_SUCCESS = "SUCCESS"                        # optimizer converged per its own stopping criteria
STATUS_CHECKPOINT_FALLBACK = "CHECKPOINT_FALLBACK"  # did not converge; a checkpoint's best point was intentionally substituted
STATUS_FAILED = "FAILED"                          # no meaningful calibration result produced
STATUS_INTERRUPTED = "INTERRUPTED"                # externally terminated

VALID_STATUSES = {STATUS_SUCCESS, STATUS_CHECKPOINT_FALLBACK, STATUS_FAILED, STATUS_INTERRUPTED}

# ---------------------------------------------------------------------------
# Required fields - every caller of log_calibration_run() must supply these.
# Named as real function parameters (not dict keys), so a missing required
# field is an immediate, real TypeError at the call site, not a silent gap.
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ["script", "run_name", "status", "status_reason", "git_commit", "vehicle_spec_path"]

# ---------------------------------------------------------------------------
# Optional fields - real, but not every caller has all of these
# (e.g. run_calibration.py has no "algorithm"; train_multiseed.py has no
# "sampling_rate_hz"). Passed as **kwargs, not required positionally.
# ---------------------------------------------------------------------------

OPTIONAL_FIELDS = ["run_type", "algorithm", "seed", "task", "optimizer_settings"]

# ---------------------------------------------------------------------------
# Parameter keys (MLflow params - logged once, immutable per run)
# ---------------------------------------------------------------------------

PARAM_KEYS = [
    "mlflow_schema_version",
    "optimizer", "optimizer_settings",
    "vehicle_spec_path", "vehicle_spec_version", "vehicle_spec_hash",
    # Dataset signature - an IDENTIFIER, explicitly NOT a uniqueness
    # guarantee. Two distinct datasets could theoretically share these
    # four properties. A real cryptographic hash remains a clean, later,
    # purely-additive option if ever genuinely needed - this deliberately
    # stays cheap (no full-file hashing) for now.
    "dataset_path", "dataset_file_count", "dataset_total_size_bytes", "dataset_modified_time",
    "algorithm", "seed", "sampling_rate_hz", "max_nfev",
    # Reproducibility - the core, materially-reproducibility-relevant
    # dependencies only, not every installed package.
    "python_version", "numpy_version", "scipy_version",
    "stable_baselines3_version", "mlflow_version",
]

# ---------------------------------------------------------------------------
# Metric keys (MLflow metrics - numeric, real per-run results)
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "rmse_total", "rmse_u", "rmse_v", "rmse_w", "rmse_p", "rmse_q", "rmse_r",
    "runtime_seconds", "nfev",
]

# ---------------------------------------------------------------------------
# Tag keys (MLflow tags - string, used for filtering/searching runs)
# ---------------------------------------------------------------------------

TAG_KEYS = ["script", "git_commit", "hostname", "run_type", "status", "status_reason"]

# ---------------------------------------------------------------------------
# Artifact filenames this schema expects to be logged, when available
# ---------------------------------------------------------------------------

ARTIFACT_NAMES = ["vehicle_spec.yaml", "calibrated_parameters.yaml", "metrics.json"]


def validate_status(status: str) -> None:
    """Raises ValueError immediately if status isn't one of the four real,
    frozen values - fail loudly rather than silently log a typo'd status
    that would never match anything in a later search."""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status {status!r} - must be one of {sorted(VALID_STATUSES)}. "
            f"Detail belongs in status_reason, not a new status value."
        )