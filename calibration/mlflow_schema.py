"""
Frozen MLflow logging schema, v2 (item #5 - experiment tracking).

This is the single source of truth for what gets logged, and how -
log_calibration_run()/log_fleet_run() (mlflow_logging.py, Phase C) import
directly from this file rather than re-typing field names, so the schema
can genuinely never drift out of sync with what the code actually logs.

Changing this schema after real runs exist means bumping SCHEMA_VERSION,
not silently redefining what an existing field means - old runs stay
correctly labeled with whichever version they were actually logged under.

## v1 -> v2 (C0.5, see conversation): fleet-level fields added
Track C's multi-agent work (C0.1-C0.4) produces runs the v1 schema has no
vocabulary for at all: comm-condition metadata, per-agent reward,
coordination-success rate. These are ADDED here as new PARAM_KEYS/
TAG_KEYS/METRIC_KEYS entries and a new FLEET_REQUIRED_FIELDS list -
nothing from v1 is removed or redefined, so every existing v1 run stays
correctly, unambiguously labeled schema_version=1. SCHEMA_VERSION bumps
to 2 anyway (not left at 1), because this genuinely changes the SHAPE of
what "the schema" covers, even though the change is purely additive -
matching this file's own stated rule literally ("changing this schema
... means bumping SCHEMA_VERSION"), not just its narrower intent
(never redefining an existing field).
"""

SCHEMA_VERSION = 2

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
# C0.5 addition: required/optional fields for log_fleet_run() specifically
# - a SEPARATE list from REQUIRED_FIELDS/OPTIONAL_FIELDS above (which are
# explicitly scoped to log_calibration_run() by that file's own comment),
# not a merge into them, since log_calibration_run() and log_fleet_run()
# are genuinely different functions for genuinely different run types.
# vehicle_spec_path stays required here too - a fleet run still shares
# ONE vehicle_spec.yaml across both agents (see build_two_agent_cores.py's
# own single VEHICLE_SPEC_PATH constant); only the calibration differs
# per agent, not the vehicle spec itself.
# ---------------------------------------------------------------------------

FLEET_REQUIRED_FIELDS = [
    "script", "run_name", "status", "status_reason", "git_commit", "vehicle_spec_path",
    "comm_severity_level", "dynamics_backend", "coordination_success_rate",
    "reward_generic", "reward_calibrated",
]

FLEET_OPTIONAL_FIELDS = ["run_type", "algorithm", "seed", "task"]

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
    # --- C0.5 addition: fleet comm-condition metadata (numeric/scalar
    # config, immutable per run -- matches this list's own existing
    # convention, e.g. sampling_rate_hz, seed). Field names match
    # comm_channel.py's OWN real PacketLossConfig/BandwidthConfig
    # dataclass fields exactly (near_distance_m, far_distance_m,
    # near_field_prob, far_field_prob, bytes_per_tick) -- an earlier
    # version of this schema invented "comm_packet_loss_prob_generic_to_
    # calibrated"/"..._calibrated_to_generic", assuming a PER-DIRECTION
    # config that does not exist: PacketLossConfig is ONE shared,
    # distance-conditioned config applied identically regardless of
    # sender/receiver (see comm_channel.py's own CommChannel.send() --
    # drop_probability() takes only a distance, never a direction).
    # Fixed here once the real file was actually seen, rather than left
    # silently wrong. comm_severity_level itself (see TAG_KEYS below) is
    # a categorical label FOR these underlying numeric values, not a
    # substitute for logging them.
    "comm_near_distance_m", "comm_far_distance_m", "comm_near_field_prob", "comm_far_field_prob",
    "comm_bandwidth_bytes_per_tick", "comm_seed", "n_agents",
]

# ---------------------------------------------------------------------------
# Metric keys (MLflow metrics - numeric, real per-run results)
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "rmse_total", "rmse_u", "rmse_v", "rmse_w", "rmse_p", "rmse_q", "rmse_r",
    "runtime_seconds", "nfev",
    # --- C0.5 addition: fleet per-agent reward and coordination success
    # (real per-run RESULTS, not fixed config -- matches this list's own
    # convention, e.g. rmse_total). "coordination_success_rate" (not a
    # bare "success_rate") to make explicit this is the multi-agent
    # formation-hold success concept (FormationHoldTask's own
    # TERMINAL_REASON_FORMATION_ACHIEVED), not any future single-vehicle
    # success metric that might otherwise share the shorter name.
    "reward_generic", "reward_calibrated", "coordination_success_rate",
]

# ---------------------------------------------------------------------------
# Tag keys (MLflow tags - string, used for filtering/searching runs)
# ---------------------------------------------------------------------------

TAG_KEYS = [
    "script", "git_commit", "hostname", "run_type", "status", "status_reason",
    # --- C0.5 addition: fleet categorical/filtering fields -- matches
    # this list's own existing convention (status, run_type are tags,
    # not params, specifically because they're filtering-oriented
    # categorical labels rather than raw numeric configuration).
    "comm_severity_level", "dynamics_backend",
]

# ---------------------------------------------------------------------------
# Artifact filenames this schema expects to be logged, when available
# ---------------------------------------------------------------------------

ARTIFACT_NAMES = ["vehicle_spec.yaml", "calibrated_parameters.yaml", "metrics.json"]

# C0.5 addition: fleet runs' own expected artifacts -- the per-episode CSV
# evaluate_comm_degradation_curve.py/_mujoco.py already write via
# safe_csv_path(), not a new file format invented for this schema.
FLEET_ARTIFACT_NAMES = ["vehicle_spec.yaml"]

# C0.5 addition: the closed, real severity vocabulary this project has
# actually used (evaluate_comm_degradation_curve.py's own COMM_CONFIGS
# keys) -- a stable, closed set, same discipline as VALID_STATUSES above,
# so a typo'd severity fails loudly at log time rather than silently
# creating an unfilterable one-off tag value.
VALID_COMM_SEVERITY_LEVELS = {"L0", "L1", "L2", "L3a", "L3b"}

# C0.5 addition: the two real dynamics backends this project's own
# TwoAgentOrchestrator/build_two_agent_cores support (see
# two_agent_orchestrator.py's dynamics_backend parameter) -- matches
# phase2_config.py's TRAINING_BACKEND/VALIDATION_BACKEND vocabulary
# ("stand_in"/"mujoco") exactly, not a separately-invented naming scheme.
VALID_DYNAMICS_BACKENDS = {"stand_in", "mujoco"}


def validate_status(status: str) -> None:
    """Raises ValueError immediately if status isn't one of the four real,
    frozen values - fail loudly rather than silently log a typo'd status
    that would never match anything in a later search."""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status {status!r} - must be one of {sorted(VALID_STATUSES)}. "
            f"Detail belongs in status_reason, not a new status value."
        )


def validate_comm_severity_level(comm_severity_level: str) -> None:
    """C0.5 addition - same fail-loudly discipline as validate_status,
    applied to the new fleet vocabulary."""
    if comm_severity_level not in VALID_COMM_SEVERITY_LEVELS:
        raise ValueError(
            f"Invalid comm_severity_level {comm_severity_level!r} - must be one of "
            f"{sorted(VALID_COMM_SEVERITY_LEVELS)} (evaluate_comm_degradation_curve.py's "
            f"own COMM_CONFIGS keys). A new severity level should be added to "
            f"VALID_COMM_SEVERITY_LEVELS here explicitly, not silently logged as a typo."
        )


def validate_dynamics_backend(dynamics_backend: str) -> None:
    """C0.5 addition - same fail-loudly discipline as validate_status."""
    if dynamics_backend not in VALID_DYNAMICS_BACKENDS:
        raise ValueError(
            f"Invalid dynamics_backend {dynamics_backend!r} - must be one of "
            f"{sorted(VALID_DYNAMICS_BACKENDS)} (two_agent_orchestrator.py's own "
            f"dynamics_backend values)."
        )