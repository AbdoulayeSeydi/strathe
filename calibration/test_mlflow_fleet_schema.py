#!/usr/bin/env python3
"""
test_mlflow_fleet_schema.py -- verifies C0.5's fleet-level schema
extension (mlflow_schema.py v2) and log_fleet_run() (mlflow_logging.py),
AND that log_calibration_run() survived the same-file rewrite unchanged
-- no real MLflow install needed, a fake mlflow module captures every
call for inspection, same discipline this whole session used for
MuJoCo/gymnasium.

Usage:
    python3 test_mlflow_fleet_schema.py
"""
import sys
import types

import mlflow_schema as schema


class _FakeMlflow:
    """Captures every call log_calibration_run()/log_fleet_run() makes,
    for inspection -- does not require the real mlflow package at all."""
    def __init__(self):
        self.tracking_uri = None
        self.experiment_name = None
        self.params = {}
        self.metrics = {}
        self.tags = {}
        self.artifacts = []
        self.run_started = False
        self.run_name = None

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment_name = name

    def start_run(self, run_name=None):
        self.run_started = True
        self.run_name = run_name
        return self  # used as a context manager below

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def log_param(self, key, value):
        self.params[key] = value

    def log_metric(self, key, value):
        self.metrics[key] = value

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_artifact(self, path):
        self.artifacts.append(path)


def _install_fake_mlflow():
    fake = _FakeMlflow()
    fake_module = types.ModuleType("mlflow")
    fake_module.set_tracking_uri = fake.set_tracking_uri
    fake_module.set_experiment = fake.set_experiment
    fake_module.start_run = fake.start_run
    fake_module.log_param = fake.log_param
    fake_module.log_metric = fake.log_metric
    fake_module.set_tag = fake.set_tag
    fake_module.log_artifact = fake.log_artifact
    sys.modules["mlflow"] = fake_module
    return fake


# --- Schema-level tests ---

def test_schema_version_bumped_to_2():
    assert schema.SCHEMA_VERSION == 2


def test_v1_fields_untouched():
    assert schema.REQUIRED_FIELDS == ["script", "run_name", "status", "status_reason",
                                       "git_commit", "vehicle_spec_path"]
    assert schema.OPTIONAL_FIELDS == ["run_type", "algorithm", "seed", "task", "optimizer_settings"]
    assert "rmse_total" in schema.METRIC_KEYS
    assert set(schema.VALID_STATUSES) == {"SUCCESS", "CHECKPOINT_FALLBACK", "FAILED", "INTERRUPTED"}


def test_fleet_param_keys_added():
    for key in ("comm_near_distance_m", "comm_far_distance_m", "comm_near_field_prob",
                "comm_far_field_prob", "comm_bandwidth_bytes_per_tick", "comm_seed", "n_agents"):
        assert key in schema.PARAM_KEYS, f"{key} missing from PARAM_KEYS"


def test_fleet_tag_keys_added():
    for key in ("comm_severity_level", "dynamics_backend"):
        assert key in schema.TAG_KEYS, f"{key} missing from TAG_KEYS"


def test_fleet_metric_keys_added():
    for key in ("reward_generic", "reward_calibrated", "coordination_success_rate"):
        assert key in schema.METRIC_KEYS, f"{key} missing from METRIC_KEYS"


def test_fleet_required_fields_are_first_class():
    for key in ("comm_severity_level", "dynamics_backend", "coordination_success_rate",
                "reward_generic", "reward_calibrated"):
        assert key in schema.FLEET_REQUIRED_FIELDS, f"{key} missing from FLEET_REQUIRED_FIELDS"


def test_valid_comm_severity_levels_match_project_vocabulary():
    assert schema.VALID_COMM_SEVERITY_LEVELS == {"L0", "L1", "L2", "L3a", "L3b"}


def test_valid_dynamics_backends_match_orchestrator():
    assert schema.VALID_DYNAMICS_BACKENDS == {"stand_in", "mujoco"}


def test_validate_comm_severity_level_accepts_real_values():
    for level in schema.VALID_COMM_SEVERITY_LEVELS:
        schema.validate_comm_severity_level(level)  # must not raise


def test_validate_comm_severity_level_rejects_typo():
    raised = False
    try:
        schema.validate_comm_severity_level("L4")
    except ValueError:
        raised = True
    assert raised


def test_validate_dynamics_backend_rejects_typo():
    raised = False
    try:
        schema.validate_dynamics_backend("holoocean")  # real backend, but not a DYNAMICS backend
    except ValueError:
        raised = True
    assert raised


# --- log_fleet_run() behavioral tests ---

def test_log_fleet_run_logs_required_fleet_metrics():
    fake = _install_fake_mlflow()
    import mlflow_logging
    mlflow_logging.log_fleet_run(
        script="evaluate_comm_degradation_curve_mujoco.py",
        run_name="L2-trained@L2",
        status="SUCCESS",
        status_reason="sweep completed",
        git_commit="abc123",
        vehicle_spec_path="vehicle_spec.yaml",
        comm_severity_level="L2",
        dynamics_backend="mujoco",
        coordination_success_rate=0.2138,
        reward_generic=-150.0,
        reward_calibrated=-140.0,
    )
    assert fake.metrics["coordination_success_rate"] == 0.2138
    assert fake.metrics["reward_generic"] == -150.0
    assert fake.metrics["reward_calibrated"] == -140.0
    assert fake.tags["comm_severity_level"] == "L2"
    assert fake.tags["dynamics_backend"] == "mujoco"
    assert fake.run_started


def test_log_fleet_run_missing_required_kwarg_raises_immediately():
    """Confirms the C1-style discipline actually holds: omitting a
    required field is a real TypeError, not a silent gap -- not just
    documented, actually enforced by Python's keyword-only parameters."""
    _install_fake_mlflow()
    import mlflow_logging
    raised = False
    try:
        mlflow_logging.log_fleet_run(
            script="x", run_name="y", status="SUCCESS", status_reason="z",
            git_commit="abc", vehicle_spec_path="vehicle_spec.yaml",
            comm_severity_level="L0", dynamics_backend="mujoco",
            coordination_success_rate=0.5, reward_generic=-1.0,
            # reward_calibrated deliberately omitted
        )
    except TypeError:
        raised = True
    assert raised, "omitting a required fleet field should raise TypeError immediately"


def test_log_fleet_run_bad_comm_severity_is_caught_not_raised():
    """Observational-not-operational invariant: a bad value must NOT
    crash the caller -- caught internally, printed as a warning."""
    _install_fake_mlflow()
    import mlflow_logging
    mlflow_logging.log_fleet_run(
        script="x", run_name="y", status="SUCCESS", status_reason="z",
        git_commit="abc", vehicle_spec_path="vehicle_spec.yaml",
        comm_severity_level="L4",  # invalid
        dynamics_backend="mujoco", coordination_success_rate=0.5,
        reward_generic=-1.0, reward_calibrated=-1.0,
    )  # must not raise


def test_log_fleet_run_optional_params_and_metrics_pass_through():
    fake = _install_fake_mlflow()
    import mlflow_logging
    mlflow_logging.log_fleet_run(
        script="x", run_name="y", status="SUCCESS", status_reason="z",
        git_commit="abc", vehicle_spec_path="vehicle_spec.yaml",
        comm_severity_level="L1", dynamics_backend="stand_in",
        coordination_success_rate=0.35, reward_generic=-186.0, reward_calibrated=-186.0,
        params={"comm_seed": 3, "n_agents": 2},
        metrics={"collision_rate": 0.05},
        tags={"custom_tag": "value"},
    )
    assert fake.params["comm_seed"] == 3
    assert fake.params["n_agents"] == 2
    assert fake.metrics["collision_rate"] == 0.05
    assert fake.tags["custom_tag"] == "value"


# --- log_calibration_run() unchanged (full-file rewrite risk check) ---

def test_log_calibration_run_still_works_unchanged():
    fake = _install_fake_mlflow()
    import mlflow_logging
    mlflow_logging.log_calibration_run(
        script="run_calibration.py",
        run_name="test_run",
        status="SUCCESS",
        status_reason="converged",
        git_commit="abc123",
        vehicle_spec_path="vehicle_spec.yaml",
        metrics={"rmse_total": 0.219287},
    )
    assert fake.metrics["rmse_total"] == 0.219287
    assert fake.tags["script"] == "run_calibration.py"
    assert fake.run_started
    # A calibration run must NOT log fleet-only tags -- confirms the two
    # functions stayed genuinely separate, not silently merged.
    assert "comm_severity_level" not in fake.tags


def test_log_calibration_run_bad_status_is_caught_not_raised():
    _install_fake_mlflow()
    import mlflow_logging
    mlflow_logging.log_calibration_run(
        script="x", run_name="y", status="BOGUS", status_reason="z",
        git_commit="abc", vehicle_spec_path="vehicle_spec.yaml",
    )  # must not raise


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        raise SystemExit(1)