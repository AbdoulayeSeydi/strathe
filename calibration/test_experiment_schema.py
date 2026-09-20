#!/usr/bin/env python3
"""
test_experiment_schema.py -- F1.4 unit test matrix for experiment_schema.py.

Verifies:
  - Every valid config variation passes without error
  - Each invalid case produces the SPECIFIC correct error, not just "some error"
    (checking that the message actually names the broken field/constraint)

Coverage per F1.4:
  - wrong vehicle count (1, 0, 3)
  - bad severity name
  - bad algorithm name
  - bad dynamics_backend
  - invalid comm override combos (via PacketLossConfig.validate())
  - no comm spec at all
  - non-positive episode/seed counts
  - empty comm_seeds
  - missing model file
  - unwritable output_dir
  - empty output_dir
  - YAML file not found (load_and_validate)

Run from calibration/:
    python3 test_experiment_schema.py
"""
import os
import stat
import sys
import tempfile

from pydantic import ValidationError

from experiment_schema import (
    ExperimentConfig,
    CommOverrides,
    load_and_validate,
    _effective_severity_label,
)

# Pre-shipped model that actually exists on disk -- used for all tests
# that need a valid vehicle model path.
SHIPPED_MODEL = "formation_hold_ppo_curriculum_v5"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid(**overrides):
    """Build a raw config dict that passes all validation. Override any
    individual fields to test specific failure modes. Uses /tmp as
    output_dir so makedirs always succeeds without creating test debris."""
    base = {
        "vehicles": [
            {"model_path": SHIPPED_MODEL, "algorithm": "ppo"},
            {"model_path": SHIPPED_MODEL, "algorithm": "ppo"},
        ],
        "communication": {"severity": "L2"},
        "task": {"target_separation_m": 2.0, "tolerance_m": 0.2, "hold_steps": 20},
        "evaluation": {"episodes": 10, "comm_seeds": [1, 2], "grid_seeds": 2},
        "dynamics_backend": "stand_in",
        "output_dir": "/tmp",
    }
    base.update(overrides)
    return base


def _error_contains(exc, *fragments):
    """Check that the exception message contains ALL of the given strings."""
    msg = str(exc).lower()
    return all(f.lower() in msg for f in fragments)


passed = failed = 0


def _run(test_fn):
    global passed, failed
    try:
        test_fn()
        print(f"PASS  {test_fn.__name__}")
        passed += 1
    except AssertionError as e:
        print(f"FAIL  {test_fn.__name__}: {e}")
        failed += 1
    except Exception as e:
        print(f"ERROR {test_fn.__name__}: {type(e).__name__}: {e}")
        failed += 1


def _expect_error(raw, *fragments):
    """Assert that constructing ExperimentConfig(**raw) raises an error
    whose message contains all of the given fragments."""
    try:
        ExperimentConfig(**raw)
        raise AssertionError(f"expected error containing {fragments!r} -- no error raised")
    except (ValidationError, ValueError) as e:
        if not _error_contains(e, *fragments):
            raise AssertionError(
                f"error raised but message doesn't contain {fragments!r}.\n"
                f"Got: {str(e)[:200]}"
            )


# ---------------------------------------------------------------------------
# VALID CONFIGS -- all must pass
# ---------------------------------------------------------------------------

def test_valid_minimal():
    c = ExperimentConfig(**_valid())
    assert len(c.vehicles) == 2
    assert c.communication.severity == "L2"
    assert c.dynamics_backend == "stand_in"


def test_valid_all_fields_explicit():
    c = ExperimentConfig(**_valid(
        name="my_experiment",
        task={"target_separation_m": 3.0, "tolerance_m": 0.3, "hold_steps": 15},
        evaluation={"episodes": 50, "comm_seeds": [1, 2, 3], "grid_seeds": 3},
        dynamics_backend="mujoco",
    ))
    assert c.name == "my_experiment"
    assert c.task.target_separation_m == 3.0
    assert c.dynamics_backend == "mujoco"


def test_valid_all_severity_levels():
    """Every named level must be accepted without error."""
    for level in ("L0", "L1", "L2", "L3a", "L3b"):
        c = ExperimentConfig(**_valid(communication={"severity": level}))
        assert c.communication.severity == level, f"severity {level} not set correctly"


def test_valid_all_backends():
    for backend in ("stand_in", "mujoco", "both"):
        c = ExperimentConfig(**_valid(dynamics_backend=backend))
        assert c.dynamics_backend == backend


def test_valid_overrides_only():
    """No severity -- pure custom condition. Must be labeled 'custom'."""
    c = ExperimentConfig(**_valid(communication={
        "overrides": {"near_field_prob": 0.25, "far_field_prob": 0.25}
    }))
    assert c.communication.severity is None
    label = _effective_severity_label(c.communication)
    assert label == "custom", f"expected 'custom', got {label!r}"


def test_valid_severity_with_overrides():
    """Named severity + partial override. Must be labeled '<severity>_custom'."""
    c = ExperimentConfig(**_valid(communication={
        "severity": "L2",
        "overrides": {"speed_of_sound_mps": 1450.0},
    }))
    assert c.communication.severity == "L2"
    assert c.communication.overrides.speed_of_sound_mps == 1450.0
    label = _effective_severity_label(c.communication)
    assert label == "L2_custom", f"expected 'L2_custom', got {label!r}"


def test_valid_different_roles():
    c = ExperimentConfig(**_valid(vehicles=[
        {"model_path": SHIPPED_MODEL, "algorithm": "ppo", "role": "calibrated"},
        {"model_path": SHIPPED_MODEL, "algorithm": "ppo", "role": "generic"},
    ]))
    assert c.vehicles[0].role == "calibrated"
    assert c.vehicles[1].role == "generic"


def test_valid_output_dir_created():
    """makedirs creates the directory at validation time -- confirm it exists."""
    with tempfile.TemporaryDirectory() as parent:
        target = os.path.join(parent, "nested", "output")
        c = ExperimentConfig(**_valid(output_dir=target))
        assert os.path.isdir(target), f"output_dir {target!r} was not created"
        assert c.output_dir == target


# ---------------------------------------------------------------------------
# INVALID: VEHICLE COUNT
# ---------------------------------------------------------------------------

def test_invalid_one_vehicle():
    _expect_error(_valid(vehicles=[
        {"model_path": SHIPPED_MODEL, "algorithm": "ppo"},
    ]), "exactly", "2")


def test_invalid_three_vehicles():
    _expect_error(_valid(vehicles=[
        {"model_path": SHIPPED_MODEL, "algorithm": "ppo"},
        {"model_path": SHIPPED_MODEL, "algorithm": "ppo"},
        {"model_path": SHIPPED_MODEL, "algorithm": "ppo"},
    ]), "exactly", "2")


def test_invalid_zero_vehicles():
    _expect_error(_valid(vehicles=[]), "exactly", "2")


# ---------------------------------------------------------------------------
# INVALID: BAD SEVERITY / ALGORITHM / BACKEND
# ---------------------------------------------------------------------------

def test_invalid_bad_severity():
    # Pydantic Literal validation: message references the bad value or valid options
    _expect_error(_valid(communication={"severity": "L99"}), "l99")


def test_invalid_bad_algorithm():
    _expect_error(_valid(vehicles=[
        {"model_path": SHIPPED_MODEL, "algorithm": "dqn"},
        {"model_path": SHIPPED_MODEL, "algorithm": "ppo"},
    ]), "dqn")


def test_invalid_bad_dynamics_backend():
    _expect_error(_valid(dynamics_backend="holoocean"), "holoocean")


# ---------------------------------------------------------------------------
# INVALID: COMM OVERRIDES (caught via PacketLossConfig.validate())
# ---------------------------------------------------------------------------

def test_invalid_override_far_lt_near():
    """far_distance_m < near_distance_m -- PacketLossConfig.validate() catches this."""
    _expect_error(_valid(communication={
        "severity": "L2",
        "overrides": {"near_distance_m": 500.0, "far_distance_m": 100.0},
    }), "far_distance")


def test_invalid_override_negative_distance():
    """Negative near_distance_m -- PacketLossConfig.validate() catches this."""
    _expect_error(_valid(communication={
        "overrides": {
            "near_distance_m": -10.0,
            "near_field_prob": 0.3,
            "far_field_prob": 0.3,
        }
    }), "non-negative")


def test_invalid_override_prob_gt_1():
    """near_field_prob > 1 -- CommOverrides validator catches this."""
    _expect_error(_valid(communication={
        "severity": "L2",
        "overrides": {"near_field_prob": 1.5},
    }), "near_field_prob", "[0, 1]")


def test_invalid_override_prob_negative():
    """Negative probability -- CommOverrides validator catches this."""
    _expect_error(_valid(communication={
        "severity": "L2",
        "overrides": {"far_field_prob": -0.1},
    }), "far_field_prob", "[0, 1]")


def test_invalid_speed_of_sound_zero():
    _expect_error(_valid(communication={
        "severity": "L2",
        "overrides": {"speed_of_sound_mps": 0.0},
    }), "speed_of_sound")


def test_invalid_no_comm_spec():
    """Empty comm block -- no severity AND no overrides."""
    _expect_error(_valid(communication={}), "severity")


# ---------------------------------------------------------------------------
# INVALID: EVALUATION FIELDS
# ---------------------------------------------------------------------------

def test_invalid_zero_episodes():
    _expect_error(
        _valid(evaluation={"episodes": 0, "comm_seeds": [1], "grid_seeds": 1}),
        "episodes", "positive",
    )


def test_invalid_negative_episodes():
    _expect_error(
        _valid(evaluation={"episodes": -5, "comm_seeds": [1], "grid_seeds": 1}),
        "episodes",
    )


def test_invalid_zero_grid_seeds():
    _expect_error(
        _valid(evaluation={"episodes": 10, "comm_seeds": [1], "grid_seeds": 0}),
        "grid_seeds", "positive",
    )


def test_invalid_negative_grid_seeds():
    _expect_error(
        _valid(evaluation={"episodes": 10, "comm_seeds": [1], "grid_seeds": -1}),
        "grid_seeds",
    )


def test_invalid_empty_comm_seeds():
    _expect_error(
        _valid(evaluation={"episodes": 10, "comm_seeds": [], "grid_seeds": 1}),
        "comm_seeds", "non-empty",
    )


# ---------------------------------------------------------------------------
# INVALID: MODEL FILE / OUTPUT DIR
# ---------------------------------------------------------------------------

def test_invalid_missing_model_file():
    _expect_error(_valid(vehicles=[
        {"model_path": "definitely_does_not_exist_12345", "algorithm": "ppo"},
        {"model_path": SHIPPED_MODEL, "algorithm": "ppo"},
    ]), "not found")


def test_invalid_unwritable_output_dir():
    """Use a read-only temp directory to guarantee the makedirs call fails."""
    tmp = tempfile.mkdtemp()
    try:
        os.chmod(tmp, stat.S_IRUSR | stat.S_IXUSR)  # r-x, not writable
        unwritable = os.path.join(tmp, "subdir")
        _expect_error(_valid(output_dir=unwritable), "output_dir")
    finally:
        os.chmod(tmp, stat.S_IRWXU)  # restore so rmdir works
        os.rmdir(tmp)


def test_invalid_empty_output_dir():
    _expect_error(_valid(output_dir=""), "output_dir", "non-empty")


# ---------------------------------------------------------------------------
# LOAD_AND_VALIDATE -- YAML file handling
# ---------------------------------------------------------------------------

def test_load_missing_yaml_raises_value_error():
    try:
        load_and_validate("definitely_not_a_real_file_12345.yaml")
        raise AssertionError("should have raised ValueError")
    except ValueError as e:
        assert "not found" in str(e).lower(), f"wrong message: {str(e)[:120]}"


# ---------------------------------------------------------------------------
# RUN ALL
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_valid_minimal,
        test_valid_all_fields_explicit,
        test_valid_all_severity_levels,
        test_valid_all_backends,
        test_valid_overrides_only,
        test_valid_severity_with_overrides,
        test_valid_different_roles,
        test_valid_output_dir_created,
        test_invalid_one_vehicle,
        test_invalid_three_vehicles,
        test_invalid_zero_vehicles,
        test_invalid_bad_severity,
        test_invalid_bad_algorithm,
        test_invalid_bad_dynamics_backend,
        test_invalid_override_far_lt_near,
        test_invalid_override_negative_distance,
        test_invalid_override_prob_gt_1,
        test_invalid_override_prob_negative,
        test_invalid_speed_of_sound_zero,
        test_invalid_no_comm_spec,
        test_invalid_zero_episodes,
        test_invalid_negative_episodes,
        test_invalid_zero_grid_seeds,
        test_invalid_negative_grid_seeds,
        test_invalid_empty_comm_seeds,
        test_invalid_missing_model_file,
        test_invalid_unwritable_output_dir,
        test_invalid_empty_output_dir,
        test_load_missing_yaml_raises_value_error,
    ]

    for t in tests:
        _run(t)

    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        raise SystemExit(1)
