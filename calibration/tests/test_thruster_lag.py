"""
Thruster lag (PROJECT_STATUS.md #66): a first-order spin-up on the applied wrench, thruster_time_constant_s.
At 0 everything is bit-for-bit the model before it; otherwise the calibration replay, validation and every simulator
core apply the same discretisation, and a calibration recovers a known lag.
"""
import math
import os

import numpy as np
import pytest
import yaml

from dynamics_model import (CalibrationParameters, SimulationConfig, ThrusterLag, TrajectoryInputs,
                            apply_thruster_lag, simulate_trajectory)

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
with open(os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml"), encoding="utf-8") as _f:
    _raw = yaml.safe_load(_f)
SHIPPED = {k: float(v) for k, v in _raw.get("parameters", _raw).items()}


def params(T=0.0, **over):
    return CalibrationParameters(**{**SHIPPED, "thruster_time_constant_s": T, **over})


def test_no_lag_is_the_command_itself():
    F = np.random.default_rng(0).normal(size=(50, 6))
    assert apply_thruster_lag(F, 0.0, 0.1) is F
    lag = ThrusterLag()
    tau = [1.0, 2, 3, 4, 5, 6]
    assert lag.step(tau, 0.0, 0.1) is tau


def test_the_applied_wrench_is_the_exact_average_of_a_first_order_lag():
    """A step from 0 to 1 at t=0 with T = 0.25 s: over [k dt, (k+1) dt] the true response 1 - exp(-t/T) averages
    to exactly what apply_thruster_lag gives."""
    T, dt, n = 0.25, 0.1, 30
    u = np.zeros((n, 6))
    u[:, 0] = 1.0
    got = apply_thruster_lag(u, T, dt, initial=np.zeros(6))[:, 0]
    for k in range(n):
        t0, t1 = k * dt, (k + 1) * dt
        exact = 1 - (T / dt) * (math.exp(-t0 / T) - math.exp(-t1 / T))
        assert got[k] == pytest.approx(exact, abs=1e-12)


def test_series_and_step_by_step_agree():
    rng = np.random.default_rng(1)
    u = np.repeat(rng.normal(size=(12, 6)) * 10, 5, axis=0)  # held commands
    series = apply_thruster_lag(u, 0.3, 0.1, initial=np.zeros(6))
    lag = ThrusterLag()
    steps = np.array([lag.step(row, 0.3, 0.1) for row in u])
    assert np.allclose(series, steps, atol=1e-12)


def test_every_core_lags_the_same_way():
    """ArenaCore, FastCore and MuJoCoArenaCore, stepped with the same surge command: identical lag, and slower than
    without it."""
    from arena_core import ArenaCore
    from fast_dynamics import FastCore
    from optimizer import load_vehicle_spec
    vehicle = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml")).vehicle
    cfg = SimulationConfig(dt=0.1)
    cores = [ArenaCore(params(0.3), vehicle, cfg), FastCore(params(0.3), vehicle, cfg)]
    try:
        from mujoco_backend import MuJoCoArenaCore
        cores.append(MuJoCoArenaCore(params(0.3), vehicle, cfg))
    except Exception:
        pass
    tau = np.array([30.0, 0, 0, 0, 0, 0])

    def surge(core):
        core.reset()
        out = []
        for _ in range(15):
            st = core.step(tau)
            out.append(float(core.nu[0]) if isinstance(core, FastCore) else float(st.nu[0]))
        return out
    traces = [surge(c) for c in cores]
    for t in traces[1:]:
        assert np.allclose(t, traces[0], atol=1e-4), (t, traces[0])  # MuJoCo's own integration: ~3e-6 apart
    fast = ArenaCore(params(0.0), vehicle, cfg)
    fast.reset()
    no_lag = [float(fast.step(tau).nu[0]) for _ in range(15)]
    assert traces[0][2] < 0.8 * no_lag[2]  # still spinning up after 0.3 s
    assert traces[0][-1] == pytest.approx(no_lag[-1], rel=0.05)  # same speed once spun up


def test_a_new_episode_starts_with_the_thrusters_stopped():
    from fast_dynamics import FastCore
    from optimizer import load_vehicle_spec
    vehicle = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml")).vehicle
    core = FastCore(params(0.3), vehicle, SimulationConfig(dt=0.1))
    for _ in range(20):
        core.step([30.0, 0, 0, 0, 0, 0])
    core.reset()
    core.step([30.0, 0, 0, 0, 0, 0])
    first = core.nu[0]
    core.reset(applied_wrench=[30.0, 0, 0, 0, 0, 0])  # a teammate belief: thrusters already at the command
    core.step([30.0, 0, 0, 0, 0, 0])
    assert core.nu[0] > 1.5 * first


def test_a_spec_without_thruster_dynamics_holds_no_lag(tmp_path):
    from optimizer import held_by_bounds, initial_parameters, load_vehicle_spec, parameter_bounds
    spec = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml"))
    p = initial_parameters(spec)
    assert p.thruster_time_constant_s == 0.0
    held = held_by_bounds(*parameter_bounds(spec), list(p.param_names))
    assert held == {"thruster_time_constant_s": 0.0} or "thruster_time_constant_s" not in held
    raw = yaml.safe_load(open(os.path.join(CAL, "vehicle_spec.yaml"), encoding="utf-8"))
    raw["calibration_targets"].pop("thruster_dynamics", None)
    (tmp_path / "old_spec.yaml").write_text(yaml.safe_dump(raw))
    old = load_vehicle_spec(str(tmp_path / "old_spec.yaml"))
    assert held_by_bounds(*parameter_bounds(old), list(p.param_names)) == {"thruster_time_constant_s": 0.0}


def _lagged_recording(T_true, seconds=60.0, rate=10.0, seed=0):
    """A recording whose velocities come from the shipped model with a T_true lag, teacher-forced level."""
    from objective import RunData
    from optimizer import load_vehicle_spec
    vehicle = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml")).vehicle
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    F = np.zeros((n, 6))
    for j, size in ((0, 30.0), (1, 20.0), (2, 15.0), (5, 3.0)):
        for k in range(0, n, int(3 * rate)):
            F[k:k + int(3 * rate), j] = size * rng.choice([-1, -0.5, 0, 0.5, 1])
    rp = np.zeros((n, 2))
    applied = apply_thruster_lag(F, T_true, 1 / rate)
    nu = simulate_trajectory(params(T_true), vehicle, SimulationConfig(dt=1 / rate),
                             TrajectoryInputs(nu0=np.zeros(6), forces_torques=applied, roll_pitch_series=rp)).nu
    return RunData(filepath="lag-01-01-2030-run1.csv", forces_torques=F, roll_pitch=rp, real_nu=nu,
                   nu0=nu[0].copy(), dt=1 / rate, sample_rate_hz=rate), vehicle


def test_calibration_recovers_a_known_lag(tmp_path):
    """Everything else held at the truth: the fit finds T = 0.25 s from logged commands and velocities alone."""
    from optimizer import CalibrationConfig, calibrate, load_vehicle_spec
    run, vehicle = _lagged_recording(0.25)
    raw = yaml.safe_load(open(os.path.join(CAL, "vehicle_spec.yaml"), encoding="utf-8"))
    raw["calibration_targets"]["thruster_dynamics"] = {"thruster_time_constant_s": {
        "unit": "s", "initial_guess": 0.05, "bounds": [0.0, 1.0], "bounds_rationale": "test", "source": "test",
        "confidence": "assumed"}}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(raw))
    spec = load_vehicle_spec(str(tmp_path / "spec.yaml"))
    fixed = {k: v for k, v in SHIPPED.items() if k in CalibrationParameters.__dataclass_fields__}
    lo_hi = {n: e["bounds"] for n, e in spec.flat_calibration_targets.items()}
    fixed = {k: min(max(v, lo_hi[k][0]), lo_hi[k][1]) for k, v in fixed.items()}
    res = calibrate(spec, [run], CalibrationConfig(fixed_params=fixed, max_nfev=40))
    # the truth's own values were clamped into the spec's bounds above; the lag is what's left to explain
    assert res.parameters.thruster_time_constant_s == pytest.approx(0.25, abs=0.03)


def test_validation_step_response_sees_the_lag():
    """On a recording made with a 0.25 s lag, a model without lag responds faster than the vehicle; with the lag,
    as fast."""
    import validation as V
    run, vehicle = _lagged_recording(0.25, seconds=120.0)
    s = V.step_responses({"with_lag": params(0.25), "no_lag": params(0.0)}, [run], vehicle)["summary"]["u"]
    assert s["steps"] >= 3
    assert s["with_lag_rise_s"] == pytest.approx(s["real_rise_s"], abs=0.05)
    assert s["no_lag_rise_s"] < s["real_rise_s"]
