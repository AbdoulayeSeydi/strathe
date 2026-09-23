"""
Equivalence tests: fast_dynamics.py must reproduce the reference physics
(dynamics_model.compute_acceleration + ArenaCore.step) to floating-point
tolerance. If any of these fail, fast_dynamics is wrong -- never "fix" the
reference to match it.
"""
import time

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from arena_core import ArenaCore
from dynamics_model import CalibrationParameters, SimulationConfig, compute_acceleration
from fast_dynamics import FastCore, FastDynamics, _rotvec_matrix, roll_pitch_from_matrix
from optimizer import initial_parameters, load_vehicle_spec, parameter_bounds

SPEC = load_vehicle_spec("vehicle_spec.yaml")
VEHICLE = SPEC.vehicle


def _random_params(rng):
    """Parameters drawn uniformly inside vehicle_spec.yaml's own bounds,
    rejecting the rare draw whose mass matrix isn't positive-definite."""
    lb, ub = parameter_bounds(SPEC)
    names = initial_parameters(SPEC).param_names
    while True:
        vec = rng.uniform(lb, ub)
        try:
            p = CalibrationParameters(**dict(zip(names, vec)))
            FastDynamics(p, VEHICLE)
            return p
        except ValueError:
            continue


def _param_sets():
    rng = np.random.default_rng(0)
    sets = [("generic", initial_parameters(SPEC))]
    with open("calibrated_parameters_a1.yaml", encoding="utf-8") as f:
        sets.append(("a1", CalibrationParameters(**yaml.safe_load(f)["parameters"])))
    sets += [(f"random{i}", _random_params(rng)) for i in range(5)]
    return sets


PARAM_SETS = _param_sets()


@pytest.mark.parametrize("name,params", PARAM_SETS, ids=[n for n, _ in PARAM_SETS])
def test_acceleration_matches_reference(name, params):
    rng = np.random.default_rng(1)
    fast = FastDynamics(params, VEHICLE)
    for _ in range(200):
        v = rng.normal(0, [0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
        tau = rng.uniform(-30, 30, 6)
        roll, pitch = rng.uniform(-1.2, 1.2, 2)
        ref = compute_acceleration(params, VEHICLE, v, tau, roll, pitch)[0]
        got = np.array(fast.acceleration(list(v), list(tau), roll, pitch))
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("name,params", PARAM_SETS, ids=[n for n, _ in PARAM_SETS])
def test_trajectory_matches_arena_core(name, params):
    """200 ticks (20 s) of random, smoothly-varying thrust from a random
    start: position, velocity and orientation must track ArenaCore."""
    rng = np.random.default_rng(2)
    cfg = SimulationConfig(dt=0.1, physics_dt=0.005)
    ref, fast = ArenaCore(params, VEHICLE, cfg), FastCore(params, VEHICLE, cfg)
    nu0 = rng.normal(0, 0.2, 6)
    pos0 = rng.normal(0, 2, 3)
    ref.reset(position=pos0, nu=nu0)
    fast.reset(position=pos0, nu=nu0)
    tau = np.zeros(6)
    for _ in range(200):
        tau = np.clip(tau + rng.normal(0, 3, 6), -30, 30)
        s_ref = ref.step(tau)
        fast.step(tau)
    np.testing.assert_allclose(fast.nu_array(), s_ref.nu, atol=1e-8)
    np.testing.assert_allclose(fast.position_array(), s_ref.position, atol=1e-8)
    np.testing.assert_allclose(np.array(fast.R), s_ref.orientation.as_matrix(), atol=1e-8)


def test_reset_with_orientation_matches_arena_core_integration():
    """FastCore can start from a non-identity orientation (needed for the
    fleet belief model). Check against an ArenaCore whose state is set to
    the same orientation directly."""
    params = initial_parameters(SPEC)
    cfg = SimulationConfig(dt=0.1)
    rot = Rotation.from_euler("xyz", [0.2, -0.3, 1.1])
    ref, fast = ArenaCore(params, VEHICLE, cfg), FastCore(params, VEHICLE, cfg)
    ref.reset(position=[1, 2, 3], nu=[0.3, 0, 0, 0, 0, 0.1])
    ref.state.orientation = rot
    fast.reset(position=[1, 2, 3], nu=[0.3, 0, 0, 0, 0, 0.1], orientation=rot)
    for _ in range(50):
        s_ref = ref.step([5, 0, 0, 0, 0, 1])
        fast.step([5, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(fast.position_array(), s_ref.position, atol=1e-8)
    np.testing.assert_allclose(np.array(fast.R), s_ref.orientation.as_matrix(), atol=1e-8)


def test_rotvec_matrix_matches_scipy_including_tiny_angles():
    rng = np.random.default_rng(3)
    for scale in (1e-9, 1e-6, 1e-3, 0.1, 1.0, 3.0):
        for _ in range(50):
            w = rng.normal(0, scale, 3)
            np.testing.assert_allclose(np.array(_rotvec_matrix(*w)),
                                       Rotation.from_rotvec(w).as_matrix(), atol=1e-12)


def test_roll_pitch_matches_scipy_euler_xyz():
    rng = np.random.default_rng(4)
    for _ in range(500):
        angles = [rng.uniform(-np.pi, np.pi), rng.uniform(-1.4, 1.4), rng.uniform(-np.pi, np.pi)]
        R = Rotation.from_euler("xyz", angles)
        roll, pitch = roll_pitch_from_matrix(R.as_matrix().tolist())
        ref_roll, ref_pitch, _ = R.as_euler("xyz")
        assert abs(roll - ref_roll) < 1e-9 and abs(pitch - ref_pitch) < 1e-9


def test_state_property_round_trips():
    fast = FastCore(initial_parameters(SPEC), VEHICLE, SimulationConfig(dt=0.1))
    fast.reset(position=[1, 2, 3], nu=[0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
    fast.step([3, -2, 1, 0.5, 0, -0.5])
    s = fast.state
    np.testing.assert_allclose(s.position, fast.position_array())
    np.testing.assert_allclose(s.nu, fast.nu_array())
    np.testing.assert_allclose(s.orientation.as_matrix(), np.array(fast.R), atol=1e-12)


def test_non_finite_state_raises():
    fast = FastCore(initial_parameters(SPEC), VEHICLE, SimulationConfig(dt=0.1))
    fast.reset(nu=[np.inf, 0, 0, 0, 0, 0])
    with pytest.raises(FloatingPointError):
        fast.step([0] * 6)


def test_fast_core_is_meaningfully_faster():
    """Performance guard: the whole point of this module. Loose bound so it
    isn't flaky on slow CI runners."""
    params = initial_parameters(SPEC)
    cfg = SimulationConfig(dt=0.1)
    ref, fast = ArenaCore(params, VEHICLE, cfg), FastCore(params, VEHICLE, cfg)
    tau = [5, 1, -2, 0.3, 0.1, -0.2]
    t = time.perf_counter()
    for _ in range(100):
        ref.step(tau)
    t_ref = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(100):
        fast.step(tau)
    t_fast = time.perf_counter() - t
    assert t_fast * 3 < t_ref, f"FastCore {t_fast:.3f}s vs ArenaCore {t_ref:.3f}s"


@pytest.mark.parametrize("name,params", PARAM_SETS, ids=[n for n, _ in PARAM_SETS])
def test_simulate_trajectory_fast_path_matches_reference_loop(name, params):
    """dynamics_model.simulate_trajectory uses fast_dynamics unless debug output
    is requested; debug=True runs the original reference loop. Both must agree."""
    from dynamics_model import TrajectoryInputs, simulate_trajectory
    rng = np.random.default_rng(5)
    n = 150
    traj = TrajectoryInputs(nu0=rng.normal(0, 0.2, 6),
                            forces_torques=np.clip(np.cumsum(rng.normal(0, 2, (n, 6)), axis=0), -30, 30),
                            roll_pitch_series=rng.uniform(-0.4, 0.4, (n, 2)))
    for dt, pdt in ((0.1, 0.005), (0.02, 0.005)):
        fast = simulate_trajectory(params, VEHICLE, SimulationConfig(dt=dt, physics_dt=pdt), traj).nu
        ref = simulate_trajectory(params, VEHICLE, SimulationConfig(dt=dt, physics_dt=pdt, debug=True), traj).nu
        np.testing.assert_allclose(fast, ref, rtol=1e-9, atol=1e-10)


def test_simulate_trajectory_fast_path_still_reports_divergence():
    from dynamics_model import TrajectoryInputs, simulate_trajectory
    params = initial_parameters(SPEC)
    traj = TrajectoryInputs(nu0=np.array([1e200, 0, 0, 0, 0, 0]), forces_torques=np.full((20, 6), 1e200),
                            roll_pitch_series=np.zeros((20, 2)))
    with pytest.raises(FloatingPointError):
        simulate_trajectory(params, VEHICLE, SimulationConfig(dt=0.1), traj)
