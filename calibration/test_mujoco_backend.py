"""
Tests for mujoco_backend.py. Since mujoco isn't installed in every
environment this might run in, and since the point is to test THIS
FILE's OWN logic (not MuJoCo itself), these tests inject a fake `mujoco`
module implementing the real equations-of-motion structure (M @ qacc =
xfrc_applied_generalized - qfrc_bias, qvel += qacc*dt -- the
semi-implicit-Euler structure B0.4 confirmed against real MuJoCo) but
with a completely ARBITRARY native mass matrix and bias force, unrelated
to the calibrated physics. If mujoco_backend.py's full-override logic is
correct, it reproduces dynamics_model.simulate_trajectory()'s own output
regardless of what fake "native" dynamics sit underneath -- a genuine
test of this file's real code, not just isolated math snippets.

IMPORTANT, discovered after a real bug (see conversation): MuJoCo's
xfrc_applied TORQUE component (indices 3-5) must be specified in WORLD
frame -- MuJoCo internally converts it via R.T (current body->world
rotation, transposed) before adding it as the generalized force conjugate
to qvel[3:] (confirmed body-frame). The fake harness below models this
R.T conversion explicitly (deriving R from data.qpos's quaternion each
call) -- earlier versions of this harness did NOT model it, which meant
every "machine precision" pass from this suite was blind to that specific
bug (it only ever exercised the linear path meaningfully at nonzero
rotation). Modeling it here is what makes nonzero-rotation tests actually
meaningful checks of mujoco_backend.py's real angular-frame handling,
not just its linear handling.

Deliberately scoped to roll=pitch=0 for the core end-to-end test
(test_matches_stand_in_at_identity_orientation), since that's the
simplest, most fundamental case. Nonzero rotation is now also directly
tested (test_matches_stand_in_at_nonzero_rotation and the broader sweep
test) following the divergence-bug fix -- see mujoco_backend.py's own
docstring for the fix itself.
"""
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation as _FakeRotation

from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec
from dynamics_model import (SimulationConfig, TrajectoryInputs, simulate_trajectory,
                             quaternion_to_roll_pitch)

_FAKE_NATIVE_M = np.diag([20.0, 15.0, 25.0, 3.0, 2.0, 4.0])  # arbitrary, unlike calibrated M


def _fake_current_rotation_matrix(data) -> np.ndarray:
    """Derives R (body->world) from data.qpos's quaternion [w,x,y,z],
    matching mujoco_backend.py's own quaternion convention -- used to
    model MuJoCo's real R.T conversion of xfrc_applied's torque
    component."""
    w, x, y, z = data.qpos[3:7]
    return _FakeRotation.from_quat([x, y, z, w]).as_matrix()


def _install_fake_mujoco():
    """Installs a fake `mujoco` module into sys.modules, so
    mujoco_backend.py's own `import mujoco` picks it up. Returns fresh
    MjModel/MjData factory functions bound to this fake module."""
    class FakeOpt:
        timestep = 0.002

    class FakeMjModel:
        def __init__(self):
            self.nv = 6
            self.opt = FakeOpt()
            # B3.2 addition (see conversation): mujoco_backend.py's
            # _body_qpos_dof_addr() now needs these on EVERY model,
            # including single-body ones -- real MuJoCo always has them.
            # Single-body scene: world=index 0 (no joint/dofs, -1),
            # body 1 = the only free body, joint 0, dofs starting at 0.
            self.body_jntadr = np.array([-1, 0])
            self.jnt_qposadr = np.array([0])
            self.body_dofadr = np.array([-1, 0])

    class FakeMjData:
        def __init__(self):
            self.qpos = np.zeros(7)
            self.qpos[3] = 1.0
            self.qvel = np.zeros(6)
            self.qfrc_bias = np.zeros(6)
            self.xfrc_applied = np.zeros((2, 6))
            self.qacc = np.zeros(6)

    def fake_mj_forward(model, data):
        v = data.qvel
        # Arbitrary fake bias -- some nonzero function of v, nothing like
        # real gravity/Coriolis. The point is that it's UNRELATED to the
        # calibrated physics, to prove the override truly cancels it.
        data.qfrc_bias[:] = _FAKE_NATIVE_M @ (0.1 * v) + np.array([5.0, 0.0, 50.0, 0.0, 0.0, 0.0])

    def fake_mj_fullM(model, data, dst):
        dst[:, :] = _FAKE_NATIVE_M

    def fake_mj_step(model, data):
        R = _fake_current_rotation_matrix(data)
        # Models MuJoCo's real, confirmed convention: xfrc_applied's
        # torque (world-frame input) is converted to the body-frame
        # generalized force via R.T before use. Linear part needs no
        # such conversion (confirmed: matches directly).
        xfrc_generalized = np.concatenate([data.xfrc_applied[1, :3], R.T @ data.xfrc_applied[1, 3:]])
        net_force = xfrc_generalized - data.qfrc_bias
        qacc = np.linalg.solve(_FAKE_NATIVE_M, net_force)
        data.qacc[:] = qacc
        data.qvel[:] = data.qvel + qacc * model.opt.timestep

        # B1.6: also integrate orientation and position -- earlier
        # versions of this fake never did this at all, since
        # teacher-forced mode always overwrote qpos every substep
        # regardless. Genuine forward simulation needs this to be
        # meaningfully testable. Uses the confirmed body-frame
        # right-multiply formula from b1_3_velocity_frame_check.py's own
        # confirmed test (semi-implicit: uses the just-updated qvel).
        dt = model.opt.timestep
        omega_body_new = data.qvel[3:]
        incremental_rotation = _FakeRotation.from_rotvec(omega_body_new * dt).as_matrix()
        R_new = R @ incremental_rotation
        quat_xyzw_new = _FakeRotation.from_matrix(R_new).as_quat()
        data.qpos[3:7] = [quat_xyzw_new[3], quat_xyzw_new[0], quat_xyzw_new[1], quat_xyzw_new[2]]
        data.qpos[:3] = data.qpos[:3] + data.qvel[:3] * dt

    fake_mujoco = types.ModuleType("mujoco")
    fake_mujoco.MjModel = FakeMjModel
    fake_mujoco.MjData = FakeMjData
    fake_mujoco.mj_forward = fake_mj_forward
    fake_mujoco.mj_fullM = fake_mj_fullM
    fake_mujoco.mj_step = fake_mj_step
    sys.modules["mujoco"] = fake_mujoco

    if "mujoco_backend" in sys.modules:
        del sys.modules["mujoco_backend"]  # force re-import against the fake

    return fake_mujoco


_fake_mujoco = _install_fake_mujoco()
import mujoco_backend  # noqa: E402 -- must come after the fake mujoco is installed

_vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
_params = load_calibrated_parameters("calibrated_parameters_a1.yaml")


def _make_test_trajectory(n_steps=20, seed=0, roll_pitch_value=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    return TrajectoryInputs(
        nu0=np.array([0.3, -0.1, 0.05, 0.02, -0.01, 0.0]),
        forces_torques=rng.uniform(-3, 3, size=(n_steps, 6)),
        roll_pitch_series=np.tile(roll_pitch_value, (n_steps, 1)),
    )


def test_matches_stand_in_at_identity_orientation():
    trajectory = _make_test_trajectory()
    sim_config = SimulationConfig(dt=0.1, physics_dt=0.02)

    reference = simulate_trajectory(_params, _vehicle, sim_config, trajectory)
    model, data = _fake_mujoco.MjModel(), _fake_mujoco.MjData()
    result = mujoco_backend.simulate_trajectory_mujoco(_params, _vehicle, sim_config, trajectory, model, data, teacher_forced=True)

    max_diff = np.max(np.abs(reference.nu - result.nu))
    assert max_diff < 1e-9, f"diverged from the calibrated stand-in by {max_diff}"
    assert result.used_teacher_forced_orientation is True


def test_matches_stand_in_at_nonzero_rotation():
    trajectory = _make_test_trajectory(roll_pitch_value=(0.15, -0.25))
    sim_config = SimulationConfig(dt=0.1, physics_dt=0.02)

    reference = simulate_trajectory(_params, _vehicle, sim_config, trajectory)
    model, data = _fake_mujoco.MjModel(), _fake_mujoco.MjData()
    result = mujoco_backend.simulate_trajectory_mujoco(_params, _vehicle, sim_config, trajectory, model, data, teacher_forced=True)

    assert np.all(np.isfinite(result.nu)), "still diverging -- the fix did not hold"
    max_diff = np.max(np.abs(reference.nu - result.nu))
    assert max_diff < 1e-9, f"diverged from the calibrated stand-in by {max_diff}"


def test_matches_stand_in_across_a_range_of_rotations():
    for roll, pitch in [(0.05, 0.05), (-0.3, 0.1), (0.4, -0.4), (0.0, 0.2), (-0.15, -0.15)]:
        trajectory = _make_test_trajectory(n_steps=15, roll_pitch_value=(roll, pitch))
        sim_config = SimulationConfig(dt=0.1, physics_dt=0.02)

        reference = simulate_trajectory(_params, _vehicle, sim_config, trajectory)
        model, data = _fake_mujoco.MjModel(), _fake_mujoco.MjData()
        result = mujoco_backend.simulate_trajectory_mujoco(_params, _vehicle, sim_config, trajectory, model, data, teacher_forced=True)

        assert np.all(np.isfinite(result.nu)), f"diverged at roll={roll}, pitch={pitch}"
        max_diff = np.max(np.abs(reference.nu - result.nu))
        assert max_diff < 1e-9, f"mismatch at roll={roll}, pitch={pitch}: {max_diff}"


def test_simulate_fn_closure_matches_objective_py_shape():
    trajectory = _make_test_trajectory(n_steps=10)
    sim_config = SimulationConfig(dt=0.1, physics_dt=0.02)
    model, data = _fake_mujoco.MjModel(), _fake_mujoco.MjData()

    simulate_fn = mujoco_backend.make_mujoco_simulate_fn(model, data)
    result = simulate_fn(_params, _vehicle, sim_config, trajectory)  # exactly 4 args
    assert result.nu.shape == (10, 6)


def test_shared_model_timestep_is_restored_after_the_call():
    trajectory = _make_test_trajectory()
    sim_config = SimulationConfig(dt=0.1, physics_dt=0.02)
    model, data = _fake_mujoco.MjModel(), _fake_mujoco.MjData()
    original_timestep = model.opt.timestep

    mujoco_backend.simulate_trajectory_mujoco(_params, _vehicle, sim_config, trajectory, model, data, teacher_forced=True)
    assert model.opt.timestep == original_timestep


def test_roll_pitch_to_quat_round_trips_through_dynamics_models_own_convention():
    for roll, pitch in [(0.1, -0.2), (0.0, 0.0), (-0.05, 0.3), (0.4, 0.15)]:
        quat_wxyz = mujoco_backend._roll_pitch_to_quat_wxyz(roll, pitch)
        w, x, y, z = quat_wxyz
        roll_back, pitch_back = quaternion_to_roll_pitch(x, y, z, w)
        assert abs(roll_back.item() - roll) < 1e-10
        assert abs(pitch_back.item() - pitch) < 1e-10


def test_objective_py_functions_work_unmodified_with_this_backend():
    import objective
    from objective import RunData

    rng = np.random.default_rng(0)
    n_samples = 15
    real_nu = rng.uniform(-0.5, 0.5, size=(n_samples, 6))
    run = RunData(
        filepath="fake_run.csv",
        forces_torques=rng.uniform(-2, 2, size=(n_samples, 6)),
        roll_pitch=np.zeros((n_samples, 2)),
        real_nu=real_nu,
        nu0=real_nu[0].copy(),
        dt=0.1,
        sample_rate_hz=10.0,
    )

    model, data = _fake_mujoco.MjModel(), _fake_mujoco.MjData()
    simulate_fn = mujoco_backend.make_mujoco_simulate_fn(model, data)

    rmse = objective.rmse_for_run(_params, run, _vehicle, simulate_fn=simulate_fn)
    assert np.isfinite(rmse)

    model2, data2 = _fake_mujoco.MjModel(), _fake_mujoco.MjData()
    simulate_fn2 = mujoco_backend.make_mujoco_simulate_fn(model2, data2)
    rmse_by_dof = objective.rmse_by_dof(_params, run, _vehicle, simulate_fn=simulate_fn2)
    assert rmse_by_dof.shape == (6,)
    assert np.all(np.isfinite(rmse_by_dof))

    model3, data3 = _fake_mujoco.MjModel(), _fake_mujoco.MjData()
    simulate_fn3 = mujoco_backend.make_mujoco_simulate_fn(model3, data3)
    mean_rmse = objective.mean_run_rmse(_params, [run], _vehicle, simulate_fn=simulate_fn3)
    assert np.isfinite(mean_rmse)
    assert mean_rmse == rmse


def test_extract_roll_pitch_from_mujoco_quat_matches_scipy_directly():
    from scipy.spatial.transform import Rotation

    for roll, pitch in [(0.12, -0.33), (0.0, 0.0), (-0.4, 0.2), (0.05, 0.5)]:
        quat_xyzw = Rotation.from_euler("xyz", [roll, pitch, 0.0]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        data = _fake_mujoco.MjData()
        data.qpos[3:7] = quat_wxyz

        roll_out, pitch_out = mujoco_backend._extract_roll_pitch_from_mujoco_quat(data)
        assert abs(roll_out - roll) < 1e-10
        assert abs(pitch_out - pitch) < 1e-10


def test_extract_roll_pitch_round_trips_with_this_files_own_write_direction():
    for roll, pitch in [(0.1, -0.2), (0.3, 0.05), (-0.15, -0.25)]:
        quat_wxyz = mujoco_backend._roll_pitch_to_quat_wxyz(roll, pitch)
        data = _fake_mujoco.MjData()
        data.qpos[3:7] = quat_wxyz

        roll_out, pitch_out = mujoco_backend._extract_roll_pitch_from_mujoco_quat(data)
        assert abs(roll_out - roll) < 1e-10
        assert abs(pitch_out - pitch) < 1e-10


def test_forward_simulation_default_and_orientation_actually_evolves():
    n_steps = 5
    constant_roll_pitch = (0.1, -0.1)
    trajectory = TrajectoryInputs(
        nu0=np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0]),
        forces_torques=np.zeros((n_steps, 6)),
        roll_pitch_series=np.tile(constant_roll_pitch, (n_steps, 1)),
    )
    sim_config = SimulationConfig(dt=0.1, physics_dt=0.02)
    model, data = _fake_mujoco.MjModel(), _fake_mujoco.MjData()

    result = mujoco_backend.simulate_trajectory_mujoco(_params, _vehicle, sim_config, trajectory, model, data)

    assert result.used_teacher_forced_orientation is False

    final_roll, _final_pitch = mujoco_backend._extract_roll_pitch_from_mujoco_quat(data)
    assert not np.isclose(final_roll, constant_roll_pitch[0], atol=1e-3), (
        f"orientation stayed pinned to the logged value ({final_roll} ~= {constant_roll_pitch[0]}) "
        f"-- teacher-forcing appears to still be silently happening despite the default"
    )


def test_forward_simulation_produces_finite_reasonable_output():
    trajectory = _make_test_trajectory(n_steps=15, roll_pitch_value=(0.1, -0.15))
    sim_config = SimulationConfig(dt=0.1, physics_dt=0.02)
    model, data = _fake_mujoco.MjModel(), _fake_mujoco.MjData()

    result = mujoco_backend.simulate_trajectory_mujoco(_params, _vehicle, sim_config, trajectory, model, data)

    assert result.used_teacher_forced_orientation is False
    assert result.nu.shape == (15, 6)
    assert np.all(np.isfinite(result.nu))


def test_forward_simulation_stays_close_to_teacher_forced_over_a_short_horizon():
    trajectory = _make_test_trajectory(n_steps=5, roll_pitch_value=(0.05, -0.05))
    sim_config = SimulationConfig(dt=0.1, physics_dt=0.02)

    model_tf, data_tf = _fake_mujoco.MjModel(), _fake_mujoco.MjData()
    teacher_forced_result = mujoco_backend.simulate_trajectory_mujoco(
        _params, _vehicle, sim_config, trajectory, model_tf, data_tf, teacher_forced=True)

    model_fwd, data_fwd = _fake_mujoco.MjModel(), _fake_mujoco.MjData()
    forward_result = mujoco_backend.simulate_trajectory_mujoco(
        _params, _vehicle, sim_config, trajectory, model_fwd, data_fwd)

    assert np.all(np.isfinite(forward_result.nu))
    max_diff = np.max(np.abs(teacher_forced_result.nu - forward_result.nu))
    assert max_diff < 1.0, f"forward-sim result implausibly far from teacher-forced over a short horizon: {max_diff}"


def test_forward_simulation_calls_rotation_matrix_path_not_scalar_extraction():
    original_scalar = mujoco_backend.compute_acceleration
    original_matrix = mujoco_backend.compute_acceleration_from_rotation_matrix
    scalar_calls = {"n": 0}
    matrix_calls = {"n": 0}

    def spy_scalar(*args, **kwargs):
        scalar_calls["n"] += 1
        return original_scalar(*args, **kwargs)

    def spy_matrix(*args, **kwargs):
        matrix_calls["n"] += 1
        return original_matrix(*args, **kwargs)

    mujoco_backend.compute_acceleration = spy_scalar
    mujoco_backend.compute_acceleration_from_rotation_matrix = spy_matrix
    try:
        trajectory = _make_test_trajectory(n_steps=5, roll_pitch_value=(0.1, -0.1))
        sim_config = SimulationConfig(dt=0.1, physics_dt=0.02)
        model, data = _fake_mujoco.MjModel(), _fake_mujoco.MjData()
        mujoco_backend.simulate_trajectory_mujoco(_params, _vehicle, sim_config, trajectory, model, data)
    finally:
        mujoco_backend.compute_acceleration = original_scalar
        mujoco_backend.compute_acceleration_from_rotation_matrix = original_matrix

    assert scalar_calls["n"] == 0, (
        f"forward simulation called the scalar roll/pitch path {scalar_calls['n']} times -- "
        f"the fix is not actually wired in"
    )
    assert matrix_calls["n"] > 0, "forward simulation never called the rotation-matrix path at all"


def test_forward_simulation_handles_near_gimbal_lock_orientation():
    n_steps = 10
    trajectory = TrajectoryInputs(
        nu0=np.array([0.1, 0.05, -0.05, 0.02, 0.01, 0.03]),
        forces_torques=np.random.default_rng(1).uniform(-2, 2, size=(n_steps, 6)),
        roll_pitch_series=np.tile([0.1, np.radians(89.9)], (n_steps, 1)),
    )
    sim_config = SimulationConfig(dt=0.1, physics_dt=0.02)
    model, data = _fake_mujoco.MjModel(), _fake_mujoco.MjData()

    result = mujoco_backend.simulate_trajectory_mujoco(_params, _vehicle, sim_config, trajectory, model, data)

    assert np.all(np.isfinite(result.nu)), "forward simulation diverged starting near gimbal lock"
    assert result.used_teacher_forced_orientation is False


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