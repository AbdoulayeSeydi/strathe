"""
holoocean_backend.py logic tests against a FAKE HoloOcean (HoloOcean can't
be installed in CI or on macOS).

The fake integrates a free rigid body the way HoloOcean's Custom Dynamics
is documented to: it takes a WORLD-frame linear/angular acceleration per
env.step(), integrates velocity and orientation over one tick, and reports
PoseSensor / VelocitySensor / IMUSensor in the conventions
holoocean_backend.py documents (including the diag(1,-1,-1) body-frame
correction). These tests therefore check THIS REPO'S frame conversion
logic, not HoloOcean itself -- verify_holoocean_frame_conventions.py is the
check to run on real HoloOcean.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import holoocean_backend as hb
from dynamics_model import SimulationConfig, TrajectoryInputs
from fast_dynamics import FastCore
from optimizer import initial_parameters, load_vehicle_spec

SPEC = load_vehicle_spec("vehicle_spec.yaml")
PARAMS = initial_parameters(SPEC)
C = np.diag([1.0, -1.0, -1.0])


class FakeAgent:
    def __init__(self, env):
        self.env = env

    def set_control_scheme(self, scheme):
        assert scheme == hb.CUSTOM_DYNAMICS_SCHEME

    def set_physics_state(self, location, rotation, velocity, angular_velocity):
        e = self.env
        e.pos = np.array(location, dtype=float)
        e.R = Rotation.from_euler("xyz", np.radians(rotation)).as_matrix()
        e.V = np.array(velocity, dtype=float)
        e.W = C @ np.array(angular_velocity, dtype=float)  # backend sends C @ (R w_body)


class FakeHoloOcean:
    def __init__(self, tick_s=0.1, n_sub=20, ticks_per_sec=10.0):
        self.tick_s, self.n_sub = tick_s, n_sub
        self.agents = {"auv0": FakeAgent(self)}
        if ticks_per_sec is not None:
            self._strathe_ticks_per_sec = ticks_per_sec

    def step(self, command):
        a_lin = np.asarray(command[:3], dtype=float)
        a_ang = C @ np.asarray(command[3:], dtype=float)  # undo the backend's angular correction
        h = self.tick_s / self.n_sub
        for _ in range(self.n_sub):
            self.pos = self.pos + self.V * h
            self.V = self.V + a_lin * h
            self.W = self.W + a_ang * h
            self.R = Rotation.from_rotvec(self.W * h).as_matrix() @ self.R
        pose = np.eye(4)
        pose[:3, :3] = self.R @ C  # backend multiplies by C again to recover R
        pose[:3, 3] = self.pos
        imu = np.zeros((4, 3))
        imu[1] = self.R.T @ self.W  # body-frame angular velocity
        return {"PoseSensor": pose, "VelocitySensor": self.V.copy(), "IMUSensor": imu}


def _turning_trajectory(n=60):
    """Constant forward thrust + yaw torque from a moving, turning start."""
    forces = np.tile([20.0, 0.0, 0.0, 0.0, 0.0, 3.0], (n, 1))
    return TrajectoryInputs(nu0=np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.4]), forces_torques=forces,
                            roll_pitch_series=np.zeros((n, 2)))


def _reference(traj, n):
    # physics_dt=dt: acceleration recomputed once per 0.1 s tick, exactly as the
    # backend does (it sends one command per logged sample). With FastCore's
    # usual 5 ms recompute the yaw rate alone differs by ~0.26 rad/s here, for
    # both backend variants equally -- an update-rate effect, not a frame one.
    core = FastCore(PARAMS, SPEC.vehicle, SimulationConfig(dt=0.1, physics_dt=0.1))
    core.reset(nu=traj.nu0)
    out = [traj.nu0]
    for k in range(n - 1):
        core.step(traj.forces_torques[k])
        out.append(core.nu_array())
    return np.array(out)


def _replay(transport):
    traj = _turning_trajectory()
    res = hb.simulate_trajectory_holoocean(PARAMS, SPEC.vehicle, SimulationConfig(dt=0.1), traj,
                                           FakeHoloOcean(), transport_term=transport)
    return res.nu, _reference(traj, 60)


def test_transport_term_makes_turning_replay_match_the_reference_integrator():
    fixed, ref = _replay(True)
    buggy, _ = _replay(False)
    err_fixed = np.abs(fixed - ref).max()
    err_buggy = np.abs(buggy - ref).max()
    # measured: fixed ~0.03, original (no transport term) ~0.40 -- surge decays
    # to 0.22 m/s instead of 0.48 because world velocity doesn't turn with the vehicle
    assert err_fixed < 0.06, err_fixed
    assert err_buggy > 5 * err_fixed, (err_buggy, err_fixed)


def test_without_turning_the_transport_term_changes_nothing():
    traj = TrajectoryInputs(nu0=np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0]),
                            forces_torques=np.tile([10.0, 0, 0, 0, 0, 0], (30, 1)),
                            roll_pitch_series=np.zeros((30, 2)))
    a = hb.simulate_trajectory_holoocean(PARAMS, SPEC.vehicle, SimulationConfig(dt=0.1), traj,
                                         FakeHoloOcean(), transport_term=True).nu
    b = hb.simulate_trajectory_holoocean(PARAMS, SPEC.vehicle, SimulationConfig(dt=0.1), traj,
                                         FakeHoloOcean(), transport_term=False).nu
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_tick_rate_mismatch_is_refused():
    with pytest.raises(ValueError, match="ticks/s"):
        hb.simulate_trajectory_holoocean(PARAMS, SPEC.vehicle, SimulationConfig(dt=0.1), _turning_trajectory(5),
                                         FakeHoloOcean(ticks_per_sec=30.0))


def test_unknown_tick_rate_warns_once(capsys):
    env = FakeHoloOcean(ticks_per_sec=None)
    for _ in range(2):
        hb.simulate_trajectory_holoocean(PARAMS, SPEC.vehicle, SimulationConfig(dt=0.1), _turning_trajectory(5), env)
    assert capsys.readouterr().out.count("tick rate unknown") == 1


@pytest.mark.parametrize("tick_s,expected_exit", [(0.1, 0), (1 / 30, 1)])
def test_verification_script_detects_tick_mismatch(monkeypatch, tick_s, expected_exit):
    """The on-HoloOcean verification script, run against the fake: passes at
    a 0.1 s tick, fails (non-zero exit) at HoloOcean's 30 ticks/s default."""
    import verify_holoocean_frame_conventions as v
    monkeypatch.setattr(hb, "make_deterministic_environment", lambda: FakeHoloOcean(tick_s=tick_s))
    assert v.main() == expected_exit


def test_verification_script_detects_a_missing_transport_term(monkeypatch, capsys):
    """If the engine rotated world velocity with the body by itself (so the
    term would double-count), the script must say transport_term is wrong."""
    import verify_holoocean_frame_conventions as v

    class CoRotatingFake(FakeHoloOcean):
        def step(self, command):
            R_before = self.R.copy()
            out = super().step(command)
            self.V = self.R @ R_before.T @ self.V  # engine carries velocity round with the body
            out["VelocitySensor"] = self.V.copy()
            return out
    monkeypatch.setattr(hb, "make_deterministic_environment", lambda: CoRotatingFake())
    assert v.main() == 1
    assert "WRONG for this HoloOcean build" in capsys.readouterr().out
