"""Fin angles reach every place a run is replayed (#97): a recording made with fins must replay exactly from its own truth."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import objective
from dynamics_model import CalibrationParameters, SimulationConfig, TrajectoryInputs, VehicleConstants, simulate_trajectory
from estimate_current import _half
from fin_force import Actuators, Fin, servo_angles
from measurements import slice_run
from objective import RunData, with_thruster_lag
from simfit import core
from validation import _lagged, free_run, window_replay

P = CalibrationParameters(Ieff_roll=2, Ieff_pitch=9, Ieff_yaw=9, Xu=-5, Yv=-6, Zw=-7, Kp=-1, Mq=-3, Nr=-3, Xuu=-5, Yvv=-6, Zww=-7,
                          Kpp=-.5, Mqq=-.5, Nrr=-.5, Xu_dot=3, Yv_dot=30, Zw_dot=30, net_buoyancy_z=0, cg_cb_offset_z=.02,
                          fin_lift_slope_a=3.0, fin_lift_slope_b=2.5, fin_inflow_a=.8, fin_inflow_b=.8, fin_drag0=.02)
ACT = Actuators(fins=(Fin("r", (-.8, 0, 0), (0, 0, 1), .03, .2, group=0), Fin("s", (-.8, 0, 0), (0, 1, 0), .03, .2, group=1)))
V = VehicleConstants(mass_kg=30.0, actuators=ACT)
N = 400


def recording(servo=False, params=P):
    rng = np.random.default_rng(4)
    ctl = np.repeat(rng.uniform(-.15, .15, (N // 20, 2)), 20, axis=0)
    rp = np.c_[.05 * np.sin(np.arange(N) / 30), .1 * np.sin(np.arange(N) / 50)]
    F = np.tile([25., 0, 0, 0, 0, 0], (N, 1))
    applied = servo_angles(ACT, ctl, params.fin_servo_T, .1) if servo else ctl
    nu0 = np.array([1.5, 0, 0, 0, 0, 0.])
    traj = TrajectoryInputs(nu0=nu0, forces_torques=F, roll_pitch_series=rp, control_series=applied)
    nu = simulate_trajectory(params, V, SimulationConfig(dt=.1), traj).nu
    q = Rotation.from_euler("xyz", np.c_[rp, np.zeros(N)]).as_quat()
    return RunData("f-01-01-2030-run1.csv", F, rp, nu, nu0, .1, 10.0, orientation_quat=q, controls=ctl,
                   servo=ACT if servo else None)


def test_the_fit_replays_a_fin_recording_exactly():
    r = recording()
    assert np.abs(r.real_nu[:, 5]).max() > .02                      # the fins really turned it
    assert np.abs(objective.residuals_for_run(P, r, V, normalize_by_run_length=False)).max() < 1e-9


def test_re_anchored_windows_carry_the_fins():
    """Every window starts from the logged state with its own slice of the fin angles: a wrong slice shows at once."""
    r = recording()
    res = objective.residuals_for_run(P, r, V, normalize_by_run_length=False, horizon_samples=25)
    assert np.abs(res).max() < 1e-2
    wrong = replace(r, controls=np.roll(r.controls, 50, axis=0))
    assert np.abs(objective.residuals_for_run(P, wrong, V, normalize_by_run_length=False, horizon_samples=25)).max() > .02


def test_the_fit_refuses_a_fin_vehicle_with_no_fin_angles():
    with pytest.raises(ValueError, match="logged controls"):
        objective.residuals_for_run(P, replace(recording(), controls=None), V)


def test_slices_keep_the_fins():
    r = recording()
    for s in (slice_run(r, 10, 60), _half(r, 10, 60)):
        assert s.controls.shape == (50, 2) and np.array_equal(s.controls, r.controls[10:60])


def test_validations_window_replay_uses_the_fins():
    r = recording()
    assert np.abs(window_replay(P, V, r, 40, 100) - r.real_nu[40:100]).max() < 1e-2


def test_free_runs_use_the_fins():
    r = recording()
    ref = free_run(P, V, r, 40, 80)
    Vs, Q = core.strathe_free_run(P, V, r, 40, 80)
    rpy = Rotation.from_quat(Q).as_euler("xyz")
    assert ref is not None and np.abs(np.unwrap(rpy[1:], axis=0) - np.unwrap(ref[1:], axis=0)).max() < 1e-9
    assert np.abs(Vs[:, 5]).max() > .02


def test_commanded_fins_get_the_servo_lag_in_the_fit():
    p = replace(P, fin_servo_T=.5)
    r = recording(servo=True, params=p)
    assert np.abs(objective.residuals_for_run(p, r, V, normalize_by_run_length=False)).max() < 1e-9
    no_lag = replace(p, fin_servo_T=0.0)
    assert np.abs(objective.residuals_for_run(no_lag, r, V, normalize_by_run_length=False)).max() > 1e-3


def test_the_lag_cache_tells_servo_times_apart():
    r = recording(servo=True, params=replace(P, fin_servo_T=.5))
    a = _lagged(r, replace(P, fin_servo_T=.5)).controls
    b = _lagged(r, replace(P, fin_servo_T=1.0)).controls
    assert not np.array_equal(a, b)


def test_measured_fin_angles_are_used_as_they_are():
    r = recording()
    assert with_thruster_lag(r, replace(P, fin_servo_T=.5)) is r
