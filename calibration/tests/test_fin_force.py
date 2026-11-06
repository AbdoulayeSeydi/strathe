"""The fin physics (#97): against an independent oracle, against itself, and against what must be true of any fin."""
import math
from dataclasses import replace

import numpy as np
import pytest

import fin_oracle as oracle
from dynamics_model import (CalibrationParameters, NO_CONTROLS, SimulationConfig, TrajectoryInputs, VehicleConstants,
                            compute_acceleration, compute_acceleration_from_rotation_matrix, simulate_trajectory)
from fast_dynamics import FastDynamics
from fin_force import Actuators, Fin, MovingMass, Propeller, actuator_wrench, prepare, servo_angles
from scipy.spatial.transform import Rotation

P = CalibrationParameters(Ieff_roll=2, Ieff_pitch=9, Ieff_yaw=9, Xu=-5, Yv=-6, Zw=-7, Kp=-1, Mq=-3, Nr=-3, Xuu=-5, Yvv=-6, Zww=-7,
                          Kpp=-.5, Mqq=-.5, Nrr=-.5, Xu_dot=3, Yv_dot=30, Zw_dot=30, net_buoyancy_z=.2, cg_cb_offset_z=.02,
                          fin_lift_slope_a=3.0, fin_lift_slope_b=2.5, fin_inflow_a=.8, fin_inflow_b=.6, fin_drag0=.02)
RUDDER_TOP = Fin("top", (-.8, 0, -.1), (0, 0, 1), .02, .15, group=0, mix=(0, 0, 1), stall_rad=.35)
RUDDER_BOT = Fin("bottom", (-.8, 0, .1), (0, 0, 1), .02, .15, group=0, mix=(0, 0, 1), stall_rad=.35)
STERN_P = Fin("port", (-.8, -.1, 0), (0, 1, 0), .02, .15, group=1, mix=(0, 1, 0), stall_rad=.35)
STERN_S = Fin("star", (-.8, .1, 0), (0, 1, 0), .02, .15, group=1, mix=(0, 1, 0), stall_rad=.35)
CROSS = Actuators(fins=(RUDDER_TOP, RUDDER_BOT, STERN_P, STERN_S))
V = VehicleConstants(mass_kg=30.0, actuators=CROSS)


def wrench(act, v, c, p=P, sr=0., cr=1., sp=0., cp=1.):
    return np.array(actuator_wrench(act, prepare(act, p), p, list(v), list(c), sr, cr, sp, cp))


@pytest.mark.parametrize("seed", range(40))
def test_every_fin_matches_the_independent_oracle(seed):
    rng = np.random.default_rng(seed)
    v = np.r_[rng.uniform(.5, 2.5), rng.normal(size=2) * .15, rng.normal(size=3) * .2]
    d = rng.uniform(-.2, .2, 4)
    total = np.zeros(6)
    for f, delta, s, e in zip(CROSS.fins, d, (3., 3., 2.5, 2.5), (.8, .8, .6, .6)):
        total += oracle.fin_wrench(f.position_m, f.span_axis, f.area_m2, f.span_m, s, e, .02, f.oswald, delta, v)
    unstalled = Actuators(fins=tuple(replace(f, stall_rad=10.0) for f in CROSS.fins))   # the oracle has no stall
    assert np.allclose(wrench(unstalled, v, d), total, rtol=1e-11, atol=1e-11)


def test_no_flow_no_force_and_no_failure():
    assert not wrench(CROSS, np.zeros(6), [.3, -.3, .2, .1]).any()


def test_force_goes_as_speed_squared():
    d = [.1, .1, .05, .05]
    a, b = wrench(CROSS, [1, 0, 0, 0, 0, 0], d), wrench(CROSS, [2, 0, 0, 0, 0, 0], d)
    assert np.allclose(b, 4 * a, rtol=1e-12)


def test_a_mirrored_rudder_mirrors_the_turn():
    a = wrench(Actuators(fins=(RUDDER_TOP, RUDDER_BOT)), [1.5, 0, 0, 0, 0, 0], [.1, .1])
    b = wrench(Actuators(fins=(RUDDER_TOP, RUDDER_BOT)), [1.5, 0, 0, 0, 0, 0], [-.1, -.1])
    assert np.allclose(a * [1, -1, 1, 1, 1, -1], b, rtol=1e-12, atol=1e-12)
    assert a[5] * a[1] < 0   # behind the centre: a push one way turns the nose the other


def test_centred_fins_damp_sway_and_turning():
    act = Actuators(fins=(RUDDER_TOP, RUDDER_BOT))
    f = wrench(act, [1.5, .1, 0, 0, 0, 0], [0, 0])
    assert f[1] < 0 and f[5] > 0      # sway pushed back, nose turned into the flow
    assert wrench(act, [1.5, 0, 0, 0, 0, .1], [0, 0])[5] < 0     # turning resisted


def test_no_inflow_means_centred_fins_do_nothing_but_drag():
    f = wrench(CROSS, [1.5, .1, .1, 0, 0, 0], [0] * 4, replace(P, fin_inflow_a=0, fin_inflow_b=0))
    assert abs(f[1]) < .05 * abs(wrench(CROSS, [1.5, .1, .1, 0, 0, 0], [0] * 4)[1]) and f[0] < 0


def test_lift_stalls_smoothly():
    act = Actuators(fins=(RUDDER_TOP,))
    L = [wrench(act, [1.5, 0, 0, 0, 0, 0], [a], replace(P, fin_inflow_a=0))[1] for a in np.linspace(0, 1.2, 241)]
    assert np.max(np.abs(np.diff(L))) < 4 * np.abs(np.diff(L))[:40].mean()   # no jump at stall
    assert abs(L[int(.35 / .005)]) > abs(L[-1])


def test_flow_from_behind_still_makes_sense():
    f = wrench(Actuators(fins=(RUDDER_TOP,)), [-1.0, 0, 0, 0, 0, 0], [.1])
    g = wrench(Actuators(fins=(RUDDER_TOP,)), [1.0, 0, 0, 0, 0, 0], [.1])
    assert np.all(np.isfinite(f)) and f[1] * g[1] < 0   # the fin pushes the other way when the water comes from behind


@pytest.mark.parametrize("seed", range(20))
def test_three_physics_paths_agree(seed):
    rng = np.random.default_rng(seed)
    v, tau, d = np.r_[1.5, rng.normal(size=5) * .15], rng.normal(size=6) * 4, list(rng.uniform(-.2, .2, 4))
    roll, pitch = rng.normal(size=2) * .2
    cb = rng.normal(size=3) * .2
    ref = compute_acceleration(P, V, v, tau, roll, pitch, cb, d)[0]
    fast = FastDynamics(P, V).acceleration_trig(list(v), list(tau), math.sin(roll), math.cos(roll), math.sin(pitch),
                                                math.cos(pitch), list(cb), d)
    R = Rotation.from_euler("xyz", [roll, pitch, rng.normal()]).as_matrix()
    rot = compute_acceleration_from_rotation_matrix(P, V, v, tau, R, R @ cb, d)[0]
    assert np.allclose(fast, ref, rtol=1e-11, atol=1e-11) and np.allclose(rot, ref, rtol=1e-9, atol=1e-10)


def test_fins_see_the_water_not_the_ground():
    """In a uniform current, moving with the water means no flow past the fins."""
    v, cb = np.r_[.4, .2, -.1, 0, 0, 0], np.array([.4, .2, -.1])
    a = compute_acceleration(replace(P, net_buoyancy_z=0, cg_cb_offset_z=0), V, v, np.zeros(6), 0, 0, cb, [.2, .2, .1, .1])[0]
    assert np.allclose(a, 0, atol=1e-12)


def test_a_vehicle_with_fins_refuses_a_replay_without_them():
    with pytest.raises(ValueError, match="logged controls"):
        compute_acceleration(P, V, np.ones(6), np.zeros(6), 0, 0)
    with pytest.raises(ValueError, match="logged controls"):
        FastDynamics(P, V).acceleration(list(np.ones(6)), [0] * 6, 0, 0)


def test_a_vehicle_without_actuators_is_untouched():
    hover = VehicleConstants(mass_kg=30.0)
    a = compute_acceleration(P, hover, np.ones(6) * .2, np.ones(6), .1, .2)[0]
    b = compute_acceleration(P, hover, np.ones(6) * .2, np.ones(6), .1, .2, None, [1, 2, 3])[0]
    assert np.array_equal(a, b)


def test_simulation_paths_agree_with_fins_driving():
    n = 60
    rng = np.random.default_rng(1)
    ctl = np.repeat(rng.uniform(-.2, .2, (6, 4)), 10, axis=0)
    traj = TrajectoryInputs(nu0=np.array([1.5, 0, 0, 0, 0, 0.]), forces_torques=np.tile([20., 0, 0, 0, 0, 0], (n, 1)),
                            roll_pitch_series=np.zeros((n, 2)), control_series=ctl)
    fast = simulate_trajectory(P, V, SimulationConfig(dt=.1), traj).nu
    ref = simulate_trajectory(P, V, SimulationConfig(dt=.1, debug=True), traj).nu
    assert np.allclose(fast, ref, rtol=1e-9, atol=1e-10) and np.abs(fast[:, 5]).max() > .02


def test_propeller_matches_the_published_form():
    """Fossen's REMUS: T = rho D^4 (KT0 |n| n + (KTmax - KT0)/Jmax (Va/D) |n|), Va = 0.944 U, with thrust deduction 0.1."""
    D, KT0, KTm, Jm = .14, .4566, .1798, .6632
    p = Propeller(D, KT0, (KTm - KT0) / Jm, wake=.944, thrust_deduction=.1)
    act = Actuators(propeller=p, rho=1026.0)
    n, U = 1200 / 60, 1.4
    want = .9 * 1026 * D ** 4 * (KT0 * n * n + (KTm - KT0) / Jm * (.944 * U / D) * n)
    assert wrench(act, [U, 0, 0, 0, 0, 0], [1200.0])[0] == pytest.approx(want, rel=1e-12)


def test_thrust_scale_scales_thrust():
    act = Actuators(propeller=Propeller(.14, .45))
    a, b = wrench(act, [1, 0, 0, 0, 0, 0], [1000.]), wrench(act, [1, 0, 0, 0, 0, 0], [1000.], replace(P, thrust_scale=.5))
    assert b[0] == pytest.approx(.5 * a[0])


def test_propeller_reaction_torque_and_reverse():
    act = Actuators(propeller=Propeller(.14, .45, kq0=.07))
    f = wrench(act, [0, 0, 0, 0, 0, 0], [1000.])
    assert f[3] > 0 and f[0] > 0 and wrench(act, [0, 0, 0, 0, 0, 0], [-1000.])[0] < 0


def test_a_vectored_thruster_turns_the_vehicle():
    act = Actuators(propeller=Propeller(.14, .45, position_m=(-.8, 0, 0), gimbal=True))
    straight = wrench(act, [1, 0, 0, 0, 0, 0], [1000., 0, 0])
    assert not straight[1:].any()
    side = wrench(act, [1, 0, 0, 0, 0, 0], [1000., 0, .1])
    up = wrench(act, [1, 0, 0, 0, 0, 0], [1000., .1, 0])
    assert side[1] > 0 and side[5] < 0 and up[2] < 0 and up[4] < 0
    assert np.hypot(side[0], side[1]) == pytest.approx(straight[0])


@pytest.mark.parametrize("roll,pitch", [(0, 0), (.3, -.2), (-.5, .4)])
def test_moving_mass_is_a_centre_of_gravity_shift(roll, pitch):
    travel, ms, m = .05, 6.0, 30.0
    act = Actuators(moving_mass=MovingMass(ms))
    v = VehicleConstants(mass_kg=m, actuators=act)
    a = compute_acceleration(P, v, np.zeros(6), np.zeros(6), roll, pitch, None, [travel])[0]
    b = compute_acceleration(replace(P, cg_cb_offset_x=ms / m * travel), VehicleConstants(mass_kg=m), np.zeros(6), np.zeros(6), roll, pitch)[0]
    assert np.allclose(a, b, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("roll,pitch", [(0, 0), (.3, -.2)])
def test_a_buoyancy_engine_is_a_change_in_net_buoyancy(roll, pitch):
    v = VehicleConstants(mass_kg=30.0, actuators=Actuators(buoyancy_engine=True))
    a = compute_acceleration(P, v, np.zeros(6), np.zeros(6), roll, pitch, None, [1.5])[0]
    b = compute_acceleration(replace(P, net_buoyancy_z=P.net_buoyancy_z + 1.5), VehicleConstants(mass_kg=30.0), np.zeros(6), np.zeros(6), roll, pitch)[0]
    assert np.allclose(a, b, rtol=1e-10, atol=1e-12)


def test_control_columns_follow_the_vehicle():
    act = Actuators(fins=(RUDDER_TOP,), propeller=Propeller(.14, .4, gimbal=True), moving_mass=MovingMass(5.), buoyancy_engine=True)
    assert act.columns() == ["fin_top", "rpm", "gimbal_y", "gimbal_z", "moving_mass", "buoyancy_change"] and act.width == 6


def test_servo_lags_stops_and_limits_speed():
    act = Actuators(fins=(replace(RUDDER_TOP, limit_rad=.3),), servo_rate_rad_s=.5)
    c = np.r_[np.zeros(5), np.full(45, .6)][:, None]
    s = servo_angles(act, c, .2, .1)[:, 0]
    assert s[:5].max() == 0 and s.max() <= .3 + 1e-12 and np.abs(np.diff(s)).max() <= .05 + 1e-12
    assert s[-1] == pytest.approx(.3, abs=1e-3)
