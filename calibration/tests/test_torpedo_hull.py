"""
Torpedo-shaped hulls (PROJECT_STATUS.md #71): hull lift, its moment, speed-dependent rotational damping and a fore-aft
offset of the centre of gravity. Off (all 0) they leave every other vehicle exactly as before; on, every physics path
agrees and the signs are the physical ones.
"""
import dataclasses
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dynamics_model import (CalibrationParameters, VehicleConstants, compute_acceleration,
                            compute_acceleration_from_rotation_matrix)
from fast_dynamics import FastDynamics
from optimizer import held_by_bounds, initial_parameters, load_vehicle_spec, parameter_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
TEMPLATE = os.path.join(CAL, "templates", "vehicle_spec_template.yaml")
HULL = ("cg_cb_offset_x", "Zuw", "Muw", "Muq")


def _base():
    return initial_parameters(load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml")))


def _torpedo(**kw):
    vals = dict(cg_cb_offset_x=0.02, Zuw=-30.0, Muw=5.0, Muq=-8.0)
    vals.update(kw)
    return dataclasses.replace(_base(), **vals)


def test_every_physics_path_agrees_with_the_hull_terms_on():
    rng = np.random.default_rng(1)
    veh = VehicleConstants(33.0)
    for _ in range(200):
        p = _torpedo(cg_cb_offset_x=rng.uniform(-.05, .05), Zuw=rng.uniform(-60, 0), Muw=rng.uniform(-20, 20),
                     Muq=rng.uniform(-20, 0))
        v, tau = rng.normal(0, .5, 6), rng.normal(0, 10, 6)
        roll, pitch, yaw = rng.uniform(-.6, .6, 3)
        ref = compute_acceleration(p, veh, v, tau, roll, pitch)[0]
        fast = FastDynamics(p, veh).acceleration(list(v), list(tau), roll, pitch)
        R = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
        rot = compute_acceleration_from_rotation_matrix(p, veh, v, tau, R)[0]
        assert np.allclose(ref, fast, atol=1e-10) and np.allclose(ref, rot, atol=1e-10)


def test_off_the_model_is_exactly_the_previous_one():
    p, veh = _base(), VehicleConstants(11.2)
    assert all(getattr(p, n) == 0.0 for n in HULL)
    rng = np.random.default_rng(2)
    v, tau = rng.normal(0, .5, 6), rng.normal(0, 10, 6)
    a = compute_acceleration(p, veh, v, tau, 0.1, -0.2)[0]
    # the same computation with the hull helper bypassed: identical bits
    M = np.diag([11.2 + p.Xu_dot, 11.2 + p.Yv_dot, 11.2 + p.Zw_dot, p.Ieff_roll, p.Ieff_pitch, p.Ieff_yaw])
    M[0, 4] = M[4, 0] = p.Xq_dot
    M[2, 4] = M[4, 2] = p.Zq_dot
    from dynamics_model import _coriolis_centripetal_force, _restoring_forces
    lin = np.array([p.Xu, p.Yv, p.Zw, p.Kp, p.Mq, p.Nr])
    quad = np.array([p.Xuu, p.Yvv, p.Zww, p.Kpp, p.Mqq, p.Nrr])
    damping = lin * v + quad * np.abs(v) * v  # the same operations, in the same order, as before #71
    b = np.linalg.solve(M, tau + damping - _restoring_forces(0.1, -0.2, 11.2, p.net_buoyancy_z, p.cg_cb_offset_z)
                        - _coriolis_centripetal_force(M, v))
    assert np.array_equal(a, b)


def _force(p, v, roll=0.0, pitch=0.0):
    """The net hydrodynamic + restoring force/moment (no thrust), body FRD."""
    veh = VehicleConstants(33.0)
    v_dot = compute_acceleration(p, veh, np.asarray(v, float), np.zeros(6), roll, pitch)[0]
    from dynamics_model import _mass_and_damping_arrays
    return _mass_and_damping_arrays(p, veh)[0] @ v_dot


def test_signs_are_physical():
    off = _torpedo(cg_cb_offset_x=0.0, Zuw=0.0, Muw=0.0, Muq=0.0)
    # centre of gravity ahead of buoyancy: nose-down (negative) pitch moment when level, none when not offset
    assert _force(_torpedo(Zuw=0.0, Muw=0.0, Muq=0.0), np.zeros(6))[4] < 0 == _force(off, np.zeros(6))[4]
    # lift opposes heave and sway at forward speed, and vanishes when stopped
    moving = [0.8, 0.1, 0.1, 0, 0, 0]
    d = _force(_torpedo(cg_cb_offset_x=0.0, Muw=0.0, Muq=0.0), moving) - _force(off, moving)
    assert d[2] < 0 and d[1] < 0
    still = [0.0, 0.1, 0.1, 0, 0, 0]
    assert np.allclose(_force(_torpedo(cg_cb_offset_x=0.0, Muw=0.0, Muq=0.0), still), _force(off, still))
    # speed-dependent damping opposes pitch and yaw rates when moving forward
    turning = [0.8, 0, 0, 0, 0.2, 0.2]
    d = _force(_torpedo(cg_cb_offset_x=0.0, Zuw=0.0, Muw=0.0), turning) - _force(off, turning)
    assert d[4] < 0 and d[5] < 0
    # the lift moment acts in pitch for heave and the opposite way in yaw for sway (axisymmetric hull)
    lm = _torpedo(cg_cb_offset_x=0.0, Zuw=0.0, Muq=0.0)
    dw = _force(lm, [0.8, 0, 0.1, 0, 0, 0]) - _force(off, [0.8, 0, 0.1, 0, 0, 0])
    dv = _force(lm, [0.8, 0.1, 0, 0, 0, 0]) - _force(off, [0.8, 0.1, 0, 0, 0, 0])
    assert dw[4] == pytest.approx(5.0 * 0.8 * 0.1) and dv[5] == pytest.approx(-5.0 * 0.8 * 0.1)


def test_specs_hold_them_or_fit_them():
    spec = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml"))
    lb, ub = parameter_bounds(spec)
    held = held_by_bounds(lb, ub, initial_parameters(spec).param_names)
    assert all(held[n] == 0.0 for n in HULL)  # a BlueROV2 doesn't fit them
    t = load_vehicle_spec(TEMPLATE)
    lb, ub = parameter_bounds(t)
    assert not set(HULL) & set(held_by_bounds(lb, ub, initial_parameters(t).param_names))  # the torpedo template does


def test_older_calibrations_still_load():
    old = {n: getattr(_base(), n) for n in _base().param_names if n not in HULL}
    p = CalibrationParameters(**old)
    assert all(getattr(p, n) == 0.0 for n in HULL)
