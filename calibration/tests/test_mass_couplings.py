"""The general mass matrix (#97): all 15 off-diagonal entries, one builder, everywhere it's used."""
import copy
import os
from dataclasses import replace

import numpy as np
import pytest
import yaml

from dynamics_model import (CalibrationParameters, MASS_COUPLINGS, NonPhysicalMass, VehicleConstants, XZ_SYMMETRIC_COUPLINGS,
                            _PARAM_ORDER, compute_acceleration, mass_matrix)
from fast_dynamics import FastDynamics
from optimizer import held_by_bounds, initial_parameters, load_vehicle_spec, parameter_bounds, worst_mass_margin

CAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = VehicleConstants(mass_kg=30.0)
P = CalibrationParameters(Ieff_roll=2, Ieff_pitch=9, Ieff_yaw=9, Xu=-5, Yv=-6, Zw=-7, Kp=-1, Mq=-1, Nr=-1, Xuu=-5, Yvv=-6, Zww=-7,
                          Kpp=-.5, Mqq=-.5, Nrr=-.5, Xu_dot=3, Yv_dot=30, Zw_dot=30, net_buoyancy_z=.2, cg_cb_offset_z=.02)
NAMES = [n for _, _, n in MASS_COUPLINGS]


def _spec(tmp_path, **coupling):
    raw = yaml.safe_load(open(os.path.join(CAL, "vehicle_spec.yaml"), encoding="utf-8"))
    raw["calibration_targets"]["added_mass_coupling"].update(coupling)
    f = tmp_path / "spec.yaml"
    f.write_text(yaml.safe_dump(raw))
    return str(f)


def _entry(lo, hi, guess=0.0):
    return {"unit": "kg", "initial_guess": guess, "bounds": [lo, hi], "bounds_rationale": "test", "source": "test",
            "confidence": "assumed"}


def test_the_table_covers_the_whole_upper_triangle_once():
    assert sorted((i, j) for i, j, _ in MASS_COUPLINGS) == [(i, j) for i in range(6) for j in range(i + 1, 6)]
    assert len(set(NAMES)) == 15 and set(NAMES) <= set(_PARAM_ORDER) and set(XZ_SYMMETRIC_COUPLINGS) <= set(NAMES)


@pytest.mark.parametrize("i,j,name", MASS_COUPLINGS)
def test_each_entry_lands_where_its_name_says_and_only_there(i, j, name):
    M = mass_matrix(replace(P, **{name: .25}), V)
    want = mass_matrix(P, V)
    want[i, j] = want[j, i] = .25
    assert np.array_equal(M, want) and np.array_equal(M, M.T)


def test_saved_values_from_before_load_and_mean_the_same():
    a = np.array(P.to_vector())
    assert a.size == 40 and not a[-13:].any()
    assert np.array_equal(mass_matrix(P, V), np.diag([33., 60, 60, 2, 9, 9]))


@pytest.mark.parametrize("seed", range(12))
def test_fast_and_reference_physics_agree_with_any_coupling(seed):
    rng = np.random.default_rng(seed)
    p = replace(P, **{n: float(rng.uniform(-.3, .3)) for n in NAMES})
    v, tau, (r, q) = rng.normal(size=6) * .4, rng.normal(size=6) * 8, rng.normal(size=2) * .2
    ref = compute_acceleration(p, V, v, tau, r, q)[0]
    assert np.allclose(FastDynamics(p, V).acceleration(list(v), list(tau), r, q), ref, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("seed", range(6))
def test_coriolis_does_no_work_with_any_coupling(seed):
    """With only inertia at work the kinetic energy can't change: v.(C(v)v) = 0 whatever M is, so a wrong term shows."""
    rng = np.random.default_rng(seed)
    z = dict(Xu=0, Yv=0, Zw=0, Kp=0, Mq=0, Nr=0, Xuu=0, Yvv=0, Zww=0, Kpp=0, Mqq=0, Nrr=0, net_buoyancy_z=0, cg_cb_offset_z=0)
    p = replace(P, **z, **{n: float(rng.uniform(-.3, .3)) for n in NAMES})
    M, v = mass_matrix(p, V), rng.normal(size=6) * .5
    a = compute_acceleration(p, V, v, np.zeros(6), 0.0, 0.0)[0]
    assert abs(v @ M @ a) < 1e-9


def test_a_nonphysical_entry_is_named():
    with pytest.raises(NonPhysicalMass, match="Yr_dot=100"):
        mass_matrix(replace(P, Yr_dot=100.0), V)


def test_a_spec_lists_a_coupling_and_it_can_be_fitted(tmp_path):
    s = load_vehicle_spec(_spec(tmp_path, Yr_dot=_entry(-.3, .3, .1)))
    p = initial_parameters(s)
    assert p.Yr_dot == .1 and "Yr_dot" not in held_by_bounds(*parameter_bounds(s), list(p.param_names))


def test_a_spec_without_them_holds_them_at_zero():
    s = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml"))
    held = held_by_bounds(*parameter_bounds(s), list(initial_parameters(s).param_names))
    assert all(held[n] == 0.0 for n in NAMES if n not in ("Xq_dot", "Zq_dot"))


@pytest.mark.parametrize("name", [n for n in NAMES if n not in XZ_SYMMETRIC_COUPLINGS])
def test_an_asymmetric_coupling_needs_symmetry_none(tmp_path, name):
    with pytest.raises(ValueError, match="symmetry: none"):
        load_vehicle_spec(_spec(tmp_path, **{name: _entry(-.1, .1)}))
    assert getattr(initial_parameters(load_vehicle_spec(_spec(tmp_path, symmetry="none", **{name: _entry(-.1, .1, .05)}))), name) == .05


def test_a_wrong_symmetry_word_is_refused(tmp_path):
    with pytest.raises(ValueError, match="isn't xz"):
        load_vehicle_spec(_spec(tmp_path, symmetry="port"))


def test_the_margin_finds_limits_that_reach_a_nonphysical_mass(tmp_path, caplog):
    s = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml"))
    assert worst_mass_margin(s.flat_calibration_targets, s.vehicle.mass_kg) > 0
    flat = copy.deepcopy(s.flat_calibration_targets)
    flat["Yr_dot"]["bounds"] = [-50.0, 50.0]
    assert worst_mass_margin(flat, s.vehicle.mass_kg) <= 0
    load_vehicle_spec(_spec(tmp_path, Yr_dot=_entry(-50, 50)))
    assert "non-physical" in caplog.text


def test_the_margin_agrees_with_cholesky_on_random_limits():
    s = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml"))
    rng = np.random.default_rng(0)
    for _ in range(60):
        flat = copy.deepcopy(s.flat_calibration_targets)
        b = float(rng.uniform(.05, 1.5))
        for n in ("Xq_dot", "Zq_dot"):
            flat[n]["bounds"] = [-b, b]
        got = worst_mass_margin(flat, s.vehicle.mass_kg) > 0
        d = [s.vehicle.mass_kg + flat[k]["bounds"][0] for k in ("Xu_dot", "Yv_dot", "Zw_dot")] + \
            [flat[k]["bounds"][0] for k in ("Ieff_roll", "Ieff_pitch", "Ieff_yaw")]
        ok = True
        for sx in (-b, b):
            for sz in (-b, b):
                M = np.diag(d)
                M[0, 4] = M[4, 0] = sx
                M[2, 4] = M[4, 2] = sz
                ok &= bool(np.linalg.eigvalsh(M)[0] > 0)
        assert got == ok


@pytest.mark.parametrize("i,j,name", MASS_COUPLINGS)
def test_gazebo_gets_every_entry_in_its_own_axes(i, j, name):
    """Gazebo's frame flips y and z: the export writes each entry with that sign, under Gazebo's own name for it."""
    from simfit.adapters.gazebo import GazeboAdapter
    from simfit.editors import XmlFile
    ad = GazeboAdapter(os.path.join(CAL, "simulators", "gazebo", "example", "world.sdf"), tool=["unused"])
    info = {"collisions": [{"name": "hull", "type": "box", "volume": 0.01, "center_flu": [0, 0, 0.02]}]}
    v, _ = ad.export_values(replace(P, **{name: .25}), 30.0, info, [], 1000.0)
    fa = "//link[@name='base_link']/inertial/fluid_added_mass"
    flip = (1, -1, -1, 1, -1, -1)
    assert XmlFile(ad._edit(v)).get(f"{fa}/{'xyzpqr'[i]}{'xyzpqr'[j]}") == pytest.approx(flip[i] * flip[j] * .25)


def test_holooceans_generated_class_builds_the_same_matrix(tmp_path):
    from simfit.adapters.holoocean import strathe_fossen_source
    p = replace(P, Yr_dot=.2, Zq_dot=-.1, Kr_dot=.05)
    ns = {}
    exec(strathe_fossen_source(p, 30.0, [0, 0, 0]), ns)
    assert np.allclose(ns["StratheFossen"]().M_ref, mass_matrix(p, V))


def test_stonefish_says_which_couplings_it_dropped():
    src = open(os.path.join(CAL, "stonefish_export.py"), encoding="utf-8").read()
    assert "for _, _, n in MASS_COUPLINGS" in src and "not carried" in src
