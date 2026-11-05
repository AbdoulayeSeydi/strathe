"""Bugs the Fin Steer audit found, before any fin code (#97): each stays fixed."""
import re

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

import calibration_report as cr
import objective
from data_ingestion import IngestSpec
from dynamics_model import _PARAM_ORDER, CalibrationParameters, NonPhysicalMass, VehicleConstants
from objective import RunData, DIVERGED_PENALTY
from simfit import core
from simfit.adapters.external import load_description
from validation import free_run

VEHICLE = VehicleConstants(mass_kg=10.0)
P = CalibrationParameters(Ieff_roll=.5, Ieff_pitch=.6, Ieff_yaw=.7, Xu=-5, Yv=-6, Zw=-7, Kp=-1, Mq=-1, Nr=-1, Xuu=-5, Yvv=-6,
                          Zww=-7, Kpp=-.5, Mqq=-.5, Nrr=-.5, Xu_dot=2, Yv_dot=3, Zw_dot=3, net_buoyancy_z=.2,
                          cg_cb_offset_z=.02)


def _run(n=120, lag=False, current=False):
    rng = np.random.default_rng(3)
    F = np.repeat(rng.normal(size=(n // 10 + 1, 6)) * 3, 10, axis=0)[:n]
    q = Rotation.from_euler("xyz", np.c_[.1 * np.sin(np.arange(n) / 20), .05 * np.cos(np.arange(n) / 30), np.arange(n) / 50]).as_quat()
    rp = np.c_[.1 * np.sin(np.arange(n) / 20), .05 * np.cos(np.arange(n) / 30)]
    real = np.tile([.5, .02, -.01, .01, .02, .03], (n, 1))
    cb = np.tile([.2, -.1, .05], (n, 1)) if current else None
    return RunData("a-01-01-2030-run1.csv", F, rp, real, real[0].copy(), .1, 10.0, current_body=cb, orientation_quat=q)


def test_every_check_ingest_can_raise_can_be_skipped():
    src = open(objective.__file__.replace("objective.py", "data_ingestion.py"), encoding="utf-8").read()
    names = set(re.findall(r'checks(?:\.setdefault\(|\[)"(\w+)"', src))
    allowed = set(IngestSpec.model_fields["skip_checks"].annotation.__args__[0].__args__)
    assert "depth_vs_thrust" in names and names <= allowed, names - allowed


def test_every_parameter_has_a_plain_name():
    assert set(_PARAM_ORDER) <= set(cr.PLAIN), set(_PARAM_ORDER) - set(cr.PLAIN)


def test_a_coupling_at_a_limit_is_tied_to_the_axes_it_moves():
    assert "Xq_dot" in cr.AXIS_PARAMS["u"] and "Zq_dot" in cr.AXIS_PARAMS["w"]


def test_a_description_can_give_if_zero(tmp_path):
    f = tmp_path / "sim.yaml"
    f.write_text(yaml.safe_dump({"replay": "x", "model": "m.json", "knobs": [
        {"name": "drag", "group": "drag", "at": "/drag", "if_zero": 5.0}]}))
    assert load_description(str(f))["knobs"][0]["if_zero"] == 5.0


def test_a_nonphysical_mass_is_a_bad_fit_and_doesnt_stop_the_calibration():
    from dataclasses import replace
    bad = replace(P, Xq_dot=100.0)
    with pytest.raises(NonPhysicalMass):
        objective.simulate_analytical(bad, _run(), VEHICLE) if hasattr(objective, "simulate_analytical") else \
            __import__("dynamics_model")._mass_and_damping_arrays(bad, VEHICLE)
    run = _run()
    r = objective.residuals_for_run(bad, run, VEHICLE, normalize_by_run_length=False, horizon_samples=20)
    good = objective.residuals_for_run(P, run, VEHICLE, normalize_by_run_length=False, horizon_samples=20)
    assert r.shape == good.shape and np.all(np.isfinite(r))
    assert np.abs(r).max() >= DIVERGED_PENALTY * np.abs(run.real_nu).max() * .99


@pytest.mark.parametrize("lag,current", [(False, False), (True, False), (False, True), (True, True)])
def test_strathes_free_run_in_a_simulator_comparison_is_validations(lag, current):
    """fit-sim's "Strathe's own model" line and validation's free run are the same model: lag and current included."""
    from dataclasses import replace
    p = replace(P, thruster_time_constant_s=.4 if lag else 0.0)
    run = _run(lag=lag, current=current)
    _, Q = core.strathe_free_run(p, VEHICLE, run, 10, 60)
    rpy = Rotation.from_quat(Q).as_euler("xyz")
    ref = free_run(p, VEHICLE, run, 10, 60)
    assert ref is not None and np.abs(np.unwrap(rpy[1:], axis=0) - np.unwrap(ref[1:], axis=0)).max() < 1e-9


def test_the_free_run_feels_the_lag_and_the_current():
    from dataclasses import replace
    run = _run(current=True)
    still = _run()
    a = core.strathe_free_run(replace(P, thruster_time_constant_s=1.0), VEHICLE, still, 10, 60)[0]
    b = core.strathe_free_run(P, VEHICLE, still, 10, 60)[0]
    c = core.strathe_free_run(P, VEHICLE, run, 10, 60)[0]
    assert np.abs(a - b).max() > 1e-3 and np.abs(c - b).max() > 1e-3
