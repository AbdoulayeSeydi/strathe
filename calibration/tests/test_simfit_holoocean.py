"""`strathe fit-sim --sim holoocean` / `export-sim` (PROJECT_STATUS.md #84). HoloOcean doesn't install on macOS or in CI,
so these use tests/fake_holoocean (HoloOcean's Fossen interface with a simple model); the real check runs on Windows
or Linux with HoloOcean (calibration/simulators/holoocean/README.md)."""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO = os.path.join(HERE, "..", "simulators", "holoocean", "example", "scenario.json")


@pytest.fixture(autouse=True)
def fake_holoocean(monkeypatch):
    monkeypatch.syspath_prepend(os.path.join(HERE, "fake_holoocean"))
    import sys
    for m in [m for m in sys.modules if m.startswith("holoocean")]:
        monkeypatch.delitem(sys.modules, m)


def _rows(F, start=(0, 0, 0, 0, 0, 0), q=(0, 0, 0, 1)):
    F = np.asarray(F, float)
    rows = np.zeros((len(F), 16))
    rows[:, 10:] = F
    rows[0, :6], rows[0, 6:10] = start, np.array(q) / np.linalg.norm(q)
    return rows


def _scenario(tmp_path, **dyn):
    sc = json.load(open(SCENARIO))
    sc["agents"][0]["dynamics"].update(dyn)
    p = tmp_path / "scenario.json"
    p.write_text(json.dumps(sc, indent=2))
    return str(p)


def test_the_agent_and_its_knobs_come_from_the_scenario_and_the_class():
    from simfit.adapters.holoocean import HolooceanAdapter
    ad = HolooceanAdapter(SCENARIO)
    assert ad.agent == "auv0"
    specs = {k.name: k for k in ad.knob_specs({}, ("drag", "added_mass", "cg", "mass"), False)}
    assert specs["T_surge"].start == 20 and specs["zeta_roll"].start == 0.3  # from the scenario, and the class default
    assert specs["r_bg/2"].start == 0.02 and specs["mass"].start == 30.0
    out = ad.write_calibrated({"T_surge": 12.0, "zeta_roll": 0.5, "r_bg/2": 0.03}, tempfile.mkdtemp(), "calibrated")
    dyn = json.load(open(out))["agents"][0]["dynamics"]
    assert dyn["T_surge"] == 12.0 and dyn["zeta_roll"] == 0.5 and dyn["r_bg"] == [0, 0, 0.03] and dyn["Cd"] == 0.42


def test_fit_sim_holoocean_recovers_the_time_constant_and_drag(tmp_path):
    from simfit import cli
    from simfit.adapters.holoocean import HolooceanAdapter
    (tmp_path / "t").mkdir()
    truth = HolooceanAdapter(_scenario(tmp_path / "t", T_surge=8.0, Cd=0.8))
    data = tmp_path / "logs"
    data.mkdir()
    rng = np.random.default_rng(0)
    for i, date in enumerate(("01-01-2030", "02-01-2030")):
        F = np.repeat(rng.normal(0, [30, 10, 10, 0.3, 0.3, 0.3], size=(30, 6)), 30, axis=0)[:901]
        d = str(tmp_path / f"g{i}")
        (V, Q), = truth.replay(truth.write_runnable(None, d), [_rows(F, (0.5, 0, 0, 0, 0, 0))], d)
        pd.DataFrame(np.hstack([F, V, Q]), columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bu", "Bv", "Bw", "Bp", "Bq",
                                                    "Br", "qx", "qy", "qz", "qw"]).to_csv(data / f"auv-{date}-run1.csv",
                                                                                         index=False)
    out = tmp_path / "out"
    rc = cli.main(["fit", "--sim", "holoocean", "--scenario", SCENARIO, "--data", str(data), "--reference-point", "0",
                   "0", "0", "--heldout-dates", "02-01-2030", "--fit", "drag", "--hold", "T_sway=20", "T_yaw=1",
                   "zeta_roll=0.3", "zeta_pitch=0.8", "K_nomoto=0.25", "--max-iterations", "8", "--workers", "2",
                   "--out", str(out)])
    assert rc == 0
    s = yaml.safe_load((out / "holoocean_fit.yaml").read_text())
    held = s["heldout"]
    assert held["HoloOcean, calibrated by Strathe"]["30s"]["drift_median_m"] < 0.5 * held["HoloOcean, original"]["30s"]["drift_median_m"]
    assert abs(np.log(s["after"]["T_surge"] / 8.0)) < abs(np.log(20.0 / 8.0))


@pytest.mark.parametrize("ref", [(0.0, 0.0, 0.0), (-0.4, 0.02, 0.05)])
def test_export_writes_strathes_model_as_a_fossen_class_that_runs_like_it(tmp_path, ref):
    from dynamics_model import CalibrationParameters, VehicleConstants
    from simfit import core
    from simfit.adapters.holoocean import HolooceanAdapter
    p = CalibrationParameters(Ieff_roll=0.6, Ieff_pitch=8.0, Ieff_yaw=8.0, Xu=-6.0, Yv=-30.0, Zw=-30.0, Kp=-2.0,
                              Mq=-15.0, Nr=-15.0, Xuu=-40.0, Yvv=-150.0, Zww=-150.0, Kpp=-1.0, Mqq=-20.0, Nrr=-20.0,
                              Xu_dot=3.0, Yv_dot=60.0, Zw_dot=60.0, Xq_dot=0.0, Zq_dot=0.0, net_buoyancy_z=3.0,
                              cg_cb_offset_z=0.01, Zuw=-10.0, Muw=2.0, Muq=-1.0)
    ad = HolooceanAdapter(SCENARIO, ref=ref)
    values, notes = ad.export_values(p, 30.0, {}, [], 1026.0)
    assert any("Fossen class" in n for n in notes)
    rng = np.random.default_rng(2)
    F = np.repeat(rng.normal(0, [20, 10, 10, 0.5, 0.5, 0.5], size=(12, 6)), 30, axis=0)[:301]

    class Run:
        pass
    run = Run()
    run.dt, run.forces_torques = 0.1, F
    run.real_nu = np.zeros((301, 6))
    run.real_nu[0] = (0.6, 0.05, -0.02, 0.01, 0.02, -0.03)
    q0 = np.array((0.02, -0.03, 0.2, 1.0))
    run.orientation_quat = np.repeat((q0 / np.linalg.norm(q0))[None], 301, axis=0)
    d = str(tmp_path / "e")
    (V, Q), = ad.replay(ad.write_runnable(values, d), core.window_rows(run, [(0, 301)]), d)
    Vs, Qs = core.strathe_free_run(p, VehicleConstants(30.0), run, 0, 301)
    assert np.abs(V - Vs).max() < 0.02, np.abs(V - Vs).max(axis=0)
    out = ad.write_calibrated(values, str(tmp_path), "strathe")
    src = (tmp_path / "strathe_fossen.py").read_text()
    assert "class StratheFossen" in src and "import strathe" not in src and os.path.exists(out)


def test_plain_messages(tmp_path):
    from simfit.adapters.holoocean import HolooceanAdapter
    from simfit.core import SimFitError
    with pytest.raises(SimFitError, match="can't be imported"):
        HolooceanAdapter(SCENARIO, model="no_such_pkg.fossen:Torpedo")
    bad = tmp_path / "s.json"
    bad.write_text('{"agents": [{"agent_name": "a"}]}')
    with pytest.raises(SimFitError, match="Fossen agents"):
        HolooceanAdapter(str(bad))


def test_a_value_the_scenario_has_at_zero_starts_from_the_class(tmp_path):
    """#90: a factor of 0 stays 0, so a zero starts from the Fossen class's own value; with no value anywhere, a plain
    message asks for one."""
    from simfit.adapters.holoocean import HolooceanAdapter
    specs = {k.name: k for k in HolooceanAdapter(_scenario(tmp_path, T_surge=0)).knob_specs({}, ("drag",), False)}
    assert specs["T_surge"].start == 0 and specs["T_surge"].if_zero > 0 and "class" in specs["T_surge"].note
    ad = HolooceanAdapter(_scenario(tmp_path, T_surge=0))
    setattr(ad._class_defaults, "T_surge", 0.0)
    from simfit.core import SimFitError
    with pytest.raises(SimFitError, match="T_surge is 0 in your scenario and in the Fossen class"):
        ad.knob_specs({}, ("drag",), False)
