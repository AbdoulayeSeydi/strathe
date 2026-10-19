"""`strathe fit-sim --sim gazebo` / `export-sim` (PROJECT_STATUS.md #82). The file-handling tests need no Gazebo; the
rest run against real Gazebo Harmonic when the replay tool is built (calibration/simulators/gazebo/README.md, or
STRATHE_GZ_REPLAY), and are skipped otherwise (CI)."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

from simfit import cli, core
from simfit.adapters.gazebo import DEFAULT_TOOL, GazeboAdapter
from simfit.core import SimFitError
from simfit.editors import XmlFile

HERE = os.path.dirname(os.path.abspath(__file__))
WORLD = os.path.join(HERE, "..", "simulators", "gazebo", "example", "world.sdf")
TOOL = os.environ.get("STRATHE_GZ_REPLAY") or DEFAULT_TOOL
needs_gazebo = pytest.mark.skipif(not os.access(TOOL, os.X_OK), reason="Gazebo replay tool not built")


def _params(**kw):
    from dynamics_model import CalibrationParameters
    p = dict(Ieff_roll=0.3, Ieff_pitch=0.5, Ieff_yaw=0.6, Xu=-6.0, Yv=-18.0, Zw=-22.0, Kp=-1.5, Mq=-2.5, Nr=-2.0,
             Xuu=-35.0, Yvv=-70.0, Zww=-80.0, Kpp=-1.0, Mqq=-1.2, Nrr=-0.8, Xu_dot=3.0, Yv_dot=8.0, Zw_dot=9.0,
             Xq_dot=0.0, Zq_dot=0.0, net_buoyancy_z=0.0, cg_cb_offset_z=0.02)
    p.update(kw)
    return CalibrationParameters(**p)


def test_the_vehicle_and_its_knobs_are_found_in_the_world():
    ad = GazeboAdapter(WORLD, tool=["unused"])
    assert ad.link == "base_link" and ad.plugin == "//plugin[@name='gz::sim::systems::Hydrodynamics']"
    names = [n for n, *_ in ad._knob_defs()]
    assert {"xU", "xUabsU", "yVabsV", "nRabsR", "cg_x", "cg_z", "mass"} <= set(names)
    assert ad.groups[:3] == ("drag", "buoyancy", "cg") or "buoyancy" in ad.groups
    specs = ad.knob_specs({}, ("drag", "buoyancy"), True)
    s = {k.name: k for k in specs}
    assert s["xUabsU"].start == 30.0 and s["yVabsV"].tie == s["zWabsW"].tie  # magnitudes; sway = heave
    assert s["buoyancy_scale"].start == 1.0


def test_the_calibrated_model_changes_only_the_tuned_numbers(tmp_path):
    ad = GazeboAdapter(WORLD, tool=["unused"])
    out = ad.write_calibrated({"xUabsU": 40.0, "buoyancy_scale": 1.331, "cg_z": -0.01}, str(tmp_path), "calibrated")
    t = open(out).read()
    orig = open(WORLD).read()
    assert "<xUabsU>-40</xUabsU>" in t and "<size>0.55 0.33 0.0733333" in t and "<pose>0 0 -0.01 0 0 0</pose>" in t
    assert t.replace("<xUabsU>-40</xUabsU>", "").count("\n") == orig.count("\n")
    assert "Physics steps must divide" in t  # comments kept
    lines = ad.changes({"xUabsU": 40.0, "yVabsV": 60.0})
    assert any("xUabsU: -30 -> -40" in ln for ln in lines) and not any("yVabsV" in ln for ln in lines)


def test_export_maps_strathes_model_into_the_hydrodynamics_plugin(tmp_path):
    ad = GazeboAdapter(WORLD, tool=["unused"])
    info = {"collisions": [{"name": "hull", "type": "box", "volume": 0.01, "center_flu": [0, 0, 0.02]}]}
    v, notes = ad.export_values(_params(Zuw=-5.0, Muw=1.0, Muq=-2.0, net_buoyancy_z=2.0), 10.0, info, [], 1000.0)
    t = XmlFile(ad._edit(v))
    pl = "//plugin[@name='gz::sim::systems::Hydrodynamics']"
    assert t.get(f"{pl}/xU") == -6.0 and t.get(f"{pl}/zWabsW") == -80.0 and t.get(f"{pl}/nRabsR") == -0.8
    assert t.get(f"{pl}/zUW") == -5.0 and t.get(f"{pl}/nUV") == -1.0 and t.get(f"{pl}/mUQ") == -2.0  # lift, FLU signs
    fa = "//link[@name='base_link']/inertial/fluid_added_mass"
    assert t.get(f"{fa}/xx") == 3.0 and t.get(f"{fa}/zz") == 9.0 and t.get(f"{fa}/pp") == 0.0
    assert t.get("//inertial/inertia/iyy") == 0.5 and t.get("//inertial/mass") == 10.0
    assert v["buoyancy_scale"] == pytest.approx((10 * 9.81 + 2.0) / (1000 * 9.81) / 0.01)
    assert v["collision_shift"] == pytest.approx([0.0, 0.0, 0.0])  # CB already 2 cm above the CG at the origin
    assert not any("reference point is" in n for n in notes)
    ad2 = GazeboAdapter(WORLD, tool=["unused"], ref=(-0.3, 0.0, 0.0))
    v2, notes2 = ad2.export_values(_params(), 10.0, info, [], 1000.0)
    t2 = XmlFile(ad2._edit(v2))
    # damping moved from the reference point 0.3 m aft to the link origin: sway damping makes a yaw moment and back
    assert t2.get(f"{pl}/yR") == pytest.approx(5.4) and t2.get(f"{pl}/nV") == pytest.approx(5.4)
    assert t2.get(f"{pl}/nR") == pytest.approx(-2.0 - 0.09 * 18.0)
    assert any("not exact" in n for n in notes2)


def test_plain_messages(tmp_path):
    bad = tmp_path / "w.sdf"
    bad.write_text("<sdf><world name='w'><model name='m'><link name='l'/></model></world></sdf>")
    with pytest.raises(SimFitError, match="Hydrodynamics plugins"):
        GazeboAdapter(str(bad), tool=["unused"])
    with pytest.raises(SimFitError, match="doesn't exist"):
        GazeboAdapter(str(tmp_path / "nope.sdf"), tool=["unused"])


# ---- with Gazebo ---------------------------------------------------------------------------------------------------

def _window(F, start=(0, 0, 0, 0, 0, 0), q=(0, 0, 0, 1), n=101):
    class Run:
        pass
    run = Run()
    run.dt = 0.1
    run.forces_torques = np.asarray(F, float) if np.ndim(F) == 2 else np.repeat(np.array(F, float)[None], n, axis=0)
    n = len(run.forces_torques)
    run.real_nu = np.zeros((n, 6))
    run.real_nu[0] = start
    qq = np.array(q, float)
    run.orientation_quat = np.repeat((qq / np.linalg.norm(qq))[None], n, axis=0)
    return run


@needs_gazebo
def test_gazebo_frames_and_steady_states():
    ad = GazeboAdapter(WORLD, tool=[TOOL])
    d = tempfile.mkdtemp()
    model = ad.write_runnable(None, d)
    for F, axis, want in (((20, 0, 0, 0, 0, 0), 0, 0.737), ((0, 20, 0, 0, 0, 0), 1, 0.434),
                          ((0, 0, 0, 0, 0, 0.5), 5, 0.2247)):
        run = _window(F)
        (V, Q), = ad.replay(model, core.window_rows(run, [(0, 101)]), d)
        assert V[-1, axis] == pytest.approx(want, abs=2e-3), (F, V[-1])  # forward-right-down, Strathe's frames


@needs_gazebo
@pytest.mark.parametrize("net", [0.0, 2.0])
def test_gazebo_export_reproduces_strathes_model(net):
    """Gazebo's Hydrodynamics is Strathe's damping law: with the reference point at the link origin the exported model
    runs as Strathe's own model does (moving, tilted start; forces and torques; added mass; buoyant)."""
    from dynamics_model import VehicleConstants
    p = _params(net_buoyancy_z=net)
    ad = GazeboAdapter(WORLD, tool=[TOOL])
    v, _ = ad.export_values(p, 10.0, ad.info(), [], 1000.0)
    rng = np.random.default_rng(1)
    F = np.repeat(rng.normal(0, [15, 10, 10, 0.4, 0.4, 0.4], size=(12, 6)), 30, axis=0)[:301]
    run = _window(F, start=(0.3, 0.05, -0.02, 0.01, 0.02, -0.03), q=(0.02, -0.03, 0.2, 1.0))
    d = tempfile.mkdtemp()
    (V, Q), = ad.replay(ad.write_runnable(v, d), core.window_rows(run, [(0, 301)]), d)
    Vs, Qs = core.strathe_free_run(p, VehicleConstants(10.0), run, 0, 301)
    assert np.abs(V - Vs).max() < 0.02, np.abs(V - Vs).max(axis=0)


@needs_gazebo
def test_fit_sim_gazebo_recovers_the_drag_from_logs(tmp_path):
    truth = GazeboAdapter(WORLD, tool=[TOOL])
    tv = {"xUabsU": 30.0, "yVabsV": 60.0, "zWabsW": 60.0}
    data = tmp_path / "logs"
    data.mkdir()
    rng = np.random.default_rng(0)
    for i, date in enumerate(("01-01-2030", "02-01-2030")):
        F = np.repeat(rng.normal(0, [20, 15, 15, 0.3, 0.3, 0.3], size=(40, 6)), 30, axis=0)[:1201]
        run = _window(F, start=(0.2, 0, 0, 0, 0, 0))
        d = str(tmp_path / f"g{i}")
        (V, Q), = truth.replay(truth.write_runnable(tv, d), core.window_rows(run, [(0, len(F))]), d)
        pd.DataFrame(np.hstack([F, V, Q]), columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bu", "Bv", "Bw", "Bp", "Bq",
                                                    "Br", "qx", "qy", "qz", "qw"]).to_csv(data / f"auv-{date}-run1.csv",
                                                                                         index=False)
    wrong = tmp_path / "world.sdf"
    wrong.write_text(open(WORLD).read().replace("<xUabsU>-30</xUabsU>", "<xUabsU>-10</xUabsU>")
                     .replace("<yVabsV>-60</yVabsV>", "<yVabsV>-25</yVabsV>"))
    out = tmp_path / "out"
    rc = cli.main(["fit", "--sim", "gazebo", "--world", str(wrong), "--replay-tool", TOOL, "--data", str(data),
                   "--reference-point", "0", "0", "0", "--heldout-dates", "02-01-2030", "--fit", "drag",
                   "--max-iterations", "6", "--workers", "4", "--out", str(out)])
    assert rc == 0
    s = yaml.safe_load((out / "gazebo_fit.yaml").read_text())
    held = s["heldout"]
    assert held["Gazebo, calibrated by Strathe"]["30s"]["drift_median_m"] < 0.5 * held["Gazebo, original"]["30s"]["drift_median_m"]
    assert abs(np.log(s["after"]["xUabsU"] / 30)) < abs(np.log(10 / 30))
    assert "<xUabsU>-" in (out / "world_calibrated.sdf").read_text()
