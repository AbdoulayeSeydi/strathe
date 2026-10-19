"""`strathe fit-sim --sim mujoco` (PROJECT_STATUS.md #83): a lab's own MJCF, calibrated through MuJoCo's fluid model.
MuJoCo is a pip package, so these run everywhere, CI included."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

from simfit import cli
from simfit.adapters.mujoco import MujocoAdapter
from simfit.core import SimFitError
from simfit.editors import XmlFile

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "simulators", "mujoco", "example", "auv.xml")


def _rows(F, start=(0, 0, 0, 0, 0, 0), q=(0, 0, 0, 1), n=101):
    F = np.asarray(F, float)
    F = F if F.ndim == 2 else np.repeat(F[None], n, axis=0)
    rows = np.zeros((len(F), 16))
    rows[:, 10:] = F
    rows[0, :6], rows[0, 6:10] = start, np.array(q) / np.linalg.norm(q)
    return rows


def test_frames_match_strathes():
    ad = MujocoAdapter(MODEL)
    d = tempfile.mkdtemp()
    model = ad.write_runnable(None, d)
    for i, F in enumerate(np.eye(6) * [20, 20, 20, 0.3, 0.3, 0.5]):
        (V, Q), = ad.replay(model, [_rows(F)], d)
        assert V[-1, i] > 0.1 and np.abs(np.delete(V[-1], i)).max() < 1e-6, (F, V[-1])  # pushes to starboard -> +v
    (V, Q), = ad.replay(model, [_rows((0, 0, 0, 0, 0, 0.5))], d)
    assert Q[-1, 2] > 0.1  # a positive yaw rate turns clockwise from above (north-east-down)


def test_the_start_state_round_trips_at_an_offset_reference_point(tmp_path):
    """No water (density and viscosity 0): over 0.1 s the vehicle keeps the logged velocity at the reference point."""
    m = tmp_path / "dry.xml"
    m.write_text(open(MODEL).read().replace('density="1000" viscosity="0.0009"', 'density="0" viscosity="0"'))
    ad = MujocoAdapter(str(m), ref=(-0.3, 0.1, 0.05))
    d = str(tmp_path / "r")
    start = (0.4, -0.1, 0.05, 0.02, -0.03, 0.05)
    (V, Q), = ad.replay(ad.write_runnable(None, d), [_rows(np.zeros(6), start, (0.05, -0.02, 0.3, 1), n=2)], d)
    assert V[1] == pytest.approx(np.array(start), abs=0.01)
    assert Q[1] == pytest.approx(np.array((0.05, -0.02, 0.3, 1)) / np.linalg.norm((0.05, -0.02, 0.3, 1)), abs=0.01)


def test_knobs_come_from_the_file_and_only_they_change(tmp_path):
    m = tmp_path / "auv.xml"
    m.write_text(open(MODEL).read().replace(' fluidcoef="0.5 0.25 1.5 1.0 1.0"', ""))  # MJCF default, made explicit
    ad = MujocoAdapter(str(m))
    names = [n for n, *_ in ad._knob_defs()]
    assert {"hull_blunt_drag", "hull_kutta_lift", "gravcomp", "cg_x", "mass"} <= set(names)
    assert ad.groups == ("drag", "lift", "buoyancy", "cg", "mass")
    out = ad.write_calibrated({"hull_blunt_drag": 0.8, "gravcomp": 1.02}, str(tmp_path), "calibrated")
    t = open(out).read()
    assert 'fluidcoef="0.8 0.25 1.5 1.0 1.0"' in t and 'gravcomp="1.02"' in t
    assert "An example underwater vehicle" in t
    assert XmlFile(t).get("//inertial/@mass") == 10.0
    with pytest.raises(SimFitError, match="can't take Strathe's model"):
        ad.export_values(None, 10.0, {}, [], 1000.0)


def test_plain_messages(tmp_path):
    bad = tmp_path / "x.xml"
    bad.write_text("<mujoco><worldbody><body name='a'/></worldbody></mujoco>")
    with pytest.raises(SimFitError, match="free joint"):
        MujocoAdapter(str(bad))
    off = tmp_path / "step.xml"
    off.write_text(open(MODEL).read().replace('timestep="0.002"', 'timestep="0.003"'))
    ad = MujocoAdapter(str(off))
    with pytest.raises(SimFitError, match="whole number of the model's timestep"):
        ad.replay(ad.write_runnable(None, str(tmp_path / "s")), [_rows(np.zeros(6), n=3)], str(tmp_path / "s"))


def test_fit_sim_mujoco_recovers_the_fluid_coefficients(tmp_path):
    truth = MujocoAdapter(MODEL)
    tv = {"hull_blunt_drag": 0.9, "hull_slender_drag": 0.5}
    data = tmp_path / "logs"
    data.mkdir()
    rng = np.random.default_rng(0)
    for i, date in enumerate(("01-01-2030", "02-01-2030")):
        F = np.repeat(rng.normal(0, [15, 12, 12, 0.2, 0.2, 0.2], size=(30, 6)), 30, axis=0)[:901]
        d = str(tmp_path / f"g{i}")
        (V, Q), = truth.replay(truth.write_runnable(tv, d), [_rows(F, (0.2, 0, 0, 0, 0, 0))], d)
        pd.DataFrame(np.hstack([F, V, Q]), columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bu", "Bv", "Bw", "Bp", "Bq",
                                                    "Br", "qx", "qy", "qz", "qw"]).to_csv(data / f"auv-{date}-run1.csv",
                                                                                         index=False)
    out = tmp_path / "out"
    rc = cli.main(["fit", "--sim", "mujoco", "--model", MODEL, "--data", str(data), "--reference-point", "0", "0", "0",
                   "--heldout-dates", "02-01-2030", "--fit", "drag", "--max-iterations", "8", "--workers", "4",
                   "--out", str(out)])
    assert rc == 0
    s = yaml.safe_load((out / "mujoco_fit.yaml").read_text())
    held = s["heldout"]
    assert held["MuJoCo, calibrated by Strathe"]["30s"]["drift_median_m"] < 0.5 * held["MuJoCo, original"]["30s"]["drift_median_m"]
    assert abs(np.log(s["after"]["hull_blunt_drag"] / 0.9)) < abs(np.log(0.5 / 0.9))
    assert (out / "auv_calibrated.xml").exists() and (out / "mujoco_fit_report.md").exists()
    rc = cli.main(["export", "--sim", "mujoco", "--model", MODEL, "--calibration", str(tmp_path), "--data", str(data),
                   "--reference-point", "0", "0", "0", "--out", str(tmp_path / "e")])
    assert rc == 2
