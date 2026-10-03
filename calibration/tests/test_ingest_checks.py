"""
The data checks ingestion now runs on every lab's logs (PROJECT_STATUS.md #63): things found by hand on DFKI's
BlueROV2 (#62) that the tool should catch on the next lab's data.
"""
import glob
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from data_ingestion import IngestError, ingest, load_ingest_spec, orientation_vs_position, reference_run_in_frd

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
REAL = sorted(glob.glob(os.path.join(os.path.dirname(CAL), "data", "raw", "Bluerov2_trajectory_dataset", "dataset",
                                     "*.csv")))
COLS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def _logged_force_spec(tmp_path, **over):
    spec = {"vehicle": "chk", "source": {"files": "raw/*.csv"}, "time": {"column": "t", "unit": "s"},
            "session_date": {"from": "filename", "pattern": r"(\d{4}-\d{2}-\d{2})", "format": "%Y-%m-%d"},
            "body_velocity": {"columns": dict(zip("uvwpqr", ("Bu", "Bv", "Bw", "Bp", "Bq", "Br")))},
            "orientation": {"quaternion": {k: f"q{k}" for k in "xyzw"}},
            "wrench": {"columns": {c: c for c in COLS}}, **over}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    return str(tmp_path / "spec.yaml")


@pytest.mark.skipif(not REAL, reason="BlueROV2 dataset not downloaded")
@pytest.mark.parametrize("flip", ["Fy", "Tz", None])
def test_a_reversed_force_axis_is_refused(tmp_path, flip):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i, src in enumerate(REAL[:2]):
        d = reference_run_in_frd(pd.read_csv(src)).iloc[:1500].reset_index(drop=True)
        d.insert(0, "t", np.arange(len(d)) * 0.1)
        if flip:
            d[flip] = -d[flip]
        d.to_csv(raw / f"2024-01-1{i}_run.csv", index=False)
    spec = load_ingest_spec(_logged_force_spec(tmp_path))
    if flip is None:
        s = ingest(spec, str(tmp_path / "out"))
        c = s["force_direction"]["correlation_force_vs_velocity"]
        assert all(c[a] > 0.2 for a in ("surge", "sway", "heave", "yaw")) and not s["force_direction"]["reversed_axes"]
        return
    axis = {"Fy": "sway", "Tz": "yaw"}[flip]
    with pytest.raises(IngestError, match=f"force on {axis} pushes against"):
        ingest(spec, str(tmp_path / "out"))
    s = ingest(load_ingest_spec(_logged_force_spec(tmp_path, skip_checks=["force_direction"])), str(tmp_path / "o2"))
    assert any("check skipped" in w for w in s["warnings"])


def _circle(n=1200, dt=0.1):
    t = np.arange(n) * dt
    psi = 0.15 * t                                    # turning steadily
    u = 0.3 + 0.1 * np.sin(0.3 * t)
    vn, ve = u * np.cos(psi), u * np.sin(psi)
    pos = np.column_stack([np.cumsum(vn) * dt, np.cumsum(ve) * dt, np.zeros(n)])
    from scipy.spatial.transform import Rotation
    return t, pos, psi, np.column_stack([u, np.zeros(n), np.zeros(n)]), Rotation


def test_orientation_against_positions_catches_a_mirrored_or_offset_heading():
    t, pos, psi, lin, Rot = _circle()
    R = lambda yaw: Rot.from_euler("xyz", np.column_stack([0 * t, 0 * t, yaw])).as_matrix()
    ok = orientation_vs_position(t, pos + np.random.default_rng(0).normal(0, 0.004, pos.shape), R(psi), lin)
    assert "problem" not in ok and ok["match_r2"] > 0.95 and abs(ok["heading_offset_deg"]) < 3
    mirrored = orientation_vs_position(t, pos, R(-psi + np.radians(175)), lin)  # DFKI's ArduSub yaw (#62)
    assert "mirrored" in mirrored["problem"]
    offset = orientation_vs_position(t, pos, R(psi + np.radians(90)), lin)
    assert "about -90 degrees off" in offset["problem"]


@pytest.mark.skipif(not glob.glob(os.path.join(os.path.dirname(CAL), "data", "raw", "dfki_rov_tether", "strathe_csv",
                                               "*.csv")), reason="DFKI dataset not prepared")
def test_dfki_with_the_old_mirrored_layout_is_refused(tmp_path):
    """The layout as the old preset had it (ROS axes) on DFKI's ArduSub vehicle: sway, heave and yaw reversed."""
    spec = yaml.safe_load(open(os.path.join(CAL, "examples", "dfki_tether", "ingest_dfki_tether.yaml")))
    spec["source"]["files"] = os.path.join(os.path.dirname(CAL), "data", "raw", "dfki_rov_tether", "strathe_csv",
                                           "*.csv")
    for th in spec["wrench"]["thrusters"]["layout"]:
        th["position_m"] = [th["position_m"][0], -th["position_m"][1], -th["position_m"][2]]
        th["direction"] = [th["direction"][0], -th["direction"][1], -th["direction"][2]]
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(spec))
    with pytest.raises(IngestError, match="force on sway, heave, yaw pushes against"):
        ingest(load_ingest_spec(str(tmp_path / "s.yaml")), str(tmp_path / "out"))
