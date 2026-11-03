"""
ArduPilot / ArduSub logs read directly: DataFlash .bin and MAVLink .tlog.
Real dataset runs are written the way ArduSub logs them (NED EKF velocity,
attitude in centidegrees, body gyro rates, servo PWM, two IMUs) with the
shipped templates, and ingestion must recover the canonical data. Then the
hard cases: corrupt bytes, a log cut off mid-record, missing messages,
ambiguous multi-instance messages, yaw wrapping through 360, offset timing.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
pytest.importorskip("pymavlink")
from ardupilot_log_writer import write_bin, write_tlog  # noqa: E402
from data_ingestion import reference_run_in_frd, IngestError, ingest, load_ingest_spec  # noqa: E402

TEMPLATES = os.path.join(HERE, "..", "templates")
DATA = os.path.join(HERE, "..", "..", "data", "raw", "Bluerov2_trajectory_dataset", "dataset")
REAL = sorted(glob.glob(os.path.join(DATA, "*.csv")))
needs_data = pytest.mark.skipif(not REAL, reason="BlueROV2 dataset not downloaded")
T0 = 1_707_000_000.0  # 2024-02-03 22:40 UTC


def _template(tmp_path, name, **over):
    spec = yaml.safe_load(open(os.path.join(TEMPLATES, name), encoding="utf-8"))
    for k, v in over.items():
        spec[k] = {**spec[k], **v} if isinstance(v, dict) and isinstance(spec.get(k), dict) else v
    wr = spec["wrench"]["thrusters"]
    wr.pop("voltage", None)  # the dataset's own curve, so forces compare exactly
    p = tmp_path / name
    p.write_text(yaml.safe_dump(spec))
    return str(p)


def _check(csv_path, d, v_tol, rot_tol_deg, f_tol=5e-3):
    got = pd.read_csv(csv_path)
    n = min(len(got), len(d))
    for c in ("Bu", "Bv", "Bw", "Bp", "Bq", "Br"):
        np.testing.assert_allclose(got[c].values[:n], d[c].values[:n], atol=v_tol, err_msg=c)
    for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"):
        np.testing.assert_allclose(got[c].values[:n], d[c].values[:n], atol=f_tol, err_msg=c)
    Rg = Rotation.from_quat(got[["qx", "qy", "qz", "qw"]].values[:n])
    Ro = Rotation.from_quat(d[["qx", "qy", "qz", "qw"]].values[:n])
    assert np.degrees((Rg.inv() * Ro).magnitude()).max() < rot_tol_deg
    return got


@needs_data
@pytest.mark.parametrize("gps", [True, False])
def test_dataflash_bin_round_trip(tmp_path, gps):
    d = reference_run_in_frd(pd.read_csv(REAL[0])).iloc[:600].reset_index(drop=True)
    (tmp_path / "logs").mkdir()
    write_bin(tmp_path / "logs" / "2024-02-03_dive1.bin", d, t0_unix=T0 if gps else None)
    over = {"session_date": {"from": "recording"}} if gps else {}
    s = ingest(load_ingest_spec(_template(tmp_path, "ingest_ardupilot_bin.yaml", **over)), str(tmp_path / "out"))
    assert s["sessions"] == {"03-02-2024": 1} and s["sampling_rate_hz"] == pytest.approx(10.0)
    rep = s["files"][0]
    assert rep["log"]["clock"] == ("log clock (GPS time)" if gps else "time since boot (no GPS time in the log)")
    assert rep["velocity_rotated_from_world_frame"]
    # attitude is logged in centidegrees: 0.01 deg quantisation limits the world->body velocity rotation
    got = _check(tmp_path / "out" / rep["output"], d, v_tol=5e-4, rot_tol_deg=0.02)
    assert len(got) == 600


@needs_data
def test_mavlink_tlog_round_trip_with_offset_messages(tmp_path):
    d = reference_run_in_frd(pd.read_csv(REAL[1])).iloc[:600].reset_index(drop=True)
    (tmp_path / "telemetry").mkdir()
    write_tlog(tmp_path / "telemetry" / "flight.tlog", d, t0_unix=T0)
    s = ingest(load_ingest_spec(_template(tmp_path, "ingest_mavlink_tlog.yaml")), str(tmp_path / "out"))
    assert s["sessions"] == {"03-02-2024": 1}
    rep = s["files"][0]
    assert rep["log"]["clock"] == "ground-station recording clock"
    _check(tmp_path / "out" / rep["output"], d, v_tol=1e-5, rot_tol_deg=1e-3)  # float32 attitude
    # messages arriving at different moments: interpolated, PWM held
    write_tlog(tmp_path / "telemetry" / "flight.tlog", d, t0_unix=T0,
               offsets={"ATTITUDE": 0.03, "LOCAL_POSITION_NED": 0.06})
    s = ingest(load_ingest_spec(_template(tmp_path, "ingest_mavlink_tlog.yaml")), str(tmp_path / "out2"))
    got = pd.read_csv(tmp_path / "out2" / s["files"][0]["output"])
    assert np.all(np.isfinite(got.values)) and np.median(np.abs(got["Bu"].values[5:500] - d["Bu"].values[5:500])) < 0.02


@needs_data
def test_corrupt_and_truncated_logs_are_read_up_to_the_damage(tmp_path):
    d = reference_run_in_frd(pd.read_csv(REAL[2])).iloc[:400].reset_index(drop=True)
    (tmp_path / "logs").mkdir()
    write_bin(tmp_path / "logs" / "2024-02-03_a.bin", d, garbage_every=37, corrupt_tail=True)
    s = ingest(load_ingest_spec(_template(tmp_path, "ingest_ardupilot_bin.yaml")), str(tmp_path / "out"))
    got = pd.read_csv(tmp_path / "out" / s["files"][0]["output"])
    assert len(got) >= 390 and np.all(np.isfinite(got.values))
    (tmp_path / "telemetry").mkdir()
    write_tlog(tmp_path / "telemetry" / "t.tlog", d, t0_unix=T0, corrupt_every=11)
    s = ingest(load_ingest_spec(_template(tmp_path, "ingest_mavlink_tlog.yaml")), str(tmp_path / "out2"))
    rep = s["files"][0]
    assert any("corrupt" in w for w in rep["warnings"])
    got = pd.read_csv(tmp_path / "out2" / rep["output"])
    assert len(got) >= 390 and np.all(np.isfinite(got.values))
    # a corrupt record swallows the message after it, whose value is then interpolated from its
    # neighbours: most rows stay exact, none are wild
    err = np.abs(got["Bu"].values[:390] - d["Bu"].values[:390])
    assert np.mean(err < 1e-5) > 0.75 and err.max() < 0.1


@needs_data
def test_yaw_wrapping_through_360_is_not_interpolated_the_long_way(tmp_path):
    """ATT.Yaw is logged 0-360: a heading crossing north jumps 359.9 -> 0.1. Resampling at 20 Hz puts grid
    points between those samples; without unwrapping they'd be interpolated to ~180 deg (backwards)."""
    d = reference_run_in_frd(pd.read_csv(REAL[0])).iloc[:300].reset_index(drop=True)
    yaw = np.linspace(-20, 20, len(d))  # heading sweeping through north
    R = Rotation.from_euler("z", yaw[:, None], degrees=True)
    d[["qx", "qy", "qz", "qw"]] = R.as_quat()
    (tmp_path / "logs").mkdir()
    write_bin(tmp_path / "logs" / "2024-02-03_y.bin", d)
    s = ingest(load_ingest_spec(_template(tmp_path, "ingest_ardupilot_bin.yaml", resample_hz=20)), str(tmp_path / "o"))
    got = pd.read_csv(tmp_path / "o" / s["files"][0]["output"])
    head = np.degrees(np.unwrap(Rotation.from_quat(got[["qx", "qy", "qz", "qw"]].values).as_euler("xyz")[:, 2]))
    assert np.abs(np.diff(head)).max() < 1.0


@needs_data
@pytest.mark.parametrize("change,match", [
    ({"columns": {"u": "XKF1[0].VN", "v": "XKF1[0].VE", "w": "XKF1[0].VD",
                  "p": "IMU.GyrX", "q": "IMU.GyrY", "r": "IMU.GyrZ"}}, r"IMU is logged for 2 instances"),
    ({"columns": {"u": "XKF1[0].VN", "v": "XKF1[0].VE", "w": "XKF1[0].VD",
                  "p": "IMU[4].GyrX", "q": "IMU[4].GyrY", "r": "IMU[4].GyrZ"}}, r"IMU has no instance 4"),
    ({"columns": {"u": "XKF1[0].Speed", "v": "XKF1[0].VE", "w": "XKF1[0].VD",
                  "p": "IMU[0].GyrX", "q": "IMU[0].GyrY", "r": "IMU[0].GyrZ"}}, r"no field 'Speed'"),
    ({"columns": {"u": "DVL.VelX", "v": "XKF1[0].VE", "w": "XKF1[0].VD",
                  "p": "IMU[0].GyrX", "q": "IMU[0].GyrY", "r": "IMU[0].GyrZ"}}, r"\['DVL'\] not in the log"),
    ({"columns": {"u": "velocity", "v": "XKF1[0].VE", "w": "XKF1[0].VD",
                  "p": "IMU[0].GyrX", "q": "IMU[0].GyrY", "r": "IMU[0].GyrZ"}}, r"isn't an ArduPilot log column"),
])
def test_log_errors_say_what_to_fix(tmp_path, change, match):
    d = reference_run_in_frd(pd.read_csv(REAL[0])).iloc[:100]
    (tmp_path / "logs").mkdir()
    write_bin(tmp_path / "logs" / "2024-02-03_e.bin", d)
    with pytest.raises(IngestError, match=match):
        ingest(load_ingest_spec(_template(tmp_path, "ingest_ardupilot_bin.yaml", body_velocity=change)),
               str(tmp_path / "out"))


@needs_data
def test_missing_pwm_and_boot_clock_dates_are_explained(tmp_path):
    d = reference_run_in_frd(pd.read_csv(REAL[0])).iloc[:100]
    (tmp_path / "logs").mkdir()
    write_bin(tmp_path / "logs" / "2024-02-03_e.bin", d, drop_type="RCOU")
    with pytest.raises(IngestError, match=r"\['RCOU'\] not in the log"):
        ingest(load_ingest_spec(_template(tmp_path, "ingest_ardupilot_bin.yaml")), str(tmp_path / "o1"))
    write_bin(tmp_path / "logs" / "2024-02-03_e.bin", d)  # no GPS: boot clock
    with pytest.raises(IngestError, match="ArduPilot .bin with GPS time"):
        ingest(load_ingest_spec(_template(tmp_path, "ingest_ardupilot_bin.yaml",
                                          session_date={"from": "recording"})), str(tmp_path / "o2"))
    (tmp_path / "logs" / "2024-02-03_empty.bin").write_bytes(b"")
    with pytest.raises(IngestError, match="empty or missing"):
        ingest(load_ingest_spec(_template(tmp_path, "ingest_ardupilot_bin.yaml")), str(tmp_path / "o3"))


def test_world_frame_velocity_rotation_on_a_csv(tmp_path):
    """linear_frame: world works for any source: a vehicle yawed 90 deg moving north has surge 0, sway -1."""
    raw = tmp_path / "raw"
    raw.mkdir()
    n = 50
    q = Rotation.from_euler("z", 90, degrees=True).as_quat()
    pd.DataFrame({"t": np.arange(n) * 0.1, "vn": 1.0, "ve": 0.0, "vd": 0.0, "p": 0.0, "q": 0.0, "r": 0.0,
                  "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3], **{k: 1.0 for k in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}}
                 ).to_csv(raw / "2024-01-12_a.csv", index=False)
    spec = {"vehicle": "w", "source": {"files": "raw/*.csv"}, "time": {"column": "t"},
            "session_date": {"from": "filename", "pattern": r"(\d{4}-\d{2}-\d{2})", "format": "%Y-%m-%d"},
            "body_velocity": {"columns": {"u": "vn", "v": "ve", "w": "vd", "p": "p", "q": "q", "r": "r"},
                              "linear_frame": "world"},
            "orientation": {"quaternion": {k: f"q{k}" for k in "xyzw"}},
            "wrench": {"columns": {k: k for k in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}}}
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(spec))
    ingest(load_ingest_spec(str(tmp_path / "s.yaml")), str(tmp_path / "o"))
    got = pd.read_csv(tmp_path / "o" / "w-12-01-2024-run1.csv")
    np.testing.assert_allclose(got[["Bu", "Bv", "Bw"]].values, np.tile([0.0, -1.0, 0.0], (n, 1)), atol=1e-12)


def test_ardupilot_spec_rules(tmp_path):
    base = yaml.safe_load(open(os.path.join(TEMPLATES, "ingest_ardupilot_bin.yaml"), encoding="utf-8"))
    for over, match in (({"time": {"column": "TimeUS"}}, "remove the time: block"), ({"resample_hz": None}, "needs resample_hz")):
        spec = {**base, **over}
        (tmp_path / "s.yaml").write_text(yaml.safe_dump(spec))
        with pytest.raises(IngestError, match=match):
            load_ingest_spec(str(tmp_path / "s.yaml"))


@needs_data
def test_the_no_dvl_template_reads_an_ardusub_log_with_only_its_own_sensors(tmp_path):
    """templates/ingest_ardusub_no_dvl.yaml (#96) on a .bin with no velocity: the gyro, the attitude, the external
    pressure sensor (BARO[1], not the enclosure's BARO[0]) as depth, and the raw IMU's accelerometer."""
    d = reference_run_in_frd(pd.read_csv(REAL[0])).iloc[:600].reset_index(drop=True)
    R = Rotation.from_quat(d[["qx", "qy", "qz", "qw"]].values).as_matrix()
    v, w = d[["Bu", "Bv", "Bw"]].values, d[["Bp", "Bq", "Br"]].values
    down = np.einsum("nij,nj->ni", R, v)[:, 2]
    d["depth"] = 1.5 + np.concatenate([[0.0], np.cumsum(down[:-1] * 0.1)])
    acc = np.gradient(v, 0.1, axis=0) + np.cross(w, v) + np.einsum("nji,j->ni", R, np.array([0.0, 0.0, -9.80665]))
    d[["acc_x", "acc_y", "acc_z"]] = acc
    (tmp_path / "logs").mkdir()
    write_bin(tmp_path / "logs" / "2024-02-03_dive1.bin", d, sensors=True)
    s = ingest(load_ingest_spec(_template(tmp_path, "ingest_ardusub_no_dvl.yaml")), str(tmp_path / "out"))
    rep = s["files"][0]
    got = pd.read_csv(tmp_path / "out" / rep["output"])
    n = min(len(got), len(d)) - 5
    assert "Bu" not in got.columns
    np.testing.assert_allclose(got[["Bp", "Bq", "Br"]].values[:n], w[:n], atol=1e-5)
    # depth from the pressure, relative to the surface (auto: the lowest pressure, as nothing here is at the surface)
    z = got["depth_bar30"].values[:n]
    ok = np.isfinite(z)
    np.testing.assert_allclose(z[ok] - z[ok].min(), d["depth"].values[:n][ok] - d["depth"].values[:n][ok].min(),
                               atol=2e-3)
    np.testing.assert_allclose(got[["acc_x", "acc_y", "acc_z"]].values[:n], acc[:n], atol=1e-4)
    assert "accelerometer" in rep["sensor_checks"] and "problem" not in rep["sensor_checks"]["accelerometer"]
