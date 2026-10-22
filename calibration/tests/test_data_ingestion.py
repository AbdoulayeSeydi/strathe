"""
data_ingestion.py tests. The main test takes REAL dataset runs, disguises
them as a different vehicle's logs (renamed columns, ROS FLU/ENU frame,
deg/s and cm/s, Euler angles in degrees, millisecond timestamps, ISO dates in
filenames, PWM only -- no forces) and checks ingestion recovers the original
canonical data, and that the calibration objective gives the same number.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.spatial.transform import Rotation

from data_ingestion import ENU_TO_NED, FLU_TO_FRD, IngestError, ingest, load_ingest_spec, reference_run_in_frd
from data_split import split_by_date
from objective import DatasetConfig, load_run, rmse_by_dof
from optimizer import initial_parameters, load_vehicle_spec

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "Bluerov2_trajectory_dataset", "dataset")
REAL = sorted(glob.glob(os.path.join(DATA, "*.csv")))
needs_data = pytest.mark.skipif(not REAL, reason="BlueROV2 dataset not downloaded")
PWM = [f"pwm{i}" for i in range(1, 9)]


def _disguise(src_csv, out_csv, n=600, jitter_ms=0.0, seed=0, drop_forces=True):
    """Reference dataset run, converted to FRD/NED (PROJECT_STATUS.md #62) -> 'foreign' ROS-style log."""
    d = reference_run_in_frd(pd.read_csv(src_csv)).iloc[:n]
    R_ned_frd = Rotation.from_quat(d[["qx", "qy", "qz", "qw"]].values).as_matrix()
    R_enu_flu = ENU_TO_NED @ R_ned_frd @ FLU_TO_FRD
    eul = np.degrees(Rotation.from_matrix(R_enu_flu).as_euler("xyz"))
    flip = np.array([1, -1, -1])
    rng = np.random.default_rng(seed)
    t_ms = np.arange(n) * 100.0 + (rng.uniform(-jitter_ms, jitter_ms, n) if jitter_ms else 0.0)
    foreign = pd.DataFrame({
        "stamp_ms": t_ms,
        "vel_x": d["Bu"] * 100, "vel_y": -d["Bv"] * 100, "vel_z": -d["Bw"] * 100,  # cm/s, FLU
        "rate_x": np.degrees(d["Bp"]), "rate_y": -np.degrees(d["Bq"]), "rate_z": -np.degrees(d["Br"]),  # deg/s, FLU
        "roll_deg": eul[:, 0], "pitch_deg": eul[:, 1], "yaw_deg": eul[:, 2],
    })
    for i, c in enumerate(PWM):
        foreign[f"esc{i + 1}"] = d[c].values
    if not drop_forces:
        for k, s in zip(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"), (1, -1, -1, 1, -1, -1)):
            foreign[f"force_{k}"] = s * d[k].values
    foreign.to_csv(out_csv, index=False)
    return d


def _spec(tmp_path, **over):
    spec = {
        "vehicle": "labrov",
        "source": {"files": "raw/*.csv"},
        "time": {"column": "stamp_ms", "unit": "ms"},
        "session_date": {"from": "filename", "pattern": r"(\d{4}-\d{2}-\d{2})", "format": "%Y-%m-%d"},
        "frame": "flu",
        "body_velocity": {"columns": {"u": "vel_x", "v": "vel_y", "w": "vel_z",
                                      "p": "rate_x", "q": "rate_y", "r": "rate_z"},
                          "linear_unit": "cm/s", "angular_unit": "deg/s"},
        "orientation": {"euler": {"roll": "roll_deg", "pitch": "pitch_deg", "yaw": "yaw_deg", "unit": "deg"}},
        "wrench": {"thrusters": {"pwm_columns": [f"esc{i}" for i in range(1, 9)], "preset": "bluerov2_heavy"}},
    }
    for k, v in over.items():
        spec[k] = v
    p = tmp_path / "ingest.yaml"
    p.write_text(yaml.safe_dump(spec))
    return str(p)


@pytest.fixture
def foreign_logs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    originals = {}
    for src, name in ((REAL[0], "2024-01-12_log_a.csv"), (REAL[-1], "2024-02-19_log_b.csv")):
        originals[name] = _disguise(src, raw / name)
    return tmp_path, originals


def _matches(ingested_csv, original):
    got = pd.read_csv(ingested_csv)
    n = min(len(got), len(original))
    for c in ("Bu", "Bv", "Bw", "Bp", "Bq", "Br"):
        np.testing.assert_allclose(got[c].values[:n], original[c].values[:n], atol=1e-9, err_msg=c)
    for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"):
        np.testing.assert_allclose(got[c].values[:n], original[c].values[:n], atol=5e-3, err_msg=c)
    Rg = Rotation.from_quat(got[["qx", "qy", "qz", "qw"]].values[:n]).as_matrix()
    Ro = Rotation.from_quat(original[["qx", "qy", "qz", "qw"]].values[:n]).as_matrix()
    np.testing.assert_allclose(Rg, Ro, atol=1e-9)


@needs_data
def test_foreign_ros_style_pwm_only_logs_round_trip(foreign_logs):
    tmp, originals = foreign_logs
    out = tmp / "out"
    summary = ingest(load_ingest_spec(_spec(tmp)), str(out))
    assert summary["sampling_rate_hz"] == pytest.approx(10.0)
    assert summary["sessions"] == {"12-01-2024": 1, "19-02-2024": 1}
    names = {f["output"]: f for f in summary["files"]}
    assert set(names) == {"labrov-12-01-2024-run1.csv", "labrov-19-02-2024-run1.csv"}
    _matches(out / "labrov-12-01-2024-run1.csv", originals["2024-01-12_log_a.csv"])
    _matches(out / "labrov-19-02-2024-run1.csv", originals["2024-02-19_log_b.csv"])
    assert "computed from PWM" in names["labrov-12-01-2024-run1.csv"]["wrench_source"]
    assert json.loads((out / "ingest_report.json").read_text(encoding="utf-8"))["n_files"] == 2


@needs_data
def test_ingested_runs_plug_into_the_calibration_pipeline(foreign_logs):
    """Same objective value from ingested files as from the originals, and
    the held-out-by-date split recognizes the ingested session dates."""
    tmp, originals = foreign_logs
    out = tmp / "out"
    ingest(load_ingest_spec(_spec(tmp)), str(out))
    files = sorted(glob.glob(str(out / "*.csv")))
    split = split_by_date(files, heldout_dates=["19-02-2024"])
    assert len(split.train_files) == 1 and len(split.heldout_files) == 1
    spec = load_vehicle_spec("vehicle_spec.yaml")
    params = initial_parameters(spec)
    orig_path = tmp / "orig.csv"
    originals["2024-01-12_log_a.csv"].to_csv(orig_path, index=False)
    a = rmse_by_dof(params, load_run(split.train_files[0], DatasetConfig(10.0)), spec.vehicle)
    b = rmse_by_dof(params, load_run(str(orig_path), DatasetConfig(10.0)), spec.vehicle)
    np.testing.assert_allclose(a, b, rtol=1e-4)


@needs_data
def test_logged_forces_path_and_resampling_irregular_timestamps(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    orig = _disguise(REAL[0], raw / "2024-01-12_log.csv", jitter_ms=20.0, drop_forces=False)
    spec = _spec(tmp_path, resample_hz=10.0,
                 wrench={"columns": {k: f"force_{k}" for k in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}})
    summary = ingest(load_ingest_spec(spec), str(tmp_path / "out"))
    rep = summary["files"][0]
    assert rep["timestamp_jitter_pct"] > 5 and rep["resampled_to_hz"] == 10.0
    got = pd.read_csv(tmp_path / "out" / "labrov-12-01-2024-run1.csv")
    # interpolation onto a regular grid from +-20 ms jittered samples: close, not exact
    assert np.abs(got["Bu"].values[:500] - orig["Bu"].values[:500]).mean() < 0.01
    assert rep["wrench_source"] == "logged forces"


# ------------------------------------------------------------ the silent errors it must catch

def _synthetic(tmp_path, name="2024-01-12_x.csv", n=100, **cols):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    base = {"stamp_ms": np.arange(n) * 100.0, "vel_x": np.full(n, 20.0), "vel_y": np.zeros(n), "vel_z": np.zeros(n),
            "rate_x": np.full(n, 5.0), "rate_y": np.zeros(n), "rate_z": np.zeros(n),
            "roll_deg": np.zeros(n), "pitch_deg": np.zeros(n), "yaw_deg": np.zeros(n)}
    for i in range(1, 9):
        base[f"esc{i}"] = np.full(n, 1550.0)
    base.update(cols)
    pd.DataFrame(base).to_csv(raw / name, index=False)


@pytest.mark.parametrize("cols,spec_over,match", [
    ({"rate_x": np.full(100, 45.0)}, {"body_velocity": {"columns": {"u": "vel_x", "v": "vel_y", "w": "vel_z",
      "p": "rate_x", "q": "rate_y", "r": "rate_z"}, "linear_unit": "cm/s", "angular_unit": "rad/s"}},
     "logged in deg/s"),
    ({"vel_x": np.full(100, 700.0)}, {}, "check linear_unit"),
    ({"stamp_ms": np.r_[np.arange(50), np.arange(50)] * 100.0}, {}, "not strictly increasing"),
    ({"stamp_ms": np.arange(100) * 100.0 + np.random.default_rng(0).uniform(-30, 30, 100)}, {}, "set resample_hz"),
    ({"esc1": np.full(100, 0.55)}, {}, "ESC microseconds"),
    ({}, {"session_date": {"from": "filename", "pattern": r"(\d{8})", "format": "%Y%m%d"}}, "doesn't match the filename"),
    ({}, {"body_velocity": {"columns": {"u": "nope", "v": "vel_y", "w": "vel_z", "p": "rate_x", "q": "rate_y",
                                        "r": "rate_z"}, "linear_unit": "cm/s", "angular_unit": "deg/s"}},
     "column 'nope' not found"),
])
def test_silent_data_errors_are_refused(tmp_path, cols, spec_over, match):
    _synthetic(tmp_path, **cols)
    with pytest.raises(IngestError, match=match):
        ingest(load_ingest_spec(_spec(tmp_path, **spec_over)), str(tmp_path / "out"))


def test_non_unit_quaternion_is_refused(tmp_path):
    _synthetic(tmp_path, qx=np.zeros(100), qy=np.zeros(100), qz=np.zeros(100), qw=np.full(100, 0.5))
    spec = _spec(tmp_path, orientation={"quaternion": {"x": "qx", "y": "qy", "z": "qz", "w": "qw"}})
    with pytest.raises(IngestError, match="quaternion norms"):
        ingest(load_ingest_spec(spec), str(tmp_path / "out"))


def test_files_at_different_rates_are_refused(tmp_path):
    _synthetic(tmp_path, name="2024-01-12_a.csv")
    _synthetic(tmp_path, name="2024-01-13_b.csv", stamp_ms=np.arange(100) * 50.0)
    with pytest.raises(IngestError, match="different sampling rates"):
        ingest(load_ingest_spec(_spec(tmp_path)), str(tmp_path / "out"))


def test_multiple_runs_per_session_get_numbered(tmp_path):
    _synthetic(tmp_path, name="2024-01-12_a.csv")
    _synthetic(tmp_path, name="2024-01-12_b.csv")
    s = ingest(load_ingest_spec(_spec(tmp_path)), str(tmp_path / "out"))
    assert [f["output"] for f in s["files"]] == ["labrov-12-01-2024-run1.csv", "labrov-12-01-2024-run2.csv"]


@pytest.mark.parametrize("over,match", [
    ({"time": {"column": "t", "sampling_rate_hz": 10}}, "exactly one of"),
    ({"orientation": {}}, "exactly one of"),
    ({"wrench": {"thrusters": {"pwm_columns": ["a"], "directions": [1]}}}, "exactly one of: preset, layout"),
    ({"vehicle": "has spaces"}, "pattern"),
])
def test_invalid_specs_are_rejected(tmp_path, over, match):
    with pytest.raises(IngestError, match=match):
        load_ingest_spec(_spec(tmp_path, **over))


@needs_data
def test_a_rotated_sensor_mount_is_undone(foreign_logs):
    """#91 (URI's DVL is mounted rolled 180 deg, so its raw velocity failed the heave check, F4): the log's velocity
    and rates come from sensors mounted rotated; linear_/angular_mount_rpy_deg turn them back into the vehicle's axes."""
    tmp, originals = foreign_logs
    for f in (tmp / "raw").glob("*.csv"):
        df = pd.read_csv(f)
        df["vel_y"], df["vel_z"] = -df["vel_y"], -df["vel_z"]        # a DVL rolled 180 deg
        df["rate_x"], df["rate_y"] = df["rate_y"], -df["rate_x"]      # a gyro yawed 90 deg
        df.to_csv(f, index=False)
    bv = {"columns": {"u": "vel_x", "v": "vel_y", "w": "vel_z", "p": "rate_x", "q": "rate_y", "r": "rate_z"},
          "linear_unit": "cm/s", "angular_unit": "deg/s", "linear_mount_rpy_deg": [180, 0, 0],
          "angular_mount_rpy_deg": [0, 0, 90]}
    summary = ingest(load_ingest_spec(_spec(tmp, body_velocity=bv)), str(tmp / "out"))
    assert summary["files"][0]["sensor_mounts"] == {"linear_mount_rpy_deg": [180.0, 0.0, 0.0],
                                                    "angular_mount_rpy_deg": [0.0, 0.0, 90.0]}
    for name, original in originals.items():
        got = pd.read_csv(next((tmp / "out").glob(f"labrov-{name[8:10]}-{name[5:7]}-{name[:4]}-*.csv")))
        n = min(len(got), len(original))
        for c in ("Bu", "Bv", "Bw", "Bp", "Bq", "Br"):
            np.testing.assert_allclose(got[c].values[:n], original[c].values[:n], atol=1e-6, err_msg=c)
    with pytest.raises(Exception, match="three angles"):
        load_ingest_spec(_spec(tmp, body_velocity={**bv, "linear_mount_rpy_deg": [180, 0]}))
