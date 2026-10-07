"""
Battery voltage with a lab's own thrust table (PROJECT_STATUS.md #69): `curve_voltage:` (the voltage the table was
measured at) lets `voltage_column:` and `voltage:` rescale it with Blue Robotics' published T200 data, as the built-in
curve already is. And the force-direction check on normalized commands: a mirrored layout is refused.
"""
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from thruster_model import T200_DATASET_CURVE, ThrustCurve, _published_max_n, thruster_block, voltage_matching

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
ONE = [{"position_m": [0, 0, 0], "direction": [1, 0, 0]}]


def _table(tmp_path, name="cmd.csv"):
    (tmp_path / name).write_text("command,thrust_n\n-1,-40\n0,0\n1,50\n")
    return name


def test_a_lab_table_is_rescaled_by_the_published_t200_ratio(tmp_path):
    cfg = thruster_block({"layout": ONE, "command": "normalized", "curve_csv": _table(tmp_path),
                          "curve_voltage": 16}, str(tmp_path))
    cmd = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    volts = np.array([16.0, 16.0, 12.0, 12.0])
    got = cfg.wrench_at_voltage(cmd, volts)[:, 0]
    f12, r12 = _published_max_n(12.0)
    f16, r16 = _published_max_n(16.0)
    assert got[0] == pytest.approx(50.0) and got[1] == pytest.approx(-40.0)  # at its own voltage: the table itself
    assert got[2] == pytest.approx(50.0 * f12 / f16) and got[3] == pytest.approx(-40.0 * r12 / r16)
    # a fixed voltage: (for the simulators) does the same once
    at20 = thruster_block({"layout": ONE, "command": "normalized", "curve_csv": "cmd.csv", "curve_voltage": 16,
                           "voltage": 20}, str(tmp_path))
    assert at20.wrench(np.array([[1.0]]))[0, 0] == pytest.approx(50.0 * _published_max_n(20.0)[0] / f16)


def test_the_dataset_curve_as_a_lab_table_scales_like_the_built_in_one(tmp_path):
    """The built-in T200 curve given as if it were a lab's own table, with the voltage it matches: per-sample scaling
    agrees with the built-in path (the dataset's forward and reverse maxima give the same voltage, to ~1%)."""
    builtin = thruster_block({"layout": ONE, "curve": "t200"})
    v0 = voltage_matching(builtin.curve.max_forward_n)
    lab = thruster_block({"layout": ONE, "curve_csv": T200_DATASET_CURVE, "curve_voltage": v0})
    pwm = np.random.default_rng(0).uniform(1100, 1900, size=(300, 1))
    volts = np.random.default_rng(1).uniform(12.5, 16.5, size=300)
    a, b = builtin.wrench_at_voltage(pwm, volts), lab.wrench_at_voltage(pwm, volts)
    assert np.allclose(a, b, rtol=0.02, atol=1e-6)


@pytest.mark.parametrize("block,match", [
    ({"layout": ONE, "curve_voltage": 16}, "built-in T200 curve knows its own"),
    ({"layout": ONE, "curve_csv": "cmd.csv", "command": "normalized", "voltage": 15}, "give curve_voltage"),
    ({"layout": ONE, "curve_csv": "cmd.csv", "command": "normalized", "curve_voltage": 25}, "12.0-20.0 V"),
    ({"preset": "bluerov2_heavy", "curve_voltage": 16}, "remove .*curve_voltage"),
])
def test_curve_voltage_mistakes_are_explained(tmp_path, block, match):
    _table(tmp_path)
    with pytest.raises(ValueError, match=match):
        thruster_block(block, str(tmp_path))


def _write_logs(tmp_path, cmds, volts, vel):
    (tmp_path / "raw").mkdir(exist_ok=True)
    n = len(cmds)
    half = n // 2
    for day, sl in (("2030-01-05", slice(0, half)), ("2030-01-06", slice(half, n))):
        d = pd.DataFrame({"t": np.arange(sl.stop - sl.start) * 0.1})
        for c, col in zip("uvwpqr", vel.T):
            d[c] = col[sl]
        d["qx"], d["qy"], d["qz"], d["qw"] = 0.0, 0.0, 0.0, 1.0
        for i in range(cmds.shape[1]):
            d[f"c{i + 1}"] = cmds[sl, i]
        d["battery_v"] = volts[sl]
        d.to_csv(tmp_path / "raw" / f"dive-{day}.csv", index=False)


def _spec(tmp_path, thrusters, name):
    spec = {"vehicle": "mit_like", "source": {"files": str(tmp_path / "raw" / "*.csv")},
            "time": {"column": "t", "unit": "s"}, "resample_hz": 10,
            "session_date": {"from": "filename", "pattern": r"(\d{4}-\d{2}-\d{2})", "format": "%Y-%m-%d"},
            "body_velocity": {"columns": {k: k for k in "uvwpqr"}},
            "orientation": {"quaternion": {k: f"q{k}" for k in "xyzw"}},
            "wrench": {"thrusters": {"pwm_columns": [f"c{i + 1}" for i in range(len(thrusters["layout"]))],
                                     **thrusters}}}
    p = tmp_path / name
    p.write_text(yaml.safe_dump(spec))
    return str(p)


# a 6-thruster vehicle: four horizontal thrusters at 45 degrees, two vertical (FRD)
LAYOUT = [{"position_m": [0.15, 0.15, 0.0], "direction": [1, -1, 0]},
          {"position_m": [0.15, -0.15, 0.0], "direction": [1, 1, 0]},
          {"position_m": [-0.15, 0.15, 0.0], "direction": [1, 1, 0]},
          {"position_m": [-0.15, -0.15, 0.0], "direction": [1, -1, 0]},
          {"position_m": [0.0, 0.2, 0.0], "direction": [0, 0, 1]},
          {"position_m": [0.0, -0.2, 0.0], "direction": [0, 0, 1]}]
MIRRORED = [{"position_m": [p["position_m"][0], -p["position_m"][1], p["position_m"][2]],
             "direction": [p["direction"][0], -p["direction"][1], p["direction"][2]]} for p in LAYOUT]


def _vehicle_logs(tmp_path, n=1200, seed=0):
    """Commands normalized to -1..1 through the lab's own table at a sagging battery; the vehicle's velocity
    responds to the true (voltage-scaled) forces: first-order on surge, sway, heave and yaw."""
    rng = np.random.default_rng(seed)
    cmds = np.repeat(rng.uniform(-1, 1, size=(n // 20, 6)), 20, axis=0)
    volts = np.linspace(16.4, 13.2, n)
    truth = thruster_block({"layout": LAYOUT, "command": "normalized", "curve_csv": _table(tmp_path),
                            "curve_voltage": 16}, str(tmp_path))
    F = truth.wrench_at_voltage(cmds, volts)
    vel = np.zeros((n, 6))
    mass, drag = np.array([20, 25, 25, 1, 1, 2.0]), np.array([30, 40, 40, 5, 5, 25])  # yaw under ~1 rad/s
    for k in range(n - 1):
        vel[k + 1] = vel[k] + 0.1 * (F[k] - drag * vel[k]) / mass
    vel[:, 3:5] = 0.0  # level vehicle: no roll or pitch motion
    return cmds, volts, vel, truth


def test_the_mit_case_end_to_end(tmp_path):
    """Normalized commands, the lab's own T200 table, and the battery voltage logged with each sample: ingestion
    gives exactly the table's forces rescaled to each sample's voltage, and the direction check passes."""
    from data_ingestion import ingest, load_ingest_spec
    cmds, volts, vel, truth = _vehicle_logs(tmp_path)
    _write_logs(tmp_path, cmds, volts, vel)
    thr = {"command": "normalized", "curve_csv": str(tmp_path / "cmd.csv"), "curve_voltage": 16,
           "voltage_column": "battery_v", "layout": LAYOUT}
    s = ingest(load_ingest_spec(_spec(tmp_path, thr, "spec.yaml")), str(tmp_path / "out"))
    assert not s["force_direction"]["reversed_axes"]
    want = truth.wrench_at_voltage(cmds, volts)
    half = len(cmds) // 2
    for f in s["files"]:  # each day's file against its own stretch of the truth
        start = 0 if "2030-01-05" in f["file"] else half
        got = pd.read_csv(tmp_path / "out" / f["output"])[["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]].to_numpy()
        m = min(len(got), half)
        assert np.allclose(got[5:m - 5], want[start + 5:start + m - 5], atol=1e-6)
    assert s["files"][0]["voltage_v"]["max"] > 16.0 and "measured at 16 V" in s["files"][0]["wrench_source"]


def test_a_mirrored_layout_with_normalized_commands_is_refused(tmp_path):
    """The same vehicle's logs read with its layout mirrored left-right: sway and yaw forces push against the motion,
    and ingestion refuses the data instead of calibrating a mirror image."""
    from data_ingestion import IngestError, ingest, load_ingest_spec
    cmds, volts, vel, _ = _vehicle_logs(tmp_path, seed=1)
    _write_logs(tmp_path, cmds, volts, vel)
    thr = {"command": "normalized", "curve_csv": str(tmp_path / "cmd.csv"), "curve_voltage": 16,
           "voltage_column": "battery_v", "layout": MIRRORED}
    with pytest.raises(IngestError, match="pushes against the way the vehicle moves") as e:
        ingest(load_ingest_spec(_spec(tmp_path, thr, "mirrored.yaml")), str(tmp_path / "out"))
    assert "sway" in str(e.value) and "yaw" in str(e.value)


def test_voltage_column_with_a_lab_table_needs_its_voltage(tmp_path):
    from data_ingestion import load_ingest_spec
    _table(tmp_path)
    thr = {"command": "normalized", "curve_csv": str(tmp_path / "cmd.csv"), "voltage_column": "battery_v",
           "layout": LAYOUT}
    with pytest.raises(Exception, match="needs curve_voltage"):
        load_ingest_spec(_spec(tmp_path, thr, "nocv.yaml"))
