"""
Normalized thruster commands (PROJECT_STATUS.md #67): `command: normalized` with `command_range`, and a curve in the
same units or the built-in T200 curve with the range mapped onto 1100..1900 us. Logs that record e.g. -1..1 instead
of ESC microseconds give exactly the forces their PWM equivalents give.
"""
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from thruster_model import thruster_block

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
PWM = np.random.default_rng(0).uniform(1100, 1900, size=(200, 8))


def test_normalized_minus_one_to_one_gives_the_same_forces_as_pwm():
    pwm = thruster_block({"preset": "bluerov2_heavy"})
    norm = thruster_block({"preset": "bluerov2_heavy", "command": "normalized"})
    assert norm.command_range == (-1.0, 1.0) and norm.command_neutral == 0.0
    assert np.allclose(norm.wrench((PWM - 1500) / 400), pwm.wrench(PWM), atol=1e-9)


def test_a_zero_to_one_range_stops_in_the_middle():
    pwm = thruster_block({"preset": "bluerov2_heavy", "voltage": 16})
    norm = thruster_block({"preset": "bluerov2_heavy", "voltage": 16, "command": "normalized",
                           "command_range": [0, 1]})
    assert norm.command_neutral == 0.5
    assert np.allclose(norm.wrench((PWM - 1100) / 800), pwm.wrench(PWM), atol=1e-9)
    assert np.allclose(norm.wrench(np.full((1, 8), 0.5)), 0.0, atol=1e-9)  # all stopped


def test_battery_voltage_scaling_still_works_with_normalized_commands():
    pwm = thruster_block({"preset": "bluerov2_heavy"})
    norm = thruster_block({"preset": "bluerov2_heavy", "command": "normalized"})
    volts = np.linspace(13, 16, len(PWM))
    assert np.allclose(norm.wrench_at_voltage((PWM - 1500) / 400, volts), pwm.wrench_at_voltage(PWM, volts),
                       atol=1e-9)


def test_a_curve_in_command_units_and_a_reversed_thruster(tmp_path):
    (tmp_path / "curve.csv").write_text("command,thrust_n\n-1,-20\n0,0\n1,30\n")
    layout = [{"position_m": [0, 0, 0], "direction": [1, 0, 0]},
              {"position_m": [0, 0, 0], "direction": [1, 0, 0], "reversed": True}]
    cfg = thruster_block({"layout": layout, "command": "normalized", "curve_csv": "curve.csv"}, str(tmp_path))
    assert cfg.thrusts(np.array([[0.5, 0.5]])).tolist() == [[15.0, -10.0]]  # reversed: mirrored about 0
    assert cfg.wrench(np.array([[1.0, -1.0]]))[0, 0] == pytest.approx(60.0)


@pytest.mark.parametrize("block,match", [
    ({"preset": "bluerov2_heavy", "command_range": [-1, 1]}, "command_range is for command: normalized"),
    ({"preset": "bluerov2_heavy", "command": "percent"}, "command must be one of"),
    ({"preset": "bluerov2_heavy", "command": "normalized", "command_range": [1, -1]}, "low < high"),
])
def test_bad_command_settings_are_explained(block, match):
    with pytest.raises(ValueError, match=match):
        thruster_block(block)


def test_curves_in_the_wrong_units_are_explained(tmp_path):
    (tmp_path / "cmd.csv").write_text("command,thrust_n\n-1,-20\n1,30\n")
    (tmp_path / "pwm.csv").write_text("pwm_us,thrust_n\n1100,-20\n1900,30\n")
    layout = [{"position_m": [0, 0, 0], "direction": [1, 0, 0]}]
    with pytest.raises(ValueError, match="add command: normalized"):
        thruster_block({"layout": layout, "curve_csv": "cmd.csv"}, str(tmp_path))
    with pytest.raises(ValueError, match="in your command's units"):
        thruster_block({"layout": layout, "curve_csv": "pwm.csv", "command": "normalized"}, str(tmp_path))


def _spec(tmp_path, thrusters):
    spec = {"vehicle": "n", "source": {"files": str(tmp_path / "logs" / "*.csv")}, "time": {"column": "t", "unit": "s"},
            "session_date": {"from": "constant", "value": "2030-01-05"},
            "body_velocity": {"columns": dict(zip("uvwpqr", ("u", "v", "w", "p", "q", "r")))},
            "orientation": {"euler": {"roll": "roll", "pitch": "pitch", "yaw": "yaw", "unit": "deg"}},
            "skip_checks": ["force_direction"],
            "wrench": {"thrusters": {"pwm_columns": [f"c{i}" for i in range(1, 9)], **thrusters}}}
    p = tmp_path / f"spec{len(list(tmp_path.glob('spec*.yaml')))}.yaml"
    p.write_text(yaml.safe_dump(spec))
    return str(p)


def test_ingestion_reads_normalized_commands_and_refuses_out_of_range(tmp_path):
    from data_ingestion import IngestError, ingest, load_ingest_spec
    (tmp_path / "logs").mkdir()
    n = 100
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"t": np.arange(n) * 0.1, **{c: rng.normal(0, 0.05, n) for c in "uvwpqr"},
                      "roll": 0.0, "pitch": 0.0, "yaw": 10.0})
    pwm = rng.uniform(1200, 1800, size=(n, 8))
    for i in range(8):
        d[f"c{i + 1}"] = pwm[:, i]
    d.to_csv(tmp_path / "logs" / "run.csv", index=False)
    a = ingest(load_ingest_spec(_spec(tmp_path, {"preset": "bluerov2_heavy"})), str(tmp_path / "out_pwm"))
    for i in range(8):
        d[f"c{i + 1}"] = (pwm[:, i] - 1500) / 400
    d.to_csv(tmp_path / "logs" / "run.csv", index=False)
    b = ingest(load_ingest_spec(_spec(tmp_path, {"preset": "bluerov2_heavy", "command": "normalized"})),
               str(tmp_path / "out_norm"))
    fa = pd.read_csv(tmp_path / "out_pwm" / a["files"][0]["output"])
    fb = pd.read_csv(tmp_path / "out_norm" / b["files"][0]["output"])
    for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"):
        assert np.allclose(fa[c], fb[c], atol=1e-9), c
    assert "normalized commands" in b["files"][0]["wrench_source"]
    # normalized data read as PWM: the message says what to add
    with pytest.raises(IngestError, match="add command: normalized"):
        ingest(load_ingest_spec(_spec(tmp_path, {"preset": "bluerov2_heavy"})), str(tmp_path / "out_bad"))
    # commands outside the declared range
    with pytest.raises(IngestError, match="outside command_range"):
        ingest(load_ingest_spec(_spec(tmp_path, {"preset": "bluerov2_heavy", "command": "normalized",
                                                 "command_range": [0, 1]})), str(tmp_path / "out_range"))


def test_the_template_builds_its_thrusters():
    from data_ingestion import load_ingest_spec
    s = load_ingest_spec(os.path.join(CAL, "templates", "ingest_rosbag_normalized_thrusters.yaml"))
    cfg = s._thruster_cfg
    assert cfg.command_range == (-1.0, 1.0) and cfg.curve.unit == "command"
    w = cfg.wrench(np.array([[0.5, 0.5, 0.0, 0.0]]))[0]
    assert w[0] == pytest.approx(24.2) and abs(w[1]) < 1e-12
