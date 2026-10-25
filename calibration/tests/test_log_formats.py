"""PX4 (.ulg), MOOS-IvP (.alog) and LCM logs read directly (PROJECT_STATUS.md #94; LOG_FORMATS.md). Fast versions of
stress/run_94.py's round trips, on a synthetic recording: a log in each format and a CSV with the same content must
ingest to the same canonical recording. The real-data proof is in STRESS_TEST_93_95.md."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "stress"))

import data_ingestion as di  # noqa: E402
import moos_reader  # noqa: E402
import ulog_reader  # noqa: E402
from format_writers import Topic, write_alog, write_ulog  # noqa: E402

HAS_LCM = __import__("importlib").util.find_spec("lcm") is not None
needs_lcm = pytest.mark.skipif(not HAS_LCM, reason="lcm not installed (optional: pip install lcm)")


class _Rec:
    """A 120 s synthetic recording, shaped like run_94.Rec."""
    def __init__(self):
        n = 1200
        t = np.arange(n) * 0.1
        self.name, self.date, self.n = "odom-05-01-2030-run1.csv", "05-01-2030", n
        self.t = 1893834000.0 + t
        self.B = np.stack([0.3 * np.sin(0.1 * t), 0.1 * np.cos(0.13 * t), 0.05 * np.sin(0.2 * t),
                           0.02 * np.sin(0.3 * t), 0.03 * np.cos(0.25 * t), 0.1 * np.sin(0.05 * t)], 1)
        from scipy.spatial.transform import Rotation
        self.q = Rotation.from_euler("xyz", np.stack([0.05 * np.sin(0.3 * t), 0.04 * np.cos(0.2 * t), 0.1 * t], 1)).as_quat()
        self.F = np.stack([20 * np.sin(0.1 * t), 5 * np.cos(0.13 * t), 8 * np.sin(0.2 * t), np.sin(0.3 * t),
                           np.cos(0.25 * t), 2 * np.sin(0.05 * t)], 1)
        self.pwm = 1500 + 200 * np.sin(np.outer(t, np.linspace(0.05, 0.4, 8)))
        self.xyz = np.cumsum(Rotation.from_quat(self.q).apply(self.B[:, :3]) * 0.1, axis=0)


@pytest.fixture
def R94(monkeypatch, tmp_path):
    import run_94
    monkeypatch.setattr(run_94, "DATA", str(tmp_path))
    return run_94


@pytest.mark.parametrize("fmt", ["px4", "moos", pytest.param("lcm", marks=needs_lcm)])
@pytest.mark.parametrize("variant", [{}, {"actuation": "pwm_preset"}, {"orientation": "euler_deg", "frame": "flu"},
                                     {"velocity": "world", "lin_unit": "knots", "ang_unit": "deg/s"},
                                     {"actuation": "norm_t200", "date": "filename", "position": "cm"}])
def test_a_log_and_a_csv_with_the_same_content_ingest_the_same(R94, fmt, variant):
    # synthetic motion isn't driven by these forces: skip that one data check (the real-data harness keeps it)
    res = R94.round_trip(fmt, [_Rec()], {**R94.BASE, "checks": "skip_force", **variant}, f"t-{fmt}")
    assert R94._ok(res), res


@pytest.mark.parametrize("fmt,kind", [("px4", "multi_instance"), ("moos", "keyvalue"), ("moos", "source"),
                                      pytest.param("lcm", "nested", marks=needs_lcm)])
def test_every_kind_of_column_reference(R94, fmt, kind):
    v = {**R94.BASE, "checks": "skip_force", **({"actuation": "pwm_preset"} if kind == "multi_instance" else {})}
    res = R94.round_trip(fmt, [_Rec()], v, f"k-{fmt}", kind)
    assert R94._ok(res), res


def test_px4_names_and_plain_errors(tmp_path):
    assert ulog_reader.split_reference("actuator_outputs[1].output[2]") == ("actuator_outputs", 1, "output[2]")
    with pytest.raises(ulog_reader.ULogError, match="<topic>.<field>"):
        ulog_reader.split_reference("no-dot")
    t = np.arange(0, 20, 0.1)
    p = str(tmp_path / "a.ulg")
    write_ulog(p, {"vehicle_attitude": Topic(t, {f"q[{i}]": np.full(len(t), [1.0, 0, 0, 0][i]) for i in range(4)}),
                   "actuator_outputs": Topic(t, {"output[0]": 1500 + t}),
                   "actuator_outputs#1": Topic(t, {"output[0]": 1600 + t}, instance=1)})
    with pytest.raises(ulog_reader.ULogError, match="isn't in the log.*vehicle_attitude"):
        ulog_reader.read_log(p, ["nope.x"], 10)
    with pytest.raises(ulog_reader.ULogError, match="has no field 'vx'"):
        ulog_reader.read_log(p, ["vehicle_attitude.vx"], 10)
    with pytest.raises(ulog_reader.ULogError, match=r"2 instances \[0, 1\]; pick one"):
        ulog_reader.read_log(p, ["actuator_outputs.output[0]"], 10)
    df, rep = ulog_reader.read_log(p, ["actuator_outputs[1].output[0]"], 10)
    assert df["actuator_outputs[1].output[0]"].iloc[0] == pytest.approx(1600) and rep["recording_start_unix_s"] is None
    open(tmp_path / "empty.ulg", "w").close()
    with pytest.raises(ulog_reader.ULogError, match="empty"):
        ulog_reader.read_log(str(tmp_path / "empty.ulg"), ["vehicle_attitude.q[0]"], 10)


def test_moos_names_strings_and_publishers(tmp_path):
    t = 1.7e9 + np.arange(0, 20, 0.1)
    p = str(tmp_path / "a.alog")
    write_alog(p, {"nav": Topic(t, {"NAV_X": np.arange(len(t), dtype=float)}, string_fields={
        "NODE_REPORT_LOCAL": [f"NAME=a,SPD={i * 0.01},HDG=90" for i in range(len(t))]}),
        "sim": Topic(t + 0.05, {"NAV_X": np.full(len(t), -9.0)})}, logstart=t[0] - 1, sources={"sim": "uSimMarine"})
    with open(p, "a") as f:
        f.write("25.0   NAV_X   pStratheTest   no_fix\n")
    df, rep = moos_reader.read_log(p, ["NODE_REPORT_LOCAL.SPD", "NAV_X@pStratheTest"], 10)
    assert df["NODE_REPORT_LOCAL.SPD"].iloc[10] == pytest.approx(0.1, abs=1e-9)
    assert df["NAV_X@pStratheTest"].min() >= 0 and any("not a number" in w for w in rep["warnings"])
    _, rep2 = moos_reader.read_log(p, ["NAV_X"], 10)
    assert any("posted by several apps" in w for w in rep2["warnings"])
    with pytest.raises(moos_reader.MoosLogError, match="isn't in the log"):
        moos_reader.read_log(p, ["NAV_NOPE"], 10)
    with pytest.raises(moos_reader.MoosLogError, match="no posting has the key 'NOPE'"):
        moos_reader.read_log(p, ["NODE_REPORT_LOCAL.NOPE"], 10)
    (tmp_path / "x.alog").write_text("0.1 NAV_X a 1\n")
    with pytest.raises(moos_reader.MoosLogError, match="LOGSTART"):
        moos_reader.read_log(str(tmp_path / "x.alog"), ["NAV_X"], 10)


@needs_lcm
def test_lcm_types_and_wrong_types(tmp_path):
    import lcm_reader
    from format_writers import write_lcm
    t = 1.7e9 + np.arange(0, 20, 0.1)
    p = str(tmp_path / "lcmlog-x.00")
    files = write_lcm(p, str(tmp_path / "types"), {"DVL": Topic(t, {"vel[0]": t - t[0], "vel[1]": 0 * t})})
    df, rep = lcm_reader.read_log(p, ["DVL.vel[0]"], 10, type_files=files)
    assert df["DVL.vel[0]"].iloc[5] == pytest.approx(0.5) and "utime" in rep["clock"]
    _, rep = lcm_reader.read_log(p, ["DVL.vel[0]"], 10, type_files=files, stamp="receive")
    assert "receive" in rep["clock"]
    wrong = tmp_path / "wrong.lcm"
    wrong.write_text("package x;\nstruct dvl { double other; }\n")
    with pytest.raises(lcm_reader.LcmLogError, match="no message on DVL matches your type files"):
        lcm_reader.read_log(p, ["DVL.vel[0]"], 10, type_files=[str(wrong)])
    with pytest.raises(lcm_reader.LcmLogError, match="doesn't start with a channel"):
        lcm_reader.read_log(p, ["NAV.x"], 10, type_files=files)


@pytest.mark.parametrize("fmt", ["px4", "moos", "lcm"])
def test_the_spec_says_what_doesnt_belong(fmt):
    base = {"vehicle": "x", "source": {"files": "x", "format": fmt, **({"lcm_types": ["t.lcm"]} if fmt == "lcm" else {})},
            "session_date": {"from": "recording"}, "resample_hz": 10,
            "body_velocity": {"columns": {c: c for c in "uvwpqr"}}, "orientation": {"quaternion": {c: c for c in "xyzw"}},
            "wrench": {"columns": {c: c for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}}}
    di.IngestSpec.model_validate(base)
    for bad, word in (({"time": {"column": "t"}}, "time"), ({"resample_hz": None}, "resample_hz")):
        with pytest.raises(Exception, match=word):
            di.IngestSpec.model_validate({**base, **bad})
    if fmt == "lcm":
        with pytest.raises(Exception, match="lcm_types"):
            di.IngestSpec.model_validate({**base, "source": {"files": "x", "format": "lcm"}})
    else:
        with pytest.raises(Exception, match="stamp only applies"):
            di.IngestSpec.model_validate({**base, "source": {**base["source"], "stamp": "receive"}})
        with pytest.raises(Exception, match="lcm_types only applies"):
            di.IngestSpec.model_validate({**base, "source": {**base["source"], "lcm_types": ["t.lcm"]}})


def test_the_templates_load():
    tdir = os.path.join(os.path.dirname(HERE), "templates")
    for name in ("ingest_px4_ulog.yaml", "ingest_moos_alog.yaml", "ingest_lcm.yaml"):
        spec = di.load_ingest_spec(os.path.join(tdir, name))
        assert spec.source.format in ("px4", "moos", "lcm") and len(spec.columns()) >= 12


@needs_lcm
def test_two_type_sets_with_the_same_names_dont_mix(tmp_path):
    """#94: generated types are imported by package name; a second set with the same names (another vehicle, an edited
    .lcm file) must not decode with the first set's classes."""
    import lcm_reader
    from format_writers import write_lcm
    t = 1.7e9 + np.arange(0, 20, 0.1)
    a, b = str(tmp_path / "lcmlog-a.00"), str(tmp_path / "lcmlog-b.00")
    fa = write_lcm(a, str(tmp_path / "ta"), {"DVL": Topic(t, {"vel[0]": t - t[0], "vel[1]": 0 * t})})
    fb = write_lcm(b, str(tmp_path / "tb"), {"DVL": Topic(t, {"speed": 2 * (t - t[0]), "vel[0]": 3 + 0 * t,
                                                              "vel[1]": 0 * t, "vel[2]": 0 * t})})
    assert lcm_reader.read_log(a, ["DVL.vel[0]"], 10, type_files=fa)[0]["DVL.vel[0]"].iloc[5] == pytest.approx(0.5)
    assert lcm_reader.read_log(b, ["DVL.speed"], 10, type_files=fb)[0]["DVL.speed"].iloc[5] == pytest.approx(1.0)
    assert lcm_reader.read_log(a, ["DVL.vel[0]"], 10, type_files=fa)[0]["DVL.vel[0]"].iloc[5] == pytest.approx(0.5)


def test_pwm_0_is_no_signal(R94, tmp_path):
    """#94, from a real PX4 BlueROV2 log: PX4 writes PWM 0 for an output it doesn't drive and while disarmed. Those are
    stopped thrusters, not a sign the commands are normalized."""
    import data_ingestion as di
    r = _Rec()
    r.pwm[:, 6:] = 0            # two outputs never driven
    r.pwm[100:150, 0] = 0       # one switched off for 5 s
    v = {**R94.BASE, "actuation": "pwm_preset", "checks": "skip_force"}
    res = R94.round_trip("px4", [r], v, "pwm0")
    assert R94._ok(res), res
    warnings = " ".join(res["warnings_fmt"])
    assert "are 0 throughout" in warnings and "actuator_outputs.output[6]" in warnings
    assert "sample(s) have a command of 0" in warnings
    r.pwm[:] = 0
    res = R94.round_trip("px4", [r], v, "pwm0all")
    assert "every thruster command is 0 throughout" in res["error"]
