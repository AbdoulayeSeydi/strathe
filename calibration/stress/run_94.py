"""
stress/run_94.py -- stress test of the PX4, MOOS-IvP and LCM readers (PROJECT_STATUS.md #94) on the real datasets.

    python stress/run_94.py fast         # round trips, every variant, pairs, field kinds, damage, size (tens of minutes)
    python stress/run_94.py mutants      # each deliberate fault must be caught
    python stress/run_94.py e2e [ROW]    # calibrations on each format against the CSV route (hours)
    python stress/run_94.py list

The proof is a round trip on real recordings: each one (oceansystemslab's BlueROV2 Heavy: its logged forces and its
real ESC PWM; URI's ALPHA RISE) is written as a lab's log in each format (tests/format_writers.py: topics at their
own rates, commands held between updates, attitude at 50 Hz, velocity at 10 Hz) and, with the same content, as a
plain CSV. Ingesting both must give the same canonical recordings, and calibrating both the same model.
"""
from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, CAL)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(CAL, "tests"))
from common import PY, record, strathe  # noqa: E402

import data_ingestion as di  # noqa: E402
import thruster_model as tm  # noqa: E402
from format_writers import Topic, write_alog, write_lcm, write_ulog  # noqa: E402

F = "94"
FORMATS = ("px4", "moos", "lcm")
DATA = os.path.expanduser("~/strathe_formats_data")
BLUEROV = os.path.expanduser("~/marine_sim_mvp/data/raw/Bluerov2_trajectory_dataset/dataset")
URI = os.path.expanduser("~/strathe_uri_lab_v2/runs/ingest_I0")
URI_SPEC = os.path.expanduser("~/strathe_uri_lab_v2/data/alpha_rise_spec.yaml")
KNOTS = 0.514444
T_FLU = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)     # ENU <-> NED (its own inverse)
TB = np.diag([1.0, -1.0, -1.0])                                  # FLU <-> FRD
BASE = {"actuation": "forces", "velocity": "body", "orientation": "quat", "frame": "frd", "lin_unit": "m/s",
        "ang_unit": "rad/s", "resample": 10, "date": "recording", "position": "none", "exclude": "none",
        "checks": "none", "stamp": "default"}
OPTIONS = {"actuation": ("forces", "pwm_preset", "pwm_layout", "pwm_allocation", "norm_t200", "norm_custom",
                         "pwm_voltage", "pwm_t200_layout", "pwm_nominal_voltage", "pwm_curve_voltage", "norm_range01"),
           "velocity": ("body", "world", "mount"), "orientation": ("quat", "euler_deg", "euler_rad"),
           "frame": ("frd", "flu"), "lin_unit": ("m/s", "cm/s", "mm/s", "knots"), "ang_unit": ("rad/s", "deg/s"),
           "resample": (5, 10, 20), "date": ("recording", "filename", "constant"), "position": ("none", "m", "cm", "mm"),
           "exclude": ("none", "rule", "rule_below", "rule_minseg30"), "checks": ("none", "skip_force", "skip_orient"),
           "stamp": ("default", "header", "receive")}


# --- the recordings -------------------------------------------------------------------------------------------------
class Rec:
    """One real recording at its own 10 Hz: what the vehicle logged, in Strathe's frames."""

    def __init__(self, path: str, k: int):
        df = pd.read_csv(path)
        import data_split as ds
        self.name, self.date = os.path.basename(path), ds.extract_date(path)
        self.n = len(df)
        d = pd.to_datetime(self.date, format="%d-%m-%Y")
        self.t = (d.value / 1e9 + 9 * 3600 + 900 * k) + np.arange(self.n) * 0.1   # Unix s, a 15-min slot per run
        self.F = df[["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]].to_numpy(float)
        self.B = df[["Bu", "Bv", "Bw", "Bp", "Bq", "Br"]].to_numpy(float)
        self.q = df[["qx", "qy", "qz", "qw"]].to_numpy(float)
        self.pwm = df[[f"pwm{i}" for i in range(1, 9)]].to_numpy(float) if "pwm1" in df else None
        self.xyz = df[["x", "y", "z"]].to_numpy(float) if "x" in df else np.cumsum(
            Rotation.from_quat(self.q).apply(self.B[:, :3]) * 0.1, axis=0)


def recordings(which: str = "all") -> list:
    import data_split as ds
    blue = ds.discover_csv_files(BLUEROV)[0]
    uri = ds.discover_csv_files(URI)[0]
    if which == "sample":   # one recording per BlueROV2 day, and URI's bay
        blue = [next(f for f in blue if d in f) for d in ("12-01-2024", "14-02-2024", "16-02-2024", "19-02-2024")]
        uri = uri[:1]
    elif which == "pwm":
        uri = []
    return [Rec(p, k) for k, p in enumerate(blue + uri)]


# --- one variant's values, the same for both routes ----------------------------------------------------------------
def values(r: Rec, v: dict) -> dict:
    """{column: values at the recording's 10 Hz} in the variant's frames and units; 'kind' says how each is logged."""
    R = Rotation.from_quat(r.q)
    B, xyz = r.B, r.xyz
    if v["actuation"] != "forces":
        # the dataset's motion is ROS FLU / ENU; its ESC PWM, through the bluerov2_heavy preset, gives FRD forces: put
        # the motion in FRD / NED so commands and motion agree (ingestion checks that they do)
        R = Rotation.from_matrix(T_FLU @ R.as_matrix() @ TB)
        B, xyz = np.hstack([r.B[:, :3] @ TB, r.B[:, 3:] @ TB]), r.xyz @ T_FLU.T
    out = {}
    if v["frame"] == "flu":   # what a ROS vehicle logs: FLU body, ENU world
        Rf = Rotation.from_matrix(T_FLU @ R.as_matrix() @ TB)
        lin, ang, F, T = B[:, :3] @ TB, B[:, 3:] @ TB, r.F[:, :3] @ TB, r.F[:, 3:] @ TB
        Rw = T_FLU
    else:
        Rf, lin, ang, F, T, Rw = R, B[:, :3], B[:, 3:], r.F[:, :3], r.F[:, 3:], np.eye(3)
    if v["orientation"] == "quat":
        out.update({f"q{c}": x for c, x in zip("xyzw", Rf.as_quat().T)})
    else:
        e = np.unwrap(Rf.as_euler("xyz"), axis=0)   # roll, pitch, yaw
        e = np.degrees(e) if v["orientation"] == "euler_deg" else e
        out.update({"roll": e[:, 0], "pitch": e[:, 1], "yaw": e[:, 2]})
    if v["velocity"] == "world":
        lin = (Rw @ (R.apply(B[:, :3])).T).T
    elif v["velocity"] == "mount":   # a DVL rolled 180 deg, a gyro yawed 90 deg: the readings in their own axes
        lin = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).inv().apply(lin)
        ang = Rotation.from_euler("xyz", [0, 0, 90], degrees=True).inv().apply(ang)
    scale = {"m/s": 1, "cm/s": 100, "mm/s": 1000, "knots": 1 / KNOTS}[v["lin_unit"]]
    out.update({c: x * scale for c, x in zip("uvw", lin.T)})
    ang = np.degrees(ang) if v["ang_unit"] == "deg/s" else ang
    out.update({c: x for c, x in zip("pqr", ang.T)})
    a = v["actuation"]
    if a == "forces":
        out.update({c: x for c, x in zip(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"), np.hstack([F, T]).T)})
    else:
        cmd = ((r.pwm - 1100) / 800 if a == "norm_range01" else (r.pwm - 1500) / 400 if a.startswith("norm")
               else r.pwm)
        out.update({f"m{i}": cmd[:, i] for i in range(8)})
        if a in ("pwm_voltage", "pwm_curve_voltage"):
            out["volt"] = 15.6 - 1.4 * np.arange(r.n) / r.n   # a battery sagging over the run
    if v["position"] != "none":
        s = {"m": 1, "cm": 100, "mm": 1000}[v["position"]]
        out.update({c: x * s for c, x in zip(("px", "py", "pz"), (Rw @ xyz.T))})
    return out


HOLD = {"m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"}
TOPIC_OF = {**{f"q{c}": "att" for c in "xyzw"}, "roll": "att", "pitch": "att", "yaw": "att",
            **{c: "gyro" for c in "pqr"}, **{c: "vel" for c in "uvw"},
            **{c: "act" for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7")},
            "volt": "batt", "px": "pos", "py": "pos", "pz": "pos"}
RATE = {"att": 50, "gyro": 50, "vel": 10, "act": 20, "batt": 10, "pos": 10}


def upsample(t: np.ndarray, cols: dict, rate: int) -> tuple:
    """A topic at its own rate from the 10 Hz recording: linear for continuous values (quaternions slerped), the last
    command held for thruster commands, as the vehicle logged them. The 10 Hz instants are kept exactly."""
    k = rate // 10
    tt = t[0] + np.arange((len(t) - 1) * k + 1) / rate
    out = {}
    qs = [c for c in ("qx", "qy", "qz", "qw") if c in cols]
    if qs:
        from rosbag_reader import slerp
        qi = slerp(t, np.stack([cols[c] for c in ("qx", "qy", "qz", "qw")], 1), tt)
        out.update({c: qi[:, i] for i, c in enumerate(("qx", "qy", "qz", "qw"))})
    for c, x in cols.items():
        if c in qs:
            continue
        out[c] = x[np.minimum(np.arange(len(tt)) // k, len(t) - 1)] if c in HOLD else np.interp(tt, t, x)
    return tt, out


def topics_of(r: Rec, vals: dict) -> dict:
    groups = {}
    for c, x in vals.items():
        groups.setdefault(TOPIC_OF[c], {})[c] = x
    return {g: upsample(r.t, cols, RATE[g]) for g, cols in groups.items()}


# --- each format's names --------------------------------------------------------------------------------------------
PX4_NAMES = {"att": "vehicle_attitude", "gyro": "vehicle_angular_velocity", "vel": "dvl_velocity",
             "act": "actuator_outputs", "batt": "battery_status", "pos": "vehicle_local_position"}
PX4_FIELD = {"qw": "q[0]", "qx": "q[1]", "qy": "q[2]", "qz": "q[3]", "p": "xyz[0]", "q": "xyz[1]", "r": "xyz[2]",
             "u": "vx", "v": "vy", "w": "vz", "volt": "voltage_v", "px": "x", "py": "y", "pz": "z",
             **{f"m{i}": f"output[{i}]" for i in range(8)}}
MOOS_VAR = {"qx": "NAV_QX", "qy": "NAV_QY", "qz": "NAV_QZ", "qw": "NAV_QW", "roll": "NAV_ROLL", "pitch": "NAV_PITCH",
            "yaw": "NAV_YAW", "p": "IMU_P", "q": "IMU_Q", "r": "IMU_R", "u": "DVL_U", "v": "DVL_V", "w": "DVL_W",
            "volt": "BATT_VOLTAGE", "px": "NAV_X", "py": "NAV_Y", "pz": "NAV_DEPTH",
            **{f"m{i}": f"THRUSTER_{i + 1}" for i in range(8)},
            **{c: f"WRENCH_{c.upper()}" for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}}
LCM_CH = {"att": "ATTITUDE", "gyro": "GYRO", "vel": "DVL", "act": "THRUSTERS", "batt": "BATTERY", "pos": "POSE"}
LCM_FIELD = {"qx": "q[0]", "qy": "q[1]", "qz": "q[2]", "qw": "q[3]", "p": "rate[0]", "q": "rate[1]", "r": "rate[2]",
             "u": "vel[0]", "v": "vel[1]", "w": "vel[2]", "volt": "volts", "px": "xyz[0]", "py": "xyz[1]",
             "pz": "xyz[2]", **{f"m{i}": f"cmd[{i}]" for i in range(8)}}


def ref_of(fmt: str, c: str, kind=None) -> str:
    g = TOPIC_OF[c]
    if fmt == "px4":
        inst = "[1]" if g == "act" and kind == "multi_instance" else ""
        return f"{PX4_NAMES[g]}{inst}.{PX4_FIELD.get(c, c)}"
    if fmt == "moos":
        if kind == "keyvalue" and g == "vel":
            return f"DVL_REPORT.{c.upper()}"
        return MOOS_VAR[c] + ("@pDVL" if kind == "source" and g == "vel" else "")
    if kind == "nested" and g == "att":
        return f"NAV.orientation.{c[1]}"
    return f"{LCM_CH[g]}.{LCM_FIELD.get(c, c)}"


def write_format(fmt: str, r: Rec, vals: dict, folder: str, v: dict, kind=None) -> str:
    os.makedirs(folder, exist_ok=True)
    tops = topics_of(r, vals)
    day = pd.to_datetime(r.date, format="%d-%m-%Y").strftime("%Y-%m-%d")
    stem = f"{day}_{os.path.splitext(r.name)[0]}"
    if fmt == "px4":
        gps = v["date"] == "recording"
        boot0 = r.t[0] - 100.0                 # the autopilot booted 100 s before the recording
        topics = {}
        for g, (tt, cols) in tops.items():
            topics[PX4_NAMES[g]] = Topic(tt - boot0, {PX4_FIELD.get(c, c): x for c, x in cols.items()},
                                         instance=1 if (g == "act" and kind == "multi_instance") else 0)
        if kind == "multi_instance":   # instance 0 of actuator_outputs carries other numbers; ours is instance 1
            tt, cols = tops["act"]
            topics["actuator_outputs#0"] = Topic(tt - boot0, {PX4_FIELD.get(c, c): x * 0 + 1500 for c, x in cols.items()})
        p = os.path.join(folder, f"{stem}.ulg")
        write_ulog(p, topics, boot0 if gps else None)
        return p
    if fmt == "moos":
        topics = {}
        for g, (tt, cols) in tops.items():
            if kind == "keyvalue" and g == "vel":
                txt = [",".join(f"{c.upper()}={float(cols[c][i])!r}" for c in cols) + ",QUALITY=good"
                       for i in range(len(tt))]
                topics[g] = Topic(tt, {}, string_fields={"DVL_REPORT": txt})
            else:
                topics[g] = Topic(tt, {MOOS_VAR[c]: x for c, x in cols.items()})
        srcs = {"vel": "pDVL"}
        if kind == "source":   # another app posts the same DVL variables (a simulator left running)
            tt, cols = tops["vel"]
            topics["vel_sim"] = Topic(tt + 0.05, {MOOS_VAR[c]: x * 0 + 9.0 for c, x in cols.items()})
            srcs["vel_sim"] = "uSimMarine"
        p = os.path.join(folder, f"{stem}.alog")
        write_alog(p, topics, logstart=r.t[0] - 2.0, sources=srcs)
        return p
    topics = {}
    for g, (tt, cols) in tops.items():
        ch = LCM_CH[g]
        if kind == "nested" and g == "att":
            topics["NAV"] = Topic(tt, {f"orientation.{c[1]}": x for c, x in cols.items()})
        else:
            topics[ch] = Topic(tt, {LCM_FIELD.get(c, c): x for c, x in cols.items()})
    p = os.path.join(folder, f"lcmlog-{stem}.00")
    write_lcm(p, os.path.join(folder, "types"), topics, with_utime=True, receive_delay_s=0.0)
    return p


def write_csv(r: Rec, vals: dict, folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    day = pd.to_datetime(r.date, format="%d-%m-%Y").strftime("%Y-%m-%d")
    p = os.path.join(folder, f"{day}_{os.path.splitext(r.name)[0]}.csv")
    pd.DataFrame({"t": r.t, **vals}).to_csv(p, index=False, float_format="%.17g")
    return p


# --- specs ----------------------------------------------------------------------------------------------------------
def spec_for(fmt: str, v: dict, files_glob: str, _unused, folder: str, kind=None) -> dict:
    col = (lambda c: c) if fmt == "csv" else (lambda c: ref_of(fmt, c, kind))
    s = {"vehicle": "stress94", "source": {"files": files_glob}, "frame": v["frame"]}
    if fmt == "csv":
        s["time"] = {"column": "t", "unit": "s"}
        if v["resample"] != 10:
            s["resample_hz"] = v["resample"]
    else:
        s["source"]["format"] = fmt
        s["resample_hz"] = v["resample"]
        if fmt == "lcm":
            s["source"]["lcm_types"] = [os.path.join(folder, "types", "*.lcm")]
    s["session_date"] = ({"from": "recording"} if v["date"] == "recording" else
                         {"from": "constant", "value": "2024-02-16"} if v["date"] == "constant" else
                         {"from": "filename", "pattern": r"(\d{4}-\d{2}-\d{2})", "format": "%Y-%m-%d"})
    s["body_velocity"] = {"columns": {c: col(c) for c in "uvwpqr"}, "linear_unit": v["lin_unit"],
                          "angular_unit": v["ang_unit"]}
    if v["velocity"] == "world":
        s["body_velocity"]["linear_frame"] = "world"
    if v["velocity"] == "mount":
        s["body_velocity"].update({"linear_mount_rpy_deg": [180, 0, 0], "angular_mount_rpy_deg": [0, 0, 90]})
    s["orientation"] = ({"quaternion": {c: col(f"q{c}") for c in "xyzw"}} if v["orientation"] == "quat" else
                        {"euler": {k: col(k) for k in ("roll", "pitch", "yaw")},
                         "unit": "deg" if v["orientation"] == "euler_deg" else "rad"})
    if v["orientation"] != "quat":
        s["orientation"]["euler"]["unit"] = s["orientation"].pop("unit")
    a = v["actuation"]
    if a == "forces":
        s["wrench"] = {"columns": {c: col(c) for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}}
    else:
        th = {"pwm_columns": [col(f"m{i}") for i in range(8)]}
        cfg = tm.bluerov2_heavy()
        if a in ("pwm_preset", "pwm_voltage", "pwm_nominal_voltage"):
            th["preset"] = "bluerov2_heavy"
            if a == "pwm_voltage":
                th["voltage_column"] = col("volt")
            if a == "pwm_nominal_voltage":
                th["voltage"] = 14.8
        elif a == "pwm_curve_voltage":   # a lab's own T200 table, measured at 16 V, scaled to each logged voltage
            c = tm.ThrustCurve.t200_bluerov2_dataset()
            p = os.path.join(os.path.dirname(folder), "curve_pwm.csv")   # next to the logs, not among them
            pd.DataFrame({"pwm_us": c.pwm_us, "thrust_n": c.thrust_n}).to_csv(p, index=False)
            th.update({"layout": _layout(cfg), "curve_csv": p, "curve_voltage": 16.0, "voltage_column": col("volt")})
        elif a in ("pwm_layout", "pwm_t200_layout"):
            th["layout"] = _layout(cfg)
            if a == "pwm_t200_layout":
                th["curve"] = "t200"
        elif a == "pwm_allocation":
            p = os.path.join(os.path.dirname(folder), "allocation.csv")
            np.savetxt(p, cfg.allocation, delimiter=",", fmt="%.17g")
            th.update({"allocation_csv": p, "directions": cfg.directions.astype(int).tolist()})
        else:
            th.update({"preset": "bluerov2_heavy", "command": "normalized",
                       "command_range": [0, 1] if a == "norm_range01" else [-1, 1]})
            if a == "norm_custom":   # your own command-to-thrust table: with a layout (a preset fixes its curve)
                c = tm.ThrustCurve.t200_bluerov2_dataset()
                p = os.path.join(os.path.dirname(folder), "curve.csv")
                pd.DataFrame({"command": (c.pwm_us - 1500) / 400, "thrust_n": c.thrust_n}).to_csv(p, index=False)
                del th["preset"]
                th.update({"layout": _layout(cfg), "curve_csv": p})
        s["wrench"] = {"thrusters": th}
    if v["position"] != "none":
        s["position"] = {"x": col("px"), "y": col("py"), "z": col("pz"), "unit": v["position"]}
    if v["exclude"] != "none":   # a speed glitch rule; with min_segment_s, shorter pieces are discarded
        lim = 0.25 * {"m/s": 1, "cm/s": 100, "mm/s": 1000, "knots": 1 / KNOTS}[v["lin_unit"]]
        s["exclude_samples"] = [{"column": col("u"), **({"below": -lim} if v["exclude"] == "rule_below"
                                                         else {"above": lim})}]
        if v["exclude"] == "rule_minseg30":
            s["min_segment_s"] = 30.0
    if v["checks"] != "none":
        s["skip_checks"] = ["force_direction" if v["checks"] == "skip_force" else "orientation_vs_position"]
    if v["stamp"] != "default" and fmt == "lcm":
        s["source"]["stamp"] = v["stamp"]
    return s


def _layout(cfg) -> list:
    """The preset's thrusters as a layout: positions (least-norm, giving the same torques) and directions."""
    B = cfg.allocation
    return [{"position_m": (np.cross(B[:3, i], B[3:, i]) / np.dot(B[:3, i], B[:3, i])).round(12).tolist(),
             "direction": B[:3, i].round(12).tolist(), "reversed": bool(cfg.directions[i] < 0)} for i in range(8)]


# --- one round trip -------------------------------------------------------------------------------------------------
def round_trip(fmt: str, recs: list, v: dict, tag: str, kind=None) -> dict:
    """Ingest the variant through `fmt` and through a CSV with the same content; compare the canonical outputs."""
    root = os.path.join(DATA, tag)
    shutil.rmtree(root, ignore_errors=True)
    fdir, cdir = os.path.join(root, fmt, "logs"), os.path.join(root, "csv", "logs")
    for r in recs:
        vals = values(r, v)
        write_format(fmt, r, vals, fdir, v, kind)
        write_csv(r, vals, cdir)
    pat = {"px4": "*.ulg", "moos": "*.alog", "lcm": "lcmlog-*"}[fmt]
    outs = {}
    t0 = time.time()
    for route, folder, glob_ in ((fmt, fdir, os.path.join(fdir, pat)), ("csv", cdir, os.path.join(cdir, "*.csv"))):
        spec = spec_for(route, v, glob_, None, folder, kind)
        sp = os.path.join(root, f"{route}.yaml")
        yaml.safe_dump(spec, open(sp, "w"), sort_keys=False)
        out = os.path.join(root, route, "ingested")
        try:
            outs[route] = (out, di.ingest(di.load_ingest_spec(sp), out))
        except Exception as e:
            return {"error": f"{route}: {type(e).__name__}: {e}"[:400], "seconds": time.time() - t0}
    summ = outs[fmt][1]
    warned = summ.get("warnings", []) + [w for f in summ.get("files", []) for w in f.get("warnings", [])]
    return compare(outs[fmt][0], outs["csv"][0], v) | {"seconds": round(time.time() - t0, 1),
                                                        "warnings_fmt": list(dict.fromkeys(warned))[:8]}


def compare(a: str, b: str, v: dict) -> dict:
    fa = sorted(f for f in os.listdir(a) if f.endswith(".csv"))
    fb = sorted(f for f in os.listdir(b) if f.endswith(".csv"))
    if fa != fb:
        return {"same_files": False, "fmt_files": fa[:5], "csv_files": fb[:5]}
    worst, rows = {}, 0
    for f in fa:
        x, y = pd.read_csv(os.path.join(a, f)), pd.read_csv(os.path.join(b, f))
        n = min(len(x), len(y))
        rows += n
        if abs(len(x) - len(y)) > (1 if v["resample"] != 10 else 0):
            return {"same_files": True, "rows_differ": [f, len(x), len(y)]}
        for c in di.CANONICAL_COLUMNS:
            if c in ("qx", "qy", "qz", "qw"):
                continue
            d = float(np.max(np.abs(x[c].to_numpy()[:n] - y[c].to_numpy()[:n])))
            worst[c] = max(worst.get(c, 0.0), d)
        # orientation: q and -q are the same rotation (the model uses it sign-free), so compare the rotations
        qa, qb = x[["qx", "qy", "qz", "qw"]].to_numpy()[:n], y[["qx", "qy", "qz", "qw"]].to_numpy()[:n]
        worst["orientation"] = max(worst.get("orientation", 0.0),
                                   float(np.max(np.minimum(np.abs(qa - qb).max(1), np.abs(qa + qb).max(1)))))
    return {"same_files": True, "recordings": len(fa), "rows": rows, "max_abs_diff": max(worst.values()),
            "worst_column": max(worst, key=worst.get)}


TOL = 1e-6   # well under any sensor's resolution; the routes differ only by float rounding in interpolation


def _ok(res: dict) -> bool:
    return "error" not in res and res.get("same_files") and "rows_differ" not in res and res["max_abs_diff"] < TOL


# --- fast rows ------------------------------------------------------------------------------------------------------
def r_full() -> None:
    """Every real recording, through each format, the baseline variant."""
    recs = recordings("all")
    for fmt in FORMATS:
        res = round_trip(fmt, recs, BASE, f"R-{fmt}")
        record(F, f"R-{fmt}", "format", fmt, f"all {len(recs)} real recordings through {fmt} vs CSV", "bluerov+uri",
               res, f"same canonical recordings (max |diff| < {TOL:g})", _ok(res), res.get("seconds", 0))


def v_one_at_a_time() -> None:
    for fmt in FORMATS:
        for var, opts in OPTIONS.items():
            for o in opts:
                if o == BASE[var] and var != "actuation" or var == "stamp" and fmt != "lcm":
                    continue
                v = {**BASE, var: o}
                recs = recordings("pwm" if str(v["actuation"]).startswith(("pwm", "norm")) else "sample")
                if v["actuation"] != "forces":
                    recs = recs[:2]
                res = round_trip(fmt, recs, v, f"V-{fmt}-{var}-{o}".replace("/", "_"))
                record(F, f"V-{fmt}-{var}-{o}", var, o, f"{fmt}: {var} = {o}", "bluerov sample", res,
                       f"same as the CSV route (< {TOL:g})", _ok(res), res.get("seconds", 0))


def pairwise_cases(names: list) -> list:
    """All-pairs combinations of OPTIONS (a greedy cover: every pair of values of any two variables appears)."""
    need = {(a, x, b, y) for a, b in itertools.combinations(names, 2) for x in OPTIONS[a] for y in OPTIONS[b]}
    cases = []
    rng = np.random.default_rng(94)
    while need:
        best, gain = None, -1
        for _ in range(60):
            c = {n: OPTIONS[n][rng.integers(len(OPTIONS[n]))] for n in names}
            g = sum((a, c[a], b, c[b]) in need for a, b in itertools.combinations(names, 2))
            if g > gain:
                best, gain = c, g
        cases.append(best)
        need -= {(a, best[a], b, best[b]) for a, b in itertools.combinations(names, 2)}
    return cases


PAIR_NAMES = ["actuation", "velocity", "orientation", "frame", "lin_unit", "ang_unit", "date", "position"]


def p_pairwise() -> None:
    cases = pairwise_cases(PAIR_NAMES)
    for fmt in FORMATS:
        for i, c in enumerate(cases):
            v = {**BASE, **c}
            recs = recordings("pwm")[:1] if v["actuation"] != "forces" else recordings("sample")[:2]
            res = round_trip(fmt, recs, v, f"P-{fmt}-{i}")
            record(F, f"P-{fmt}-{i}", "pairwise", json.dumps(c), f"{fmt}: all-pairs case {i + 1} of {len(cases)}",
                   "bluerov sample", res, f"same as the CSV route (< {TOL:g})", _ok(res), res.get("seconds", 0))


def k_field_kinds() -> None:
    recs = recordings("sample")[:1]
    cases = [("px4", "multi_instance", {**BASE, "actuation": "pwm_preset"}),
             ("moos", "keyvalue", BASE), ("moos", "source", BASE), ("lcm", "nested", BASE)]
    for fmt, kind, v in cases:
        rr = recordings("pwm")[:1] if v["actuation"] != "forces" else recs
        res = round_trip(fmt, rr, v, f"K-{fmt}-{kind}", kind)
        record(F, f"K-{fmt}-{kind}", "field kind", kind, f"{fmt}: {kind} fields", "bluerov", res,
               f"same as the CSV route (< {TOL:g})", _ok(res), res.get("seconds", 0))


# --- damage ---------------------------------------------------------------------------------------------------------
def _one_log(fmt: str, v=BASE, kind=None) -> tuple:
    r = recordings("sample")[0]
    folder = tempfile.mkdtemp(prefix=f"d94_{fmt}_", dir=DATA)
    vals = values(r, v)
    p = write_format(fmt, r, vals, folder, v, kind)
    spec = spec_for(fmt, v, p, None, folder, kind)
    return r, vals, p, spec, folder


def _ingest_one(spec: dict, folder: str):
    sp = os.path.join(folder, "spec.yaml")
    yaml.safe_dump(spec, open(sp, "w"))
    return di.ingest(di.load_ingest_spec(sp), os.path.join(folder, "out"))


def d_damage() -> None:
    os.makedirs(DATA, exist_ok=True)
    for fmt in FORMATS:
        r, vals, p, spec, folder = _one_log(fmt)
        _ingest_one(spec, folder)
        clean = pd.read_csv(sorted(_glob(os.path.join(folder, "out", "*.csv")))[0])
        raw = open(p, "rb").read()
        cases = {
            "truncated_50pct": raw[: len(raw) // 2],
            "truncated_99pct": raw[: int(len(raw) * 0.99)],
            "corrupt_middle": raw[: len(raw) // 2] + bytes(np.random.default_rng(1).integers(0, 256, 400, dtype=np.uint8)) + raw[len(raw) // 2 + 400:],
            "empty": b"",
        }
        for label, data in cases.items():
            q = p.replace(os.path.basename(p), f"dmg_{label}_{os.path.basename(p)}")
            open(q, "wb").write(data)
            res = _damage_check(fmt, spec, q, folder, clean, label)
            record(F, f"D-{fmt}-{label}", "damage", label, f"{fmt}: {label.replace('_', ' ')}", r.name, res,
                   "a counted warning and correct values where the log is intact, or a plain refusal", res["ok"])
            os.remove(q)
        for label in ("gap", "nan", "out_of_order", "duplicates", "missing_variable", "string_value", "wrong_types"):
            if label == "string_value" and fmt != "moos" or label == "wrong_types" and fmt != "lcm":
                continue
            res = _structural(fmt, label, r, vals, folder, spec, clean)
            record(F, f"D-{fmt}-{label}", "damage", label, f"{fmt}: {label.replace('_', ' ')}", r.name, res,
                   "a counted warning and correct values, or a plain refusal", res["ok"])
        shutil.rmtree(folder, ignore_errors=True)


def _glob(p):
    import glob
    return glob.glob(p)


def _damage_check(fmt, spec, q, folder, clean, label) -> dict:
    s = {**spec, "source": {**spec["source"], "files": q}}
    out = os.path.join(folder, f"out_{label}")
    shutil.rmtree(out, ignore_errors=True)
    sp = os.path.join(folder, f"{label}.yaml")
    yaml.safe_dump(s, open(sp, "w"))
    try:
        summ = di.ingest(di.load_ingest_spec(sp), out)
    except di.IngestError as e:
        return {"ok": len(str(e)) > 20 and "Traceback" not in str(e), "refused": str(e)[:240]}
    except Exception as e:
        return {"ok": False, "crash": f"{type(e).__name__}: {e}"[:240]}
    got = pd.read_csv(_glob(os.path.join(out, "*.csv"))[0])
    n = min(len(got), len(clean))
    head = float(np.max(np.abs(got[di.CANONICAL_COLUMNS].to_numpy()[: n // 2] -
                               clean[di.CANONICAL_COLUMNS].to_numpy()[: n // 2])))
    warned = summ.get("warnings", []) + [w for f in summ.get("files", []) for w in f.get("warnings", [])]
    return {"ok": head < TOL and (bool(warned) or len(got) == len(clean)), "rows": len(got), "clean_rows": len(clean),
            "max_abs_diff_first_half": head, "warnings": warned[:3]}


def _structural(fmt, label, r, vals, folder, spec, clean) -> dict:
    """Damage made in the content, then written: gaps, NaN, order, duplicates, a missing variable, wrong types."""
    v = dict(vals)
    sub = os.path.join(folder, label)
    os.makedirs(sub, exist_ok=True)
    s = json.loads(json.dumps(spec))
    if label == "gap":   # 3 s of DVL lost
        tops = topics_of(r, v)
        tt, cols = tops["vel"]
        keep = (tt < tt[0] + 100) | (tt > tt[0] + 103)
        tops["vel"] = (tt[keep], {c: x[keep] for c, x in cols.items()})
        p = _write_tops(fmt, r, tops, sub)
    elif label == "nan":
        tops = topics_of(r, v)
        tt, cols = tops["vel"]
        cols = {c: x.copy() for c, x in cols.items()}
        cols["u"][50:55] = np.nan
        tops["vel"] = (tt, cols)
        p = _write_tops(fmt, r, tops, sub)
    elif label in ("out_of_order", "duplicates"):
        tops = topics_of(r, v)
        tt, cols = tops["vel"]
        idx = np.arange(len(tt))
        if label == "out_of_order":
            idx[100:110] = idx[100:110][::-1]
        else:
            idx = np.sort(np.r_[idx, idx[200:210]])
        tops["vel"] = (tt[idx], {c: x[idx] for c, x in cols.items()})
        p = _write_tops(fmt, r, tops, sub)
    elif label == "missing_variable":
        p = _write_tops(fmt, r, topics_of(r, v), sub)
        s["body_velocity"]["columns"]["u"] = {"px4": "dvl_velocity.vnope", "moos": "DVL_NOPE",
                                              "lcm": "DVL.nope[0]"}[fmt]
    elif label == "string_value":   # MOOS: a string posted under a numeric variable
        p = _write_tops(fmt, r, topics_of(r, v), sub)
        with open(p, "a") as f:
            f.write(f"{r.t[200] - (r.t[0] - 2.0):.6f}   DVL_U   pDVL   no_lock\n")
    else:   # LCM: type files that don't match the log
        p = _write_tops(fmt, r, topics_of(r, v), sub)
        tdir = os.path.join(sub, "types")
        for f in _glob(os.path.join(tdir, "*.lcm")):
            open(f, "w").write(open(f).read().replace("double", "float", 1))
        s["source"]["lcm_types"] = [os.path.join(tdir, "*.lcm")]
    s["source"]["files"] = p
    if fmt == "lcm" and label != "wrong_types":
        s["source"]["lcm_types"] = [os.path.join(sub, "types", "*.lcm")]
    sp = os.path.join(sub, "spec.yaml")
    yaml.safe_dump(s, open(sp, "w"))
    try:
        summ = di.ingest(di.load_ingest_spec(sp), os.path.join(sub, "out"))
    except di.IngestError as e:
        want_refusal = label in ("missing_variable", "wrong_types")
        msg = str(e)
        named = ("nope" in msg or "NOPE" in msg) if label == "missing_variable" else ("type" in msg)
        return {"ok": want_refusal and named, "refused": msg[:260]}
    except Exception as e:
        return {"ok": False, "crash": f"{type(e).__name__}: {e}"[:260]}
    if label in ("missing_variable", "wrong_types"):
        return {"ok": False, "accepted": True}
    got = pd.read_csv(_glob(os.path.join(sub, "out", "*.csv"))[0])
    n = min(len(got), len(clean))
    diff = np.abs(got[di.CANONICAL_COLUMNS].to_numpy()[:n] - clean[di.CANONICAL_COLUMNS].to_numpy()[:n])
    bad_rows = int(np.sum(diff.max(axis=1) > TOL))
    warned = summ.get("warnings", []) + [w for f in summ.get("files", []) for w in f.get("warnings", [])]
    allowed = {"gap": 40, "nan": 12, "out_of_order": 0, "duplicates": 0, "string_value": 0}[label]
    return {"ok": bad_rows <= allowed and (bool(warned) or label == "out_of_order"), "rows_changed": bad_rows,
            "allowed": allowed, "warnings": warned[:3]}


def _write_tops(fmt, r, tops, folder) -> str:
    """write_format, from already-made topics (for damage)."""
    v = BASE
    vals_like = {c: None for g in tops.values() for c in g[1]}
    saved = topics_of
    globals()["topics_of"] = lambda r_, vals_: tops
    try:
        return write_format(fmt, r, vals_like, folder, v)
    finally:
        globals()["topics_of"] = saved


# --- size -----------------------------------------------------------------------------------------------------------
def s_size() -> None:
    """One hour at 50 Hz through each format: wall time and peak memory of `strathe ingest` (a fresh process)."""
    r0 = recordings("sample")[0]
    n = 36000   # 1 h at 10 Hz, made from the real recording repeated
    reps = int(np.ceil(n / r0.n))
    big = Rec.__new__(Rec)
    big.name, big.date, big.n = "odom-12-01-2024-run99.csv", r0.date, n
    big.t = r0.t[0] + np.arange(n) * 0.1
    big.F, big.B = np.tile(r0.F, (reps, 1))[:n], np.tile(r0.B, (reps, 1))[:n]
    big.q, big.pwm, big.xyz = np.tile(r0.q, (reps, 1))[:n], None, np.tile(r0.xyz, (reps, 1))[:n]
    for fmt in FORMATS:
        folder = os.path.join(DATA, f"S-{fmt}")
        shutil.rmtree(folder, ignore_errors=True)
        saved = dict(RATE)
        RATE.update({"att": 50, "gyro": 50, "vel": 50, "act": 50})
        try:
            p = write_format(fmt, big, values(big, BASE), folder, BASE)
        finally:
            RATE.clear()
            RATE.update(saved)
        spec = spec_for(fmt, BASE, p, None, folder)
        sp = os.path.join(folder, "spec.yaml")
        yaml.safe_dump(spec, open(sp, "w"))
        t0 = time.time()
        code = ("import resource,sys;sys.path.insert(0,%r);import data_ingestion as di;"
                "di.ingest(di.load_ingest_spec(%r),%r);"
                "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)") % (CAL, sp, os.path.join(folder, "out"))
        pr = subprocess.run([PY, "-c", code], capture_output=True, text=True)
        secs = time.time() - t0
        rss = int(pr.stdout.strip().splitlines()[-1]) if pr.returncode == 0 else -1
        gb = rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)
        record(F, f"S-{fmt}", "size", "1 h at 50 Hz", f"{fmt}: a one-hour log ({os.path.getsize(p) / 1e6:.0f} MB)",
               "bluerov repeated", {"seconds": round(secs, 1), "peak_gb": round(gb, 2), "exit": pr.returncode,
                                    "stderr": pr.stderr[-200:]},
               "under 2 minutes and 2 GB", pr.returncode == 0 and secs < 120 and gb < 2, secs)


# --- entry points and the inventory's spec rows ---------------------------------------------------------------------
def e_entry_points() -> None:
    """`strathe ingest`, `strathe calibrate --ingest-spec` (short budget) and the web Convert logs card, per format."""
    import strathe_web as web
    for fmt in FORMATS:
        r, vals, p, spec, folder = _one_log(fmt)
        spec["source"]["files"] = os.path.join(folder, "*" + os.path.splitext(p)[1]) if fmt != "lcm" else \
            os.path.join(folder, "lcmlog-*")
        sp = os.path.join(folder, "spec.yaml")
        yaml.safe_dump(spec, open(sp, "w"))
        rc, s1 = strathe(["ingest", "--spec", sp, "--out", os.path.join(folder, "cli_out")],
                         os.path.join(folder, "ingest.log"))
        record(F, f"EP-{fmt}-ingest", "entry", "strathe ingest", f"{fmt}: strathe ingest --spec", r.name,
               {"exit": rc, "files": len(_glob(os.path.join(folder, "cli_out", "*.csv")))}, "converts", rc == 0, s1)
        m = web.JobManager(os.path.join(folder, "web"))
        h = web.StratheHandler.__new__(web.StratheHandler)
        h.manager = m
        job = h._start("/jobs/ingest", {"spec": sp, "out": os.path.join(folder, "web_out")})
        job.proc.wait()
        record(F, f"EP-{fmt}-web", "entry", "web Convert logs", f"{fmt}: the web page's Convert logs card", r.name,
               {"exit": job.proc.returncode}, "the job converts", job.proc.returncode == 0)
    r, vals, p, spec, folder = _one_log("px4")
    recs = recordings("sample")
    for fmt in FORMATS:   # calibrate straight from the logs, two days so it holds one out
        folder = os.path.join(DATA, f"EP-{fmt}-cal")
        shutil.rmtree(folder, ignore_errors=True)
        for rr in recs[:3]:
            write_format(fmt, rr, values(rr, BASE), folder, BASE)
        pat = {"px4": "*.ulg", "moos": "*.alog", "lcm": "lcmlog-*"}[fmt]
        spec = spec_for(fmt, BASE, os.path.join(folder, pat), None, folder)
        sp = os.path.join(folder, "spec.yaml")
        yaml.safe_dump(spec, open(sp, "w"))
        rc, secs = strathe(["calibrate", "--ingest-spec", sp, "--name", f"stress94_{fmt}", "--out",
                            os.path.join(folder, "cal"), "--max-nfev", "3", "--skip-identifiability"],
                           os.path.join(folder, "calibrate.log"))
        ok = rc == 0 and os.path.exists(os.path.join(folder, "cal", "calibration_report.md"))
        record(F, f"EP-{fmt}-calibrate", "entry", "calibrate --ingest-spec", f"{fmt}: calibrate straight from the logs",
               "bluerov 3 days", {"exit": rc}, "converts, calibrates, reports", ok, secs)


def i_independent() -> None:
    """Independent decoders on the same files: pyulog's own CSV export, a plain-text read of the .alog, lcm's own
    EventLog decode."""
    r = recordings("sample")[0]
    for fmt in FORMATS:
        folder = tempfile.mkdtemp(prefix=f"i94_{fmt}_", dir=DATA)
        vals = values(r, BASE)
        p = write_format(fmt, r, vals, folder, BASE)
        if fmt == "px4":
            import ulog_reader
            pr = subprocess.run([os.path.join(os.path.dirname(PY), "ulog2csv"), "-o", folder, p],
                                capture_output=True, text=True)
            csvs = {os.path.basename(f): pd.read_csv(f) for f in _glob(os.path.join(folder, "*.csv"))}
            ref = next(d for n, d in csvs.items() if "dvl_velocity" in n)
            ours = ulog_reader.read_log(p, ["dvl_velocity.vx"], 10)[0]
            # ulog2csv's raw messages at the 10 Hz instants against ours on the grid
            t_ref = (ref["timestamp"].to_numpy() / 1e6)
            got = np.interp(t_ref[:200], ours["__bag_time_s__"].to_numpy() - ours["__bag_time_s__"].to_numpy()[0]
                            + t_ref[0], ours["dvl_velocity.vx"].to_numpy())
            d = float(np.max(np.abs(got - ref["vx"].to_numpy()[:200])))
            record(F, f"I-{fmt}", "independent", "pyulog ulog2csv", "px4: values against pyulog's own ulog2csv export",
                   r.name, {"exit": pr.returncode, "max_abs_diff": d}, f"< {TOL:g}", pr.returncode == 0 and d < TOL)
        elif fmt == "moos":
            import moos_reader
            lines = [ln.split() for ln in open(p) if ln and not ln.startswith("%%")]
            plain = np.array([float(x[3]) for x in lines if x[1] == "DVL_U"])
            ours = moos_reader.read_log(p, ["DVL_U"], 10)[0]["DVL_U"].to_numpy()
            d = float(np.max(np.abs(plain[:len(ours)] - ours)))
            record(F, f"I-{fmt}", "independent", "plain text", "moos: values against a plain-text read of the .alog",
                   r.name, {"max_abs_diff": d}, f"< {TOL:g}", d < TOL)
        else:
            import lcm
            import lcm_reader
            types = lcm_reader.load_types([os.path.join(folder, "types", "*.lcm")])
            log = lcm.EventLog(p, "r")
            own = []
            while True:
                ev = log.read_next_event()
                if ev is None:
                    break
                if ev.channel == "DVL":
                    own.append(types[ev.data[:8]].decode(ev.data).vel[0])
            ours = lcm_reader.read_log(p, ["DVL.vel[0]"], 10, type_files=[os.path.join(folder, "types", "*.lcm")])[0]
            d = float(np.max(np.abs(np.array(own[:len(ours)]) - ours["DVL.vel[0]"].to_numpy())))
            record(F, f"I-{fmt}", "independent", "lcm EventLog", "lcm: values against lcm's own decode", r.name,
                   {"max_abs_diff": d}, f"< {TOL:g}", d < TOL)
        shutil.rmtree(folder, ignore_errors=True)


def x_spec_rules() -> None:
    """The spec's rules for log formats: a time block, message_types or lcm_types where they don't belong, a missing
    resample_hz, lcm without lcm_types: each refused with a sentence."""
    base = {"vehicle": "x", "source": {"files": "x"}, "session_date": {"from": "recording"},
            "body_velocity": {"columns": {c: c for c in "uvwpqr"}},
            "orientation": {"quaternion": {c: c for c in "xyzw"}},
            "wrench": {"columns": {c: c for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}}}
    for fmt in FORMATS:
        cases = {"time_block": {"time": {"column": "t"}, "resample_hz": 10},
                 "no_resample": {},
                 "message_types": {"resample_hz": 10, "source": {"files": "x", "format": fmt, "message_types": ["a.msg"]}},
                 **({"stamp": {"resample_hz": 10, "source": {"files": "x", "format": fmt, "stamp": "receive"}}}
                    if fmt != "lcm" else {}),
                 "lcm_types_elsewhere" if fmt != "lcm" else "lcm_without_types":
                     {"resample_hz": 10, "source": {"files": "x", "format": fmt,
                                                    **({"lcm_types": ["t.lcm"]} if fmt != "lcm" else {})}}}
        for label, extra in cases.items():
            s = json.loads(json.dumps(base))
            s["source"]["format"] = fmt
            s.update({k: v for k, v in extra.items() if k != "source"})
            if "source" in extra:
                s["source"] = extra["source"]
            try:
                di.IngestSpec.model_validate(s)
                ok, msg = False, "accepted"
            except Exception as e:
                ok, msg = "Traceback" not in str(e), str(e).splitlines()[-1][:200] if str(e) else ""
                ok = ok and any(w in str(e) for w in ("time", "resample_hz", "message_types", "lcm_types", "stamp"))
            record(F, f"X-{fmt}-{label}", "spec rule", label, f"{fmt}: {label.replace('_', ' ')} refused", "-",
                   {"message": msg}, "a plain refusal naming the key", ok)


def c_check() -> None:
    """`strathe check` lists pyulog and lcm: present, and (with an import that fails) missing, with the pip line."""
    rc, secs = strathe(["check"], os.path.join(DATA, "check.log"))
    text = open(os.path.join(DATA, "check.log")).read()
    present = "✅ pyulog" in text and "✅ lcm" in text
    fake = tempfile.mkdtemp(prefix="nolcm_", dir=DATA)
    for m in ("pyulog", "lcm"):
        open(os.path.join(fake, f"{m}.py"), "w").write("raise ImportError('not installed')\n")
    pr = subprocess.run([PY, os.path.join(CAL, "strathe_cli.py"), "check"], capture_output=True, text=True,
                        env={**os.environ, "PYTHONPATH": fake})
    missing = "➖ pyulog not installed" in pr.stdout and "➖ lcm not installed" in pr.stdout
    record(F, "C-check", "entry", "strathe check", "check lists the PX4 and LCM packages, installed and not", "-",
           {"present": present, "missing_shown": missing}, "✅ when installed, ➖ with the reason when not",
           present and missing, secs)


def all_rows() -> set:
    """Every row id this harness writes (the coverage test checks the matrix against it)."""
    rows = {f"R-{f}" for f in FORMATS} | {"C-check", "I-px4-real", "I-px4-real-ingest"}
    for f in FORMATS:
        rows |= {f"V-{f}-{var}-{o}" for var, opts in OPTIONS.items() for o in opts
                 if (o != BASE[var] or var == "actuation") and (var != "stamp" or f == "lcm")}
        rows |= {f"P-{f}-{i}" for i in range(len(pairwise_cases(PAIR_NAMES)))}
        rows |= {f"D-{f}-{d}" for d in ("truncated_50pct", "truncated_99pct", "corrupt_middle", "empty", "gap", "nan",
                                        "out_of_order", "duplicates", "missing_variable")}
        rows |= {f"S-{f}", f"I-{f}", f"EP-{f}-ingest", f"EP-{f}-web", f"EP-{f}-calibrate"}
        rows |= {f"X-{f}-{x}" for x in ("time_block", "no_resample", "message_types",
                                       *(("stamp", "lcm_types_elsewhere") if f != "lcm" else ("lcm_without_types",)))}
        rows |= {f"E-bluerov-{f}", f"E-uri-{f}"}
    rows |= {"D-moos-string_value", "D-lcm-wrong_types", "K-px4-multi_instance", "K-moos-keyvalue", "K-moos-source",
             "K-lcm-nested", "M-frame_sign_flipped", "M-PX4_time_unit_wrong_by_1000", "M-array_index_dropped"}
    return rows


PUBLIC_PX4 = os.path.expanduser("~/strathe_public_logs/px4/bluerov2_heavy_2024-07-11.ulg")   # logs.px4.io, CC-BY


def i_real_px4() -> None:
    """A real underwater PX4 log (a BlueROV2 Heavy, PX4's public archive): our reader against pyulog's own ulog2csv
    export, and the log through ingestion as a lab would read it."""
    import glob
    from pyulog import ULog
    from rosbag_reader import slerp
    import ulog_reader as ur
    if not os.path.exists(PUBLIC_PX4):
        return record(F, "I-px4-real", "independent", "public PX4 log", "the public log isn't downloaded", "-", {},
                      "present", False)
    csvd = os.path.join(os.path.dirname(PUBLIC_PX4), "ulog2csv")
    if not glob.glob(os.path.join(csvd, "*.csv")):
        os.makedirs(csvd, exist_ok=True)
        subprocess.run([os.path.join(os.path.dirname(PY), "ulog2csv"), "-o", csvd, PUBLIC_PX4], capture_output=True)
    cols = ["vehicle_attitude.q[0]", "vehicle_attitude.q[1]", "vehicle_attitude.q[2]", "vehicle_attitude.q[3]",
            "vehicle_angular_velocity.xyz[0]", "vehicle_angular_velocity.xyz[1]", "vehicle_angular_velocity.xyz[2]",
            "vehicle_local_position.vx", "vehicle_local_position.vy", "vehicle_local_position.vz",
            *[f"actuator_outputs.output[{i}]" for i in range(8)], "battery_status.voltage_v"]
    pwm = [c for c in cols if "actuator" in c]
    qg = [["vehicle_attitude.q[1]", "vehicle_attitude.q[2]", "vehicle_attitude.q[3]", "vehicle_attitude.q[0]"]]
    df, rep = ur.read_log(PUBLIC_PX4, cols, 10, hold_columns=pwm, quaternion_groups=qg)
    off = ur._utc_offset_us(ULog(PUBLIC_PX4, list(ur.GPS_TOPICS))) or 0.0
    tb = df["__bag_time_s__"].to_numpy() - off / 1e6
    worst = {}
    for c in cols:
        topic, field = c.split(".", 1)
        r = pd.read_csv(glob.glob(os.path.join(csvd, f"*_{topic}_0.csv"))[0])
        t = r["timestamp"].to_numpy() / 1e6
        if topic == "vehicle_attitude":
            q = slerp(t, r[["q[1]", "q[2]", "q[3]", "q[0]"]].to_numpy(), tb)[:, ["q[1]", "q[2]", "q[3]", "q[0]"].index(field)]
            d = np.minimum(np.abs(q - df[c]), np.abs(q + df[c])).max()
        elif c in pwm:
            d = np.abs(r[field].to_numpy()[np.clip(np.searchsorted(t, tb + 1e-6, side="right") - 1, 0, len(t) - 1)]
                       - df[c]).max()
        else:
            d = np.abs(np.interp(tb, t, r[field].to_numpy()) - df[c]).max()
        worst[c] = float(d)
    # ulog2csv prints about 8 significant digits: agreement to that is agreement
    ok = max(worst.values()) < 1e-5 and all(worst[c] == 0 for c in pwm)
    record(F, "I-px4-real", "independent", "pyulog ulog2csv, real log",
           "px4: a real BlueROV2 Heavy log (PX4 public archive) against pyulog's ulog2csv", os.path.basename(PUBLIC_PX4),
           {"max_abs_diff": max(worst.values()), "pwm_max_abs_diff": max(worst[c] for c in pwm), "clock": rep["clock"],
            "span_s": rep["span_s"], "warnings": rep["warnings"]}, "PWM identical; the rest to ulog2csv's printed precision", ok)
    folder = os.path.join(DATA, "I-px4-real")
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    spec = {"vehicle": "px4_bluerov2", "source": {"files": PUBLIC_PX4, "format": "px4"}, "resample_hz": 10,
            "session_date": {"from": "recording"}, "frame": "frd",
            "body_velocity": {"columns": {"u": "vehicle_local_position.vx", "v": "vehicle_local_position.vy",
                                          "w": "vehicle_local_position.vz", "p": "vehicle_angular_velocity.xyz[0]",
                                          "q": "vehicle_angular_velocity.xyz[1]", "r": "vehicle_angular_velocity.xyz[2]"},
                              "linear_frame": "world"},
            "orientation": {"quaternion": {"w": "vehicle_attitude.q[0]", "x": "vehicle_attitude.q[1]",
                                           "y": "vehicle_attitude.q[2]", "z": "vehicle_attitude.q[3]"}},
            "wrench": {"thrusters": {"pwm_columns": pwm, "preset": "bluerov2_heavy", "voltage_column": "battery_status.voltage_v"}}}
    sp = os.path.join(folder, "spec.yaml")
    yaml.safe_dump(spec, open(sp, "w"), sort_keys=False)
    try:
        summ = di.ingest(di.load_ingest_spec(sp), os.path.join(folder, "out"))
        got = {"recordings": summ.get("n_recordings"), "sessions": summ.get("sessions"),
               "warnings": summ.get("warnings", [])[:4]}
        ok = bool(summ.get("n_recordings"))
    except di.IngestError as e:   # a plain refusal from the data checks is an answer too: recorded as it is
        got, ok = {"refused": str(e)[:400]}, "Traceback" not in str(e)
    record(F, "I-px4-real-ingest", "entry", "strathe ingest, real log", "px4: the real BlueROV2 log through ingestion",
           os.path.basename(PUBLIC_PX4), got, "converts, or refuses with a plain sentence naming the problem", ok)


def fast() -> int:
    os.makedirs(DATA, exist_ok=True)
    for step in (r_full, v_one_at_a_time, p_pairwise, k_field_kinds, d_damage, s_size, x_spec_rules, i_independent,
                 e_entry_points, c_check):
        print(f"--- {step.__name__} ({time.strftime('%H:%M')})", flush=True)
        try:
            step()
        except Exception as e:
            import traceback
            traceback.print_exc()
            record(F, f"crash-{step.__name__}", "harness", step.__name__, "the step crashed", "-",
                   {"error": f"{type(e).__name__}: {e}"[:300]}, "runs", False)
    return 0


# --- mutants --------------------------------------------------------------------------------------------------------
def mutants() -> int:
    """Faults in the readers; a quick set of round trips must fail under each."""
    import common
    import moos_reader
    import rosbag_reader
    import ulog_reader
    real_record = common.record
    real = {"grid": rosbag_reader.series_to_grid, "ulog": ulog_reader.read_log, "moos": moos_reader.read_log}

    def sign_flip(path, series, *a, **k):   # a frame sign flipped on one axis
        series = {n: (t, {c: (-np.asarray(x) if c.endswith(("vy", "DVL_V", "vel[1]")) else x) for c, x in v.items()})
                  for n, (t, v) in series.items()}
        return real["grid"](path, series, *a, **k)

    import ulog_reader as ur

    def time_unit(path, columns, resample_hz, *a, **k):
        from pyulog import ULog
        orig = ULog.__init__

        def init(self, *aa, **kk):
            orig(self, *aa, **kk)
            for d in self.data_list:
                d.data["timestamp"] = d.data["timestamp"] * 1000
        ULog.__init__ = init
        try:
            return real["ulog"](path, columns, resample_hz, *a, **k)
        finally:
            ULog.__init__ = orig

    def dropped_index(path, columns, *a, **k):   # q[1] read as q[0]
        cols = [c.replace("q[1]", "q[0]") if "attitude" in c else c for c in columns]
        df, rep = real["ulog"](path, cols, *a, **k)
        for c, c2 in zip(columns, cols):
            if c != c2:
                df[c] = df[c2]
        return df, rep

    faults = {"frame sign flipped": (rosbag_reader, "series_to_grid", sign_flip),
              "PX4 time unit wrong by 1000": (ur, "read_log", time_unit),
              "array index dropped": (ur, "read_log", dropped_index)}
    for label, (mod, name, fn) in faults.items():
        fails = []
        globals()["record"] = lambda *a, **k: fails.append(a[1]) if not a[8] else None
        setattr(mod, name, fn)
        try:
            recs = recordings("sample")[:1]
            for fmt in FORMATS:
                res = round_trip(fmt, recs, BASE, f"M-{fmt}")
                record(F, f"M-{fmt}", "mutant", label, "", "", res, "", _ok(res))
        except Exception as e:
            fails.append(f"crash: {e}")
        finally:
            setattr(mod, name, real["grid"] if name == "series_to_grid" else real["ulog"])
            globals()["record"] = real_record
        record(F, f"M-{label.replace(' ', '_')}", "mutant", label, f"deliberate fault: {label}", "round trips",
               {"rows_failed": fails}, "at least one round trip fails", bool(fails))
    return 0


# --- end to end: calibrations ---------------------------------------------------------------------------------------
def _cal(folder_ingested: str, out: str, extra: list, spec=None) -> tuple:
    shutil.rmtree(out, ignore_errors=True)
    rc, secs = strathe(["calibrate", "--data", folder_ingested, "--name", os.path.basename(out), "--out", out,
                        "--seed", "94", "--skip-identifiability", *(["--spec", spec] if spec else []), *extra],
                       out + ".log")
    return rc, secs


def _params(out: str) -> dict:
    return yaml.safe_load(open(os.path.join(out, "calibrated_parameters.yaml")))["parameters"]


def e2e(only=None) -> int:
    """Calibrate the BlueROV2 (all four days, 16 and 19 Feb held out) and URI (pond held out) from each format's
    ingest and from the CSV route's, with the same seed: each completes. The routes aren't expected to land on the same
    parameters: on this data the fit's path is chaotic (#93: recordings that differ by 3e-14 took the optimizer to
    different places). That the routes give the same calibration problem is objective_same()'s proof."""
    cases = [("bluerov", recordings("pwm"), ["--heldout-dates", "16-02-2024", "19-02-2024"], None),
             ("uri", [Rec(p, k) for k, p in enumerate(sorted(_glob(os.path.join(URI, "*.csv"))))],
              ["--heldout-dates", "27-03-2026", "--estimate-current", "--workers", "2"], URI_SPEC)]
    for name, recs, extra, spec in cases:
        root = os.path.join(DATA, f"E-{name}")
        ingested = {}
        for fmt in FORMATS:
            if only and f"E-{name}-{fmt}" not in only:
                continue
            res = round_trip(fmt, recs, BASE, f"E-{name}-{fmt}")
            if not _ok(res):   # no calibration from a conversion that didn't match
                record(F, f"E-{name}-{fmt}", "calibration", fmt, f"{name}: {fmt} round trip before calibrating", name,
                       res, "the same recordings", False)
                continue
            ingested[fmt] = os.path.join(DATA, f"E-{name}-{fmt}", fmt, "ingested")
            ingested["csv"] = os.path.join(DATA, f"E-{name}-{fmt}", "csv", "ingested")
        if not ingested:
            continue
        outs = {}
        for route, folder in ingested.items():
            outs[route] = os.path.join(root, f"cal_{route}")
            rc, secs = _cal(folder, outs[route], extra, spec)
            record(F, f"E-{name}-{route}-cal", "calibration", route, f"calibrate {name} from the {route} route", name,
                   {"exit": rc}, "completes", rc == 0, secs)
    return 0


def _route_folders(name: str, recs: list) -> dict:
    """{format: {route: ingested folder}} for each format and its CSV route, converting what isn't there yet."""
    out = {}
    for fmt in FORMATS:
        tag = f"E-{name}-{fmt}"
        folders = {r: os.path.join(DATA, tag, r, "ingested") for r in (fmt, "csv")}
        if not all(os.path.exists(os.path.join(f, "ingest_report.json")) for f in folders.values()):
            res = round_trip(fmt, recs, BASE, tag)
            if not _ok(res):
                record(F, tag, "calibration", fmt, f"{name}: {fmt} round trip", name, res, "the same recordings", False)
                continue
        out[fmt] = folders
    return out


def objective_same(only=None) -> int:
    """Each format's route gives the same calibration problem as the CSV route (#94): the training cost calibrate
    minimizes (the same split, axis scaling and torque convention) agrees to rounding at the starting description, a
    randomly perturbed start, and a calibrated point. Calibrating twice can't show this: on this data the fit's path is
    chaotic (#93), so two calibrations of recordings that differ by 3e-14 can end apart."""
    import data_split as ds
    from dynamics_model import CalibrationParameters
    from objective import DatasetConfig, ParallelJointResiduals, compute_zero_baseline_scales, load_run
    from optimizer import _initialize_x0, initial_parameters, load_vehicle_spec, parameter_bounds
    from run_calibration import spec_torque_reference_offset
    cases = [("bluerov", recordings("pwm"), ["16-02-2024", "19-02-2024"], os.path.join(CAL, "vehicle_spec.yaml")),
             ("uri", [Rec(p, k) for k, p in enumerate(sorted(_glob(os.path.join(URI, "*.csv"))))], ["27-03-2026"],
              URI_SPEC)]
    for name, recs, held, spec_path in cases:
        if only and not any(o.startswith(f"E-{name}") for o in only):
            continue
        t0 = time.time()
        spec = load_vehicle_spec(spec_path)
        p0 = initial_parameters(spec).to_vector()
        lb, ub = parameter_bounds(spec)
        points = {"start": p0, "perturbed start (seed 94)": _initialize_x0(p0, lb, ub, np.random.default_rng(94),
                                                                           "perturbed", 0.1)}
        cal = os.path.join(DATA, f"E-{name}", "cal_px4")
        if os.path.exists(os.path.join(cal, "calibrated_parameters.yaml")):
            points["calibrated (px4 route)"] = CalibrationParameters(**_params(cal)).to_vector()

        def costs(folder):
            sp = ds.split_by_date(ds.discover_csv_files(folder)[0], heldout_dates=held)
            runs = [load_run(f, DatasetConfig(ds._rate_hz(f), spec_torque_reference_offset(spec)))
                    for f in sp.train_files]
            sc = compute_zero_baseline_scales(runs)
            with ParallelJointResiduals(runs, spec.vehicle, 6) as fn:
                return len(runs), {k: float(np.sum(fn(x, runs, spec.vehicle, sc) ** 2)) for k, x in points.items()}
        for fmt, folders in _route_folders(name, recs).items():
            n_c, base = costs(folders["csv"])
            n_f, got = costs(folders[fmt])
            rel = max(abs(got[k] - base[k]) / base[k] for k in base)
            record(F, f"E-{name}-{fmt}", "calibration", fmt,
                   f"{name}: the calibration problem from {fmt} vs from the CSV route", name,
                   {"train_recordings": [n_c, n_f], "cost_csv": base, f"cost_{fmt}": got, "max_rel_cost_diff": rel},
                   "the same training recordings, and the same training cost at every point (relative diff < 1e-9)",
                   n_c == n_f and rel < 1e-9, time.time() - t0)
    return 0


def list_rows() -> None:
    print("fast: R V P K D S X I EP; mutants: M; objective: E-bluerov-<fmt> E-uri-<fmt>; e2e: E-<name>-<route>-cal")


ROWS_FAST = tuple(sorted(all_rows()))


def _e2e_rows() -> dict:
    return {f"E-{n}-{f}": None for n in ("bluerov", "uri") for f in FORMATS}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fast"
    if cmd == "list":
        list_rows()
        sys.exit(0)
    sys.exit({"fast": fast, "mutants": mutants, "objective": lambda: objective_same(sys.argv[2:])}
             .get(cmd, lambda: e2e(sys.argv[2:]))())
