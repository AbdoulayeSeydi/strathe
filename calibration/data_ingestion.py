#!/usr/bin/env python3
"""
data_ingestion.py -- convert a vehicle's own logs into the canonical run CSVs
the calibration pipeline reads (objective.REQUIRED_COLUMNS).

    python data_ingestion.py --spec my_vehicle_ingest.yaml --out ingested/
    (or: python strathe.py ingest --spec ... --out ...; ready-to-edit specs are in templates/)

The pipeline was built on one public dataset whose column names, units and
frames it assumed. A lab's logs will differ in all three. Instead of
touching the pipeline, an ingest spec (YAML) says where each quantity lives
in the lab's files and in what units/frame; this module converts, checks
and writes canonical CSVs:

    Fx Fy Fz Tx Ty Tz   body-frame force/torque (N, N*m), FRD
    Bu Bv Bw Bp Bq Br   body-frame velocity (m/s, rad/s), FRD
    qx qy qz qw         orientation, body FRD -> world NED
at a uniform sampling rate, named <vehicle>-<DD-MM-YYYY>-run<k>.csv so the
held-out-by-date split keeps working.

FRD = forward-right-down body frame with NED world (Fossen / SNAME, what the
physics model uses). ROS logs are usually FLU body / ENU world (REP-103);
set frame: flu and they are converted.

If a log has thruster PWM but no forces, give a `thrusters:` block and the
wrench is computed with thruster_model.py: the BlueROV2 Heavy preset, a
per-thruster `layout` (position, direction, reversed) for any other vehicle,
or your own allocation matrix file; the T200 curve or your own curve file.

ROS bags (ROS 1 .bag, ROS 2 bag folders with .db3 or .mcap) are read directly
with `source: {format: rosbag}`; columns are then `<topic>.<field>` and every
topic is put on one grid at resample_hz (rosbag_reader.py).

Every output comes with ingest_report.json: detected sampling rate and
jitter, resampling, time gaps, dropped rows, and sanity checks that catch the
classic silent errors (angular rates logged in deg/s but declared rad/s,
non-unit quaternions, a wrong frame). Problems that would silently corrupt a
calibration are errors; merely suspicious data is a warning.

Example spec (every key shown; most have defaults):

    vehicle: my_rov
    source: {files: "raw_logs/*.csv", delimiter: ","}
    time: {column: stamp, unit: s}          # or: {sampling_rate_hz: 10}
    resample_hz: 10                         # optional: resample onto a uniform grid
    session_date: {from: filename, pattern: "(\\d{4}-\\d{2}-\\d{2})", format: "%Y-%m-%d"}
                                            # or {from: constant, value: "2024-02-16"}
    frame: flu                              # body/world convention of ALL vectors below
    body_velocity:
      columns: {u: vx, v: vy, w: vz, p: wx, q: wy, r: wz}
      linear_unit: m/s                      # m/s | cm/s | mm/s | knots
      angular_unit: deg/s                   # rad/s | deg/s
    orientation:
      quaternion: {x: qx, y: qy, z: qz, w: qw}
      # or euler: {roll: roll, pitch: pitch, yaw: yaw, unit: deg}
    wrench:
      columns: {Fx: fx, Fy: fy, Fz: fz, Tx: tx, Ty: ty, Tz: tz}
      # or thrusters: {pwm_columns: [pwm1, ..., pwm8], preset: bluerov2_heavy, voltage: 15.0}
      #          or {pwm_columns: [...], layout: [{position_m: [x, y, z], direction: [dx, dy, dz],
      #                                            reversed: false}, ...], curve: t200, voltage: 16}
      #          or {pwm_columns: [...], allocation_csv: B.csv, directions: [1, -1, ...],
      #              curve_csv: my_curve.csv}

ROS bag variant (only the differences):

    source: {files: "bags/*.bag", format: rosbag}   # stamp: header (default) | receive
    resample_hz: 10                                  # required for bags; no time: block
    session_date: {from: recording}                  # the bag's own start date
    body_velocity:
      columns: {u: /mavros/local_position/velocity_body.twist.linear.x, ...}
    wrench:
      thrusters: {pwm_columns: ["/mavros/rc/out.channels[0]", ...], preset: bluerov2_heavy}

Templates for all of these are in templates/ (see templates/README.md).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, PrivateAttr, ValidationError, model_validator
from scipy.spatial.transform import Rotation

from objective import REQUIRED_COLUMNS

LINEAR_UNITS = {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001, "knots": 0.514444}
ANGULAR_UNITS = {"rad/s": 1.0, "deg/s": np.pi / 180.0}
ANGLE_UNITS = {"rad": 1.0, "deg": np.pi / 180.0}
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])                                # body vectors
ENU_TO_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # world vectors

CANONICAL_COLUMNS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bu", "Bv", "Bw", "Bp", "Bq", "Br", "qx", "qy", "qz", "qw"]
assert set(CANONICAL_COLUMNS) == set(REQUIRED_COLUMNS), "keep in step with objective.REQUIRED_COLUMNS"

# Beyond these, a declared unit is almost certainly wrong (e.g. deg/s logged as rad/s)
MAX_PLAUSIBLE_LINEAR_MPS = 5.0
MAX_PLAUSIBLE_ANGULAR_RADPS = 6.0
MAX_PLAUSIBLE_ORIENTATION_JUMP_DEG = 45.0  # per sample; ~450 deg/s at 10 Hz


class IngestError(ValueError):
    """The data would silently corrupt a calibration if used as-is."""


# ----------------------------------------------------------------- spec schema

class SourceSpec(BaseModel):
    files: str
    format: Literal["csv", "rosbag"] = "csv"
    delimiter: str = ","
    stamp: Literal["header", "receive"] = "header"   # rosbag: message header.stamp, or bag receive time
    message_types: list[str] = Field(default_factory=list)  # rosbag: .msg files for custom types in .db3 bags


class TimeSpec(BaseModel):
    column: Optional[str] = None
    unit: Literal["s", "ms", "us", "ns"] = "s"
    sampling_rate_hz: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _one(self):
        if (self.column is None) == (self.sampling_rate_hz is None):
            raise ValueError("time needs exactly one of: column (timestamps) or sampling_rate_hz")
        return self


class DateSpec(BaseModel):
    source: Literal["filename", "constant", "recording"] = Field(alias="from")
    pattern: str = r"(\d{2}-\d{2}-\d{4})"
    format: str = "%d-%m-%Y"
    value: Optional[str] = None

    @model_validator(mode="after")
    def _const(self):
        if self.source == "constant" and not self.value:
            raise ValueError("session_date from: constant needs a value")
        return self


class VelocitySpec(BaseModel):
    columns: dict[Literal["u", "v", "w", "p", "q", "r"], str]
    linear_unit: Literal["m/s", "cm/s", "mm/s", "knots"] = "m/s"
    angular_unit: Literal["rad/s", "deg/s"] = "rad/s"

    @model_validator(mode="after")
    def _all(self):
        missing = {"u", "v", "w", "p", "q", "r"} - set(self.columns)
        if missing:
            raise ValueError(f"body_velocity.columns is missing {sorted(missing)}")
        return self


class QuaternionSpec(BaseModel):
    x: str
    y: str
    z: str
    w: str


class EulerSpec(BaseModel):
    roll: str
    pitch: str
    yaw: str
    unit: Literal["rad", "deg"] = "rad"


class OrientationSpec(BaseModel):
    quaternion: Optional[QuaternionSpec] = None
    euler: Optional[EulerSpec] = None

    @model_validator(mode="after")
    def _one(self):
        if (self.quaternion is None) == (self.euler is None):
            raise ValueError("orientation needs exactly one of: quaternion or euler")
        return self


class ThrusterSpec(BaseModel):
    """pwm_columns plus a thruster_model.thruster_block() block (checked when the spec is loaded)."""
    pwm_columns: list[str] = Field(min_length=1)
    model_config = {"extra": "allow"}

    def block(self) -> dict:
        return {k: v for k, v in (self.model_extra or {}).items()}


class WrenchSpec(BaseModel):
    columns: Optional[dict[Literal["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], str]] = None
    thrusters: Optional[ThrusterSpec] = None

    @model_validator(mode="after")
    def _one(self):
        if (self.columns is None) == (self.thrusters is None):
            raise ValueError("wrench needs exactly one of: columns (forces logged) or thrusters (PWM logged)")
        if self.columns is not None:
            missing = {"Fx", "Fy", "Fz", "Tx", "Ty", "Tz"} - set(self.columns)
            if missing:
                raise ValueError(f"wrench.columns is missing {sorted(missing)}")
        return self


class IngestSpec(BaseModel):
    vehicle: str = Field(pattern=r"^[A-Za-z0-9_]+$")
    source: SourceSpec
    time: Optional[TimeSpec] = None
    resample_hz: Optional[float] = Field(default=None, gt=0)
    session_date: DateSpec
    frame: Literal["frd", "flu"] = "frd"
    body_velocity: VelocitySpec
    orientation: OrientationSpec
    wrench: WrenchSpec

    model_config = {"populate_by_name": True}
    _thruster_cfg: object = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _time_source(self):
        if self.source.format == "csv" and self.time is None:
            raise ValueError("a CSV source needs time: {column: ..., unit: ...} or time: {sampling_rate_hz: ...}")
        if self.source.format == "rosbag":
            if self.time is not None:
                raise ValueError("a rosbag source takes its time from the bag; remove the time: block")
            if self.resample_hz is None:
                raise ValueError("a rosbag source needs resample_hz: topics arrive at different rates and are "
                                 "put on one grid")
        if self.source.format == "csv" and self.source.message_types:
            raise ValueError("message_types only applies to rosbag sources")
        return self

    def columns(self) -> list:
        """Every source column/field the spec reads."""
        cols = list(self.body_velocity.columns.values())
        o = self.orientation
        cols += ([getattr(o.quaternion, k) for k in "xyzw"] if o.quaternion is not None
                 else [getattr(o.euler, k) for k in ("roll", "pitch", "yaw")])
        w = self.wrench
        cols += list(w.columns.values()) if w.columns is not None else list(w.thrusters.pwm_columns)
        return cols


def load_ingest_spec(path: str) -> IngestSpec:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        spec = IngestSpec(**raw)
    except ValidationError as e:
        raise IngestError(f"invalid ingest spec {path!r}:\n{e}") from e
    base = os.path.dirname(os.path.abspath(path))
    spec.source.files = os.path.join(base, spec.source.files) if not os.path.isabs(spec.source.files) else spec.source.files
    spec.source.message_types = [p if os.path.isabs(p) else os.path.join(base, p) for p in spec.source.message_types]
    th = spec.wrench.thrusters
    if th is not None:
        import thruster_model as tm
        try:
            cfg = tm.thruster_block(th.block(), base)
        except (ValueError, OSError) as e:
            raise IngestError(f"invalid thrusters block in {path!r}: {e}") from e
        if cfg.n_thrusters != len(th.pwm_columns):
            raise IngestError(f"{len(th.pwm_columns)} PWM columns but the thruster config has "
                              f"{cfg.n_thrusters} thrusters")
        spec._thruster_cfg = cfg
    return spec


# ----------------------------------------------------------------- conversion

def _col(df, name, path):
    if name not in df.columns:
        raise IngestError(f"{path}: column {name!r} not found (have: {list(df.columns)[:20]}...)")
    return df[name].to_numpy(dtype=float)


def _thruster_config(spec: IngestSpec):
    if spec._thruster_cfg is None:  # spec built in code rather than by load_ingest_spec
        import thruster_model as tm
        spec._thruster_cfg = tm.thruster_block(spec.wrench.thrusters.block())
    return spec._thruster_cfg


def _session_date(spec: IngestSpec, path: str, start_unix_s: Optional[float] = None) -> str:
    ds = spec.session_date
    if ds.source == "recording":
        if start_unix_s is None or start_unix_s < 1e8:
            raise IngestError(f"{path}: session_date from: recording needs real clock timestamps (a ROS bag, or a "
                              f"time column in Unix seconds); use from: filename or from: constant instead")
        return dt.datetime.fromtimestamp(start_unix_s, dt.timezone.utc).strftime("%d-%m-%Y")
    if ds.source == "constant":
        raw = ds.value
    else:
        m = re.search(ds.pattern, os.path.basename(path))
        if not m:
            raise IngestError(f"{path}: session_date pattern {ds.pattern!r} doesn't match the filename")
        raw = m.group(1)
    fmt = ds.format if ds.source == "filename" else "%Y-%m-%d"
    try:
        return dt.datetime.strptime(raw, fmt).strftime("%d-%m-%Y")
    except ValueError as e:
        raise IngestError(f"{path}: session date {raw!r} doesn't match format {fmt!r}") from e


def _read_source(spec: IngestSpec, path: str) -> tuple:
    """(DataFrame, report, time column or None, time unit factor, already on a uniform grid)."""
    if spec.source.format == "csv":
        df = pd.read_csv(path, sep=spec.source.delimiter)
        return df, {"file": path, "rows_in": int(len(df)), "warnings": []}
    import rosbag_reader as rb
    o = spec.orientation
    quat = [[getattr(o.quaternion, k) for k in "xyzw"]] if o.quaternion is not None else []
    hold = spec.wrench.thrusters.pwm_columns if spec.wrench.thrusters is not None else []
    try:
        df, bag_rep = rb.read_bag(path, spec.columns(), spec.resample_hz, spec.source.stamp, hold, quat,
                                  spec.source.message_types)
    except rb.BagError as e:
        raise IngestError(str(e)) from e
    report = {"file": path, "rows_in": int(len(df)), "warnings": bag_rep.pop("warnings"), "bag": bag_rep}
    return df, report


def convert_file(spec: IngestSpec, path: str) -> tuple:
    """Returns (canonical DataFrame, per-file report dict)."""
    df, report = _read_source(spec, path)
    n = len(df)
    if n < 10:
        raise IngestError(f"{path}: only {n} rows")
    start_unix_s = None

    # time base
    if spec.source.format == "rosbag":
        import rosbag_reader as rb
        t = df[rb.TIME_COLUMN].to_numpy(dtype=float)
        rate = spec.resample_hz
        start_unix_s = report["bag"]["recording_start_unix_s"]
        report["resampled_to_hz"] = rate
    elif spec.time.column is not None:
        t = _col(df, spec.time.column, path) * TIME_UNITS[spec.time.unit]
        dts = np.diff(t)
        if np.any(dts <= 0):
            raise IngestError(f"{path}: timestamps are not strictly increasing "
                              f"({int(np.sum(dts <= 0))} non-positive steps) -- sort or de-duplicate first")
        med = float(np.median(dts))
        report["detected_sampling_rate_hz"] = round(1.0 / med, 4)
        report["timestamp_jitter_pct"] = round(100 * float(np.std(dts) / med), 3)
        gaps = np.where(dts > 2.5 * med)[0]
        report["time_gaps"] = [{"at_s": round(float(t[i] - t[0]), 3), "length_s": round(float(dts[i]), 3)}
                               for i in gaps[:20]]
        if len(gaps):
            report["warnings"].append(f"{len(gaps)} time gap(s) longer than 2.5 samples; open-loop replay "
                                      f"will integrate across them")
        if spec.resample_hz is None and report["timestamp_jitter_pct"] > 5:
            raise IngestError(f"{path}: sample spacing varies by {report['timestamp_jitter_pct']}% -- the "
                              f"pipeline assumes a uniform rate; set resample_hz in the spec")
        rate = spec.resample_hz or 1.0 / med
        start_unix_s = float(t[0])
    else:
        rate = spec.time.sampling_rate_hz
        t = np.arange(n) / rate
        report["declared_sampling_rate_hz"] = rate

    # quantities, converted to SI
    vc = spec.body_velocity.columns
    lin = np.stack([_col(df, vc[k], path) for k in "uvw"], 1) * LINEAR_UNITS[spec.body_velocity.linear_unit]
    ang = np.stack([_col(df, vc[k], path) for k in "pqr"], 1) * ANGULAR_UNITS[spec.body_velocity.angular_unit]
    o = spec.orientation
    if o.quaternion is not None:
        q = np.stack([_col(df, getattr(o.quaternion, k), path) for k in "xyzw"], 1)
        norms = np.linalg.norm(q, axis=1)
        # bags: checked on the raw messages (the grid values were slerped, so are unit length by construction)
        checked = np.array(report["bag"]["quaternion_norm_range"]) if "quaternion_norm_range" in report.get("bag", {}) \
            else norms
        if np.any(np.abs(checked - 1) > 0.05):
            raise IngestError(f"{path}: quaternion norms range {checked.min():.3f}-{checked.max():.3f}; expected 1 "
                              f"(wrong columns, or a w-first/w-last mix-up?)")
        R = Rotation.from_quat(q / norms[:, None]).as_matrix()
    else:
        eu = np.stack([_col(df, getattr(o.euler, k), path) for k in ("roll", "pitch", "yaw")], 1)
        R = Rotation.from_euler("xyz", eu * ANGLE_UNITS[o.euler.unit]).as_matrix()
    if spec.wrench.columns is not None:
        wc = spec.wrench.columns
        wrench = np.stack([_col(df, wc[k], path) for k in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")], 1)
        report["wrench_source"] = "logged forces"
    else:
        th = spec.wrench.thrusters
        cfg = _thruster_config(spec)
        pwm = np.stack([_col(df, c, path) for c in th.pwm_columns], 1)
        if np.nanmin(pwm) < 900 or np.nanmax(pwm) > 2100:
            raise IngestError(f"{path}: PWM values span {np.nanmin(pwm):.0f}-{np.nanmax(pwm):.0f}; expected ESC "
                              f"microseconds (~1100-1900)")
        wrench = cfg.wrench(pwm)
        report["wrench_source"] = f"computed from PWM ({cfg.label}, curve: {cfg.curve.label})"

    # frame conversion to FRD body / NED world
    if spec.frame == "flu":
        lin, ang = lin @ FLU_TO_FRD, ang @ FLU_TO_FRD
        if spec.wrench.columns is not None:  # PWM-derived wrenches are already FRD
            wrench = np.hstack([wrench[:, :3] @ FLU_TO_FRD, wrench[:, 3:] @ FLU_TO_FRD])
        R = ENU_TO_NED @ R @ FLU_TO_FRD

    data = np.hstack([wrench, lin, ang, Rotation.from_matrix(R).as_quat()])
    finite = np.all(np.isfinite(data), axis=1)
    if not finite.all():
        report["warnings"].append(f"dropped {int((~finite).sum())} row(s) with NaN/inf")
        if (~finite).sum() > 0.05 * n:
            raise IngestError(f"{path}: {int((~finite).sum())} of {n} rows have NaN/inf in required quantities")
        data, t = data[finite], t[finite]

    if spec.resample_hz is not None and spec.source.format == "csv" and spec.time.column is not None:
        # t[0] + k / rate rather than np.arange(t[0], ...), which drifts at Unix-epoch magnitudes
        grid = t[0] + np.arange(int(np.ceil((t[-1] - t[0]) * spec.resample_hz - 1e-9))) / spec.resample_hz
        cols = [np.interp(grid, t, data[:, j]) for j in range(12)]
        from rosbag_reader import slerp  # orientation: spherical interpolation, valid across sudden jumps
        data = np.hstack([np.stack(cols, 1), slerp(t, data[:, 12:16], grid)])
        report["resampled_to_hz"] = spec.resample_hz

    out = pd.DataFrame(data, columns=CANONICAL_COLUMNS)
    Rq = Rotation.from_quat(out[["qx", "qy", "qz", "qw"]].values)
    jumps = int(np.sum(np.degrees((Rq[1:] * Rq[:-1].inv()).magnitude()) > MAX_PLAUSIBLE_ORIENTATION_JUMP_DEG))
    if jumps:
        report["warnings"].append(f"orientation jumps by more than {MAX_PLAUSIBLE_ORIENTATION_JUMP_DEG:g} deg between "
                                  f"consecutive samples {jumps} time(s): likely sensor glitches or heading resets; "
                                  f"they add error to the calibration")

    # unit sanity: the silent errors that ruin a calibration
    lin_max = float(np.abs(out[["Bu", "Bv", "Bw"]].values).max())
    ang_max = float(np.abs(out[["Bp", "Bq", "Br"]].values).max())
    report["max_abs_linear_velocity_mps"] = round(lin_max, 4)
    report["max_abs_angular_velocity_radps"] = round(ang_max, 4)
    if ang_max > MAX_PLAUSIBLE_ANGULAR_RADPS:
        raise IngestError(f"{path}: angular rates reach {ang_max:.1f} rad/s after conversion -- implausible for "
                          f"an ROV; were they logged in deg/s while the spec says {spec.body_velocity.angular_unit}?")
    if lin_max > MAX_PLAUSIBLE_LINEAR_MPS:
        raise IngestError(f"{path}: linear speeds reach {lin_max:.1f} m/s after conversion -- check linear_unit")
    if ang_max < 1e-4 or lin_max < 1e-4:
        report["warnings"].append("velocities are essentially zero throughout -- nothing to calibrate from")
    report["rows_out"] = int(len(out))
    report["sampling_rate_hz"] = float(rate)
    report["session_date"] = _session_date(spec, path, start_unix_s)
    return out, report


def torque_reference_for(spec: IngestSpec) -> Optional[list]:
    """The torque_reference_offset_m calibration should apply to these files' torques, when the
    ingest spec determines it: the BlueROV2 preset reproduces the reference dataset's torque point
    (8.5 cm below the thrusters); a layout or allocation file gives torques about the point its
    positions are measured from (offset 0). Logged forces: unknown here (None = vehicle_spec.yaml)."""
    if spec.wrench.thrusters is None:
        return None
    return [float(x) for x in _thruster_config(spec).torque_reference_offset_m]


def ingest(spec: IngestSpec, out_dir: str) -> dict:
    if spec.source.format == "rosbag":
        import rosbag_reader as rb
        files = rb.bag_paths(spec.source.files)
    else:
        files = sorted(glob.glob(spec.source.files))
    if not files:
        raise IngestError(f"no files match {spec.source.files!r}")
    os.makedirs(out_dir, exist_ok=True)
    reports, rates, counters = [], [], {}
    for path in files:
        df, rep = convert_file(spec, path)
        date = rep["session_date"]
        counters[date] = counters.get(date, 0) + 1
        name = f"{spec.vehicle}-{date}-run{counters[date]}.csv"
        df.to_csv(os.path.join(out_dir, name), index=False)
        rep["output"] = name
        reports.append(rep)
        rates.append(rep["sampling_rate_hz"])
    if max(rates) - min(rates) > 0.01 * min(rates):
        raise IngestError(f"files have different sampling rates ({sorted(set(round(r, 3) for r in rates))} Hz); "
                          f"set resample_hz so every run shares one rate")
    summary = {"vehicle": spec.vehicle, "n_files": len(files), "sampling_rate_hz": round(rates[0], 4),
               "sessions": {d: c for d, c in sorted(counters.items())},
               "torque_reference_offset_m": torque_reference_for(spec), "files": reports,
               "next_step": f"python strathe.py calibrate --data {out_dir} --name {spec.vehicle}"}
    if spec.wrench.thrusters is not None:
        import thruster_model as tm
        cfg = _thruster_config(spec)
        summary["thrusters"] = tm.describe(cfg)
        unreach = cfg.unreachable_axes()
        if unreach:
            summary.setdefault("warnings", []).append(
                f"your thrusters can't push: {', '.join(unreach)}. That's normal for some vehicles (a 6-thruster "
                f"BlueROV2 can't pitch on command); those axes are then only moved by the water and buoyancy, so "
                f"expect the calibration to learn little about them")
    with open(os.path.join(out_dir, "ingest_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        s = ingest(load_ingest_spec(args.spec), args.out)
    except IngestError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    n_warn = sum(len(f["warnings"]) for f in s["files"]) + len(s.get("warnings", []))
    if s.get("thrusters"):
        print(f"Thrusters: {s['thrusters']}")
    for w in s.get("warnings", []):
        print(f"WARNING: {w}")
    print(f"Ingested {s['n_files']} file(s) at {s['sampling_rate_hz']} Hz across {len(s['sessions'])} session date(s) "
          f"-> {args.out} ({n_warn} warning(s); see ingest_report.json)")
    print(f"Next: {s['next_step']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
