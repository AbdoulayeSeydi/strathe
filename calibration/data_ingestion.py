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

A vehicle's own sensors (PROJECT_STATUS.md #96, SENSORS.md): body velocity is one
sensor, not a requirement. With no DVL, `body_velocity` gives the gyro alone (or
is left out, and the rates come from the orientation), and `depth:`, `position:`
(use: calibrate; a USBL/LBL/mocap fix or GPS latlon) and `accelerometer:` add
columns pos_<name>_n/_e/_d, depth_<name> and acc_x/_y/_z. Their readings are
placed at the samples where they happened (sensor_ingest.place_on_grid), never
interpolated, and each is checked for its own silent errors.

FRD = forward-right-down body frame with NED world (Fossen / SNAME, what the
physics model uses). ROS logs are usually FLU body / ENU world (REP-103);
set frame: flu and they are converted.

If a log has thruster PWM but no forces, give a `thrusters:` block and the
wrench is computed with thruster_model.py: the BlueROV2 Heavy preset, a
per-thruster `layout` (position, direction, reversed) for any other vehicle,
or your own allocation matrix file; the T200 curve or your own curve file.

ROS bags (ROS 1 .bag, ROS 2 bag folders with .db3 or .mcap) are read directly
with `source: {format: rosbag}`; columns are then `<topic>.<field>` and every
topic is put on one grid at resample_hz (rosbag_reader.py). ArduPilot (format:
ardupilot), PX4 .ulg (px4), MOOS-IvP .alog (moos) and LCM logs (lcm, with
lcm_types:) are read the same way (LOG_FORMATS.md, #94).

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
      linear_mount_rpy_deg: [180, 0, 0]     # optional: a DVL mounted rotated (roll pitch yaw, deg; also angular_)
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
      #   normalized commands instead of PWM: add command: normalized, command_range: [-1, 1], and a
      #   curve_csv in those units (command,thrust_n), or the built-in T200 curve (range -> 1100..1900 us)

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
LENGTH_UNITS_POS = {"m": 1.0, "cm": 0.01, "mm": 0.001}
ANGULAR_UNITS = {"rad/s": 1.0, "deg/s": np.pi / 180.0}
ANGLE_UNITS = {"rad": 1.0, "deg": np.pi / 180.0}
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])                                # body vectors
ENU_TO_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # world vectors

CANONICAL_COLUMNS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bu", "Bv", "Bw", "Bp", "Bq", "Br", "qx", "qy", "qz", "qw"]


def reference_run_in_frd(df: pd.DataFrame) -> pd.DataFrame:
    """A run of the reference oceansystemslab dataset, whose numbers are ROS FLU body / ENU world (z up;
    PROJECT_STATUS.md #62), converted to FRD/NED exactly as `frame: flu` ingestion converts a lab's
    ROS logs. Other columns (pwm*, positions) are kept as they are."""
    out = df.copy()
    for c in ("Bv", "Bw", "Bq", "Br", "Fy", "Fz", "Ty", "Tz"):
        out[c] = -df[c]
    R = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].values).as_matrix()
    out[["qx", "qy", "qz", "qw"]] = Rotation.from_matrix(ENU_TO_NED @ R @ FLU_TO_FRD).as_quat()
    return out
assert set(CANONICAL_COLUMNS) == set(REQUIRED_COLUMNS), "keep in step with objective.REQUIRED_COLUMNS"

# Beyond these, a declared unit is almost certainly wrong (e.g. deg/s logged as rad/s)
MAX_PLAUSIBLE_LINEAR_MPS = 5.0
MAX_PLAUSIBLE_ANGULAR_RADPS = 6.0
MAX_PLAUSIBLE_ORIENTATION_JUMP_DEG = 45.0  # per sample; ~450 deg/s at 10 Hz


class IngestError(ValueError):
    """The data would silently corrupt a calibration if used as-is."""


# ----------------------------------------------------------------- spec schema

# Log formats read by a module's read_log() and log_paths(), put on one grid at resample_hz (#94: px4, moos, lcm)
LOG_FORMATS = {"rosbag": "rosbag_reader", "ardupilot": "ardupilot_reader", "px4": "ulog_reader",
               "moos": "moos_reader", "lcm": "lcm_reader"}

class SourceSpec(BaseModel):
    files: str
    format: Literal["csv", "rosbag", "ardupilot", "px4", "moos", "lcm"] = "csv"
    delimiter: str = ","
    stamp: Literal["header", "receive"] = "header"   # rosbag: header.stamp or receive time; lcm: utime or receive
    message_types: list[str] = Field(default_factory=list)  # rosbag: .msg files for custom types in .db3 bags
    lcm_types: list[str] = Field(default_factory=list)      # lcm: your .lcm message definitions (globs) (#94)


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
    """Body velocity: a DVL (or motion capture) for u, v, w, and the gyro for p, q, r. Either half may be left out
    (PROJECT_STATUS.md #96): a vehicle with no DVL gives p, q, r only; with no gyro logged, the rates are derived from
    the orientation (with a warning)."""
    columns: dict[Literal["u", "v", "w", "p", "q", "r"], str]
    # interpolate (the default, as before): u, v, w are put on the grid like any column, so a DVL's dropouts are
    # bridged. keep: each DVL reading is placed where it happened and the rest are NaN (sensor_ingest.place_on_grid),
    # so calibration sees only real readings
    linear_gaps: Literal["interpolate", "keep"] = "interpolate"
    linear_unit: Literal["m/s", "cm/s", "mm/s", "knots"] = "m/s"
    angular_unit: Literal["rad/s", "deg/s"] = "rad/s"
    # "world": u, v, w are world-frame (NED, or ENU with frame: flu) velocities, as ArduPilot's EKF and
    # most odometry-in-map-frame logs give; they are rotated into the body frame with the orientation
    linear_frame: Literal["body", "world"] = "body"
    # a sensor mounted rotated in the vehicle (#91: URI's DVL is rolled 180 deg): its roll, pitch, yaw in degrees, in
    # the same axes as the spec's frame; readings are rotated into the vehicle's axes. Body-frame velocity only.
    linear_mount_rpy_deg: Optional[list[float]] = None
    angular_mount_rpy_deg: Optional[list[float]] = None

    @model_validator(mode="after")
    def _all(self):
        for half, what in (("uvw", "u, v, w (the DVL)"), ("pqr", "p, q, r (the gyro)")):
            have = set(half) & set(self.columns)
            if have and have != set(half):
                raise ValueError(f"body_velocity.columns has {sorted(have)} of {what}: give all three, or leave all "
                                 f"three out (no DVL: leave out u, v, w; no gyro logged: leave out p, q, r)")
        if not self.columns:
            raise ValueError("body_velocity.columns is empty: leave the block out if the logs have no DVL and no gyro")
        if self.linear_gaps == "keep" and not self.has_linear:
            raise ValueError("body_velocity.linear_gaps is about the DVL's readings (u, v, w), which aren't given")
        for name in ("linear_mount_rpy_deg", "angular_mount_rpy_deg"):
            v = getattr(self, name)
            if v is not None and len(v) != 3:
                raise ValueError(f"body_velocity.{name} is three angles in degrees: roll pitch yaw, got {v}")
        if self.linear_mount_rpy_deg is not None and self.linear_frame == "world":
            raise ValueError("body_velocity.linear_mount_rpy_deg is for a body-frame sensor (a DVL); "
                             "world-frame velocity (linear_frame: world) has no mount to undo")
        return self

    @property
    def has_linear(self) -> bool:
        return "u" in self.columns

    @property
    def has_angular(self) -> bool:
        return "p" in self.columns


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
    """pwm_columns (one logged command column per thruster: ESC PWM in microseconds, or normalized commands with
    command: normalized and command_range, PROJECT_STATUS.md #67) plus a thruster_model.thruster_block() block
    (checked when the spec is loaded).
    voltage_column: the battery voltage logged with each sample; ingestion scales each sample's thrust to its own
    voltage (the built-in T200 curve, or your own T200 table given with curve_voltage; PROJECT_STATUS.md #63, #69).
    voltage:, if also given, is then only the nominal voltage the
    simulators use when this spec is passed as --thrusters (they have no logged voltage)."""
    pwm_columns: list[str] = Field(min_length=1)
    voltage_column: Optional[str] = None
    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def _voltage(self):
        extra = self.model_extra or {}
        if self.voltage_column is not None:
            if extra.get("curve") not in (None, "t200"):
                raise ValueError(f"unknown curve {extra.get('curve')!r}: use t200 or curve_csv")
            if extra.get("curve_csv") is not None and extra.get("curve_voltage") is None:
                raise ValueError("voltage_column with your own curve_csv needs curve_voltage: the battery voltage "
                                 "your table was measured at. The scaling uses Blue Robotics' published T200 data, "
                                 "so it's for T200 thrusters")
        return self

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


class ExcludeRule(BaseModel):
    """Drop samples where a logged column is above and/or below a value: a tether pulled taut (a tension sensor),
    contact with the bottom, a manual-override flag. The recording is split at the dropped stretches
    (PROJECT_STATUS.md #63)."""
    column: str
    above: Optional[float] = None
    below: Optional[float] = None

    @model_validator(mode="after")
    def _one(self):
        if self.above is None and self.below is None:
            raise ValueError(f"exclude_samples rule for {self.column!r} needs above: and/or below:")
        return self


SENSOR_NAME = r"^[a-z][a-z0-9_]*$"


class LatLonSpec(BaseModel):
    lat: str
    lon: str
    unit: Literal["deg", "rad"] = "deg"


class PositionSpec(BaseModel):
    """World positions: motion capture, a USBL or LBL fix, GPS (latlon:), in the spec's world frame (NED, or ENU with
    frame: flu). use: check (the default, as before): only to check the orientation against the position track
    (PROJECT_STATUS.md #63). use: calibrate: a sensor the calibration fits to (#96); its fixes are placed where they
    happened, never interpolated."""
    x: Optional[str] = None
    y: Optional[str] = None
    z: Optional[str] = None
    latlon: Optional[LatLonSpec] = None
    unit: Literal["m", "cm", "mm"] = "m"
    use: Literal["check", "calibrate"] = "check"
    source: Literal["usbl", "gps", "lbl", "mocap", "odometry", "other"] = "other"
    name: Optional[str] = Field(default=None, pattern=SENSOR_NAME)
    drop_when: list[ExcludeRule] = []   # e.g. GPS has no fix underwater: {column: <depth>, above: 0.5}
    # the sensor reports this long after the vehicle was there (USBL latency). Given (0 included), calibration takes
    # it as said; left out, calibration looks for a delay and takes off one the fixes clearly show (SENSORS.md)
    delay_s: Optional[float] = None
    max_speed_mps: float = Field(default=5.0, gt=0)   # fixes the vehicle couldn't reach at this speed are dropped
    # how far apart two honest fixes can be from noise alone, per axis (m); default by source
    # (sensor_ingest.FIX_NOISE_M). It sets the outlier check's tolerance, not the calibration's weighting
    noise_m: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _where(self):
        if self.latlon is not None and (self.x or self.y):
            raise ValueError("position takes x, y (and z) in metres, or latlon, not both")
        if self.latlon is not None and self.z:
            raise ValueError("position with latlon takes no z: a GPS altitude says nothing underwater; give the "
                             "vehicle's depth sensor as a depth: block")
        if self.latlon is None and not (self.x and self.y):
            raise ValueError("position needs x and y (and z if the system measures it), or latlon: {lat, lon}")
        if self.use == "calibrate" and self.source == "odometry":
            raise ValueError("position source: odometry can't be used to calibrate: a filtered navigation solution "
                             "usually fuses the DVL and the vehicle's own model, so it isn't an independent sensor. Use "
                             "the raw fixes (USBL, GPS, LBL), or keep it for use: check")
        if self.delay_s is not None and not np.isfinite(self.delay_s):
            raise ValueError("position.delay_s must be a number of seconds")
        if self.name is None:
            self.name = self.source if self.source != "other" else "pos"
        return self

    @property
    def axes(self) -> str:
        """ned: the fix has a vertical; ne: horizontal only (GPS, or a 2-D system)."""
        return "ned" if self.z else "ne"


class PressureSpec(BaseModel):
    column: str
    unit: Literal["Pa", "hPa", "kPa", "mbar", "bar", "psi"]
    water: Union[Literal["fresh", "salt"], float] = "fresh"   # or a density in kg/m^3
    surface: Union[Literal["auto"], float] = "auto"         # surface pressure, in the same unit


class DepthSpec(BaseModel):
    """A depth sensor (#96): a depth reading (column, positive down unless positive: up) or a pressure (pressure:).
    Readings are placed where they happened, never interpolated."""
    column: Optional[str] = None
    unit: Literal["m", "cm", "mm"] = "m"
    positive: Literal["down", "up"] = "down"
    pressure: Optional[PressureSpec] = None
    name: str = Field(default="main", pattern=SENSOR_NAME)   # its column: depth_<name>
    drop_when: list[ExcludeRule] = []
    delay_s: Optional[float] = None   # as position.delay_s: given (0 included), taken as said; left out, looked for

    @model_validator(mode="after")
    def _one(self):
        if (self.column is None) == (self.pressure is None):
            raise ValueError("depth needs exactly one of: column (a depth reading) or pressure: {column, unit}")
        if self.delay_s is not None and not np.isfinite(self.delay_s):
            raise ValueError("depth.delay_s must be a number of seconds")
        return self

    @property
    def source_column(self) -> str:
        return self.column if self.column is not None else self.pressure.column


class AccelerometerSpec(BaseModel):
    """An accelerometer's specific force (#96), in the spec's body frame: what a raw IMU reports, gravity included."""
    x: str
    y: str
    z: str
    unit: Literal["m/s^2", "g"] = "m/s^2"
    mount_rpy_deg: Optional[list[float]] = None
    drop_when: list[ExcludeRule] = []
    delay_s: float = 0.0

    @model_validator(mode="after")
    def _mount(self):
        if self.mount_rpy_deg is not None and len(self.mount_rpy_deg) != 3:
            raise ValueError(f"accelerometer.mount_rpy_deg is three angles in degrees: roll pitch yaw, got "
                             f"{self.mount_rpy_deg}")
        if not np.isfinite(self.delay_s):
            raise ValueError("accelerometer.delay_s must be a number of seconds")
        return self


CHECKS = ("force_direction", "orientation_vs_position", "gyro_vs_orientation", "accelerometer", "depth_vs_position",
          "heading_vs_position")


class IngestSpec(BaseModel):
    vehicle: str = Field(pattern=r"^[A-Za-z0-9_]+$")
    source: SourceSpec
    time: Optional[TimeSpec] = None
    resample_hz: Optional[float] = Field(default=None, gt=0)
    session_date: DateSpec
    frame: Literal["frd", "flu"] = "frd"
    body_velocity: Optional[VelocitySpec] = None   # no DVL and no gyro logged: leave it out (#96)
    orientation: OrientationSpec
    wrench: WrenchSpec
    position: Union[PositionSpec, list[PositionSpec], None] = None
    depth: Union[DepthSpec, list[DepthSpec], None] = None
    accelerometer: Optional[AccelerometerSpec] = None
    skip_checks: list[Literal["force_direction", "orientation_vs_position", "gyro_vs_orientation", "accelerometer",
                              "depth_vs_position", "heading_vs_position"]] = []  # escape hatch, says so
    exclude_samples: list[ExcludeRule] = []
    min_segment_s: float = Field(default=10.0, gt=0)  # with exclude_samples: shorter pieces are discarded

    model_config = {"populate_by_name": True}
    _thruster_cfg: object = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _time_source(self):
        if self.source.format == "csv" and self.time is None:
            raise ValueError("a CSV source needs time: {column: ..., unit: ...} or time: {sampling_rate_hz: ...}")
        if self.source.format in LOG_FORMATS:
            what = "bag" if self.source.format == "rosbag" else "log"
            if self.time is not None:
                raise ValueError(f"a {self.source.format} source takes its time from the {what}; remove the time: block")
            if self.resample_hz is None:
                raise ValueError(f"a {self.source.format} source needs resample_hz: its messages arrive at different "
                                 f"rates and are put on one grid")
        if self.source.format != "rosbag" and self.source.message_types:
            raise ValueError("message_types only applies to rosbag sources")
        if self.source.format == "lcm" and not self.source.lcm_types:
            raise ValueError("an lcm source needs lcm_types: your message definitions (.lcm files), e.g. "
                             "lcm_types: [my_types/*.lcm]")
        if self.source.format != "lcm" and self.source.lcm_types:
            raise ValueError("lcm_types only applies to lcm sources")
        if "stamp" in self.source.model_fields_set and self.source.format not in ("rosbag", "lcm"):
            raise ValueError(f"stamp only applies to rosbag and lcm sources (a {self.source.format} log has one clock)")
        names = [p.name for p in self.positions() if p.use == "calibrate"] + [d.name for d in self.depths()]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"two sensors are named {dup}: give each position and depth sensor its own name:")
        return self

    def positions(self) -> list:
        return [] if self.position is None else self.position if isinstance(self.position, list) else [self.position]

    def depths(self) -> list:
        return [] if self.depth is None else self.depth if isinstance(self.depth, list) else [self.depth]

    def sensor_blocks(self) -> list:
        """The sensors read as readings of their own (not interpolated): calibrating positions, depths, the
        accelerometer, and a DVL whose gaps are kept."""
        return ([p for p in self.positions() if p.use == "calibrate"] + self.depths()
                + ([self.accelerometer] if self.accelerometer is not None else []))

    def _block_columns(self, b) -> list:
        if isinstance(b, PositionSpec):
            return [b.latlon.lat, b.latlon.lon] + ([b.z] if b.z else []) if b.latlon is not None else \
                [c for c in (b.x, b.y, b.z) if c]
        if isinstance(b, DepthSpec):
            return [b.source_column]
        return [b.x, b.y, b.z]

    def native_columns(self) -> list:
        """Columns read as each reading at its own time (rosbag_reader.series_to_grid native_columns)."""
        cols = [c for b in self.sensor_blocks() for c in self._block_columns(b)]
        v = self.body_velocity
        if v is not None and v.has_linear and v.linear_gaps == "keep":
            cols += [v.columns[k] for k in "uvw"]
        return list(dict.fromkeys(cols))

    def span_exempt_columns(self) -> list:
        """What the sensors' drop_when rules read: their last value, not limiting the grid's span."""
        return list(dict.fromkeys(r.column for b in self.sensor_blocks() for r in b.drop_when))

    def columns(self) -> list:
        """Every source column/field the spec reads."""
        cols = list(self.body_velocity.columns.values()) if self.body_velocity is not None else []
        o = self.orientation
        cols += ([getattr(o.quaternion, k) for k in "xyzw"] if o.quaternion is not None
                 else [getattr(o.euler, k) for k in ("roll", "pitch", "yaw")])
        w = self.wrench
        cols += list(w.columns.values()) if w.columns is not None else list(w.thrusters.pwm_columns)
        if w.thrusters is not None and w.thrusters.voltage_column:
            cols.append(w.thrusters.voltage_column)
        for p in self.positions():
            if p.use == "check":
                cols += self._block_columns(p)
        cols += [c for c in self.native_columns() + self.span_exempt_columns() if c not in cols]
        cols += [r.column for r in self.exclude_samples if r.column not in cols]
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
            cfg = tm.thruster_block(ingestion_thruster_block(spec), base)
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
        spec._thruster_cfg = tm.thruster_block(ingestion_thruster_block(spec))
    return spec._thruster_cfg


def ingestion_thruster_block(spec: IngestSpec) -> dict:
    """The thrusters block ingestion uses: with voltage_column, the unscaled T200 curve (each sample is rescaled
    to its own logged voltage), so a nominal voltage: meant for the simulators isn't applied twice."""
    b = spec.wrench.thrusters.block()
    if spec.wrench.thrusters.voltage_column:
        b.pop("voltage", None)
    return b


def _session_date(spec: IngestSpec, path: str, start_unix_s: Optional[float] = None) -> str:
    ds = spec.session_date
    if ds.source == "recording":
        if start_unix_s is None or start_unix_s < 1e8:
            raise IngestError(f"{path}: session_date from: recording needs real clock timestamps (a ROS bag, a "
                              f"MAVLink .tlog, an ArduPilot .bin with GPS time, or a "
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
    """(DataFrame, per-file report, native readings). Bags and logs come back already on a uniform grid, with each
    sensor reading at its own time in native ({column: (times, values)}); a CSV's rows are its readings (None)."""
    if spec.source.format == "csv":
        df = pd.read_csv(path, sep=spec.source.delimiter)
        return df, {"file": path, "rows_in": int(len(df)), "warnings": []}, None
    o = spec.orientation
    quat = [[getattr(o.quaternion, k) for k in "xyzw"]] if o.quaternion is not None else []
    angles = ({getattr(o.euler, k): (360.0 if o.euler.unit == "deg" else 2 * np.pi) for k in ("roll", "pitch", "yaw")}
              if o.euler is not None else {})
    hold = spec.wrench.thrusters.pwm_columns if spec.wrench.thrusters is not None else []
    sensors = {"native_columns": spec.native_columns(), "span_exempt": spec.span_exempt_columns()}
    if spec.source.format == "rosbag":
        import rosbag_reader as rb
        try:
            df, src_rep = rb.read_bag(path, spec.columns(), spec.resample_hz, spec.source.stamp, hold, quat,
                                      spec.source.message_types, angle_columns=angles, **sensors)
        except rb.BagError as e:
            raise IngestError(str(e)) from e
        key = "bag"
    else:   # ArduPilot, PX4, MOOS-IvP, LCM: one read_log interface (#94)
        import importlib
        reader = importlib.import_module(LOG_FORMATS[spec.source.format])
        extra = ({"type_files": spec.source.lcm_types, "stamp": spec.source.stamp}
                 if spec.source.format == "lcm" else {})
        try:
            df, src_rep = reader.read_log(path, spec.columns(), spec.resample_hz, hold, quat, angle_columns=angles,
                                          **extra, **sensors)
        except ValueError as e:   # each reader's error is a ValueError with a plain message
            raise IngestError(str(e)) from e
        key = "log"
    native = src_rep.pop("native", {})
    report = {"file": path, "rows_in": int(len(df)), "warnings": src_rep.pop("warnings"), key: src_rep}
    return df, report, native


def _block_readings(spec: IngestSpec, b, df, t, native, path) -> tuple:
    """A sensor block's own readings: (times, (n, k) values in the log's units). From a log: each column at its own
    times (one message carries them all, so the times must match); from a CSV: the rows where every column has a
    value (an empty cell is no reading)."""
    cols = b.cols if isinstance(b, VelocityBlock) else spec._block_columns(b)
    if native is None:
        vals = np.stack([_col(df, c, path) for c in cols], 1)
        ok = np.all(np.isfinite(vals), axis=1)
        return t[ok], vals[ok]
    series = [native.get(c, (np.empty(0), np.empty(0))) for c in cols]
    t0 = series[0][0]
    if any(len(ts) != len(t0) or not np.array_equal(ts, t0) for ts, _ in series[1:]):
        raise IngestError(f"{path}: {', '.join(cols)} come with different timestamps: one sensor's readings must come "
                          f"from one message ({type(b).__name__.replace('Spec', '').lower()} block)")
    return np.asarray(t0, float), np.stack([np.asarray(v, float) for _, v in series], 1) if len(t0) else \
        np.empty((0, len(cols)))


def _sensor_channel(spec: IngestSpec, b, t_n, v, path, report) -> dict:
    """One sensor block's readings -> SI units in Strathe's frames, held repeats and impossible fixes dropped, the
    sensor's delay taken off its times, and its own rate and noise measured. Returns the channel's description, with
    the readings under "_t" / "_v" (placed on the grid later)."""
    import sensor_ingest as si
    info = {"dropped": {}}
    if isinstance(b, PositionSpec):
        kind, name = "position", b.name
        if b.latlon is not None:
            ll = v * (1.0 if b.latlon.unit == "deg" else 180.0 / np.pi)
            if len(ll) and (np.nanmax(np.abs(ll[:, 0])) > 90 or np.nanmax(np.abs(ll[:, 1])) > 360):
                raise IngestError(f"{path}: position {name}: latitudes/longitudes out of range (max |lat| "
                                  f"{np.nanmax(np.abs(ll[:, 0])):.1f}); check lat/lon columns and unit")
            ok = ~((ll[:, 0] == 0) & (ll[:, 1] == 0))   # 0, 0: a GPS with no fix
            if (~ok).any():
                info["dropped"]["no_fix_zero"] = int((~ok).sum())
            t_n, ll = t_n[ok], ll[ok]
            origin = (float(ll[0, 0]), float(ll[0, 1])) if len(ll) else (0.0, 0.0)
            v = si.latlon_to_ned(ll[:, 0], ll[:, 1], *origin) if len(ll) else np.empty((0, 2))
            info["latlon_origin"] = [round(origin[0], 8), round(origin[1], 8)]
        else:
            v = v * LENGTH_UNITS_POS[b.unit]
            if spec.frame == "flu":
                v = v @ (ENU_TO_NED.T if v.shape[1] == 3 else ENU_TO_NED[:2, :2].T)
        columns = [f"pos_{name}_{a}" for a in "ned"[:v.shape[1]]]
        t_n, v, held = si.collapse_repeats(t_n, v)
        if held:
            info["dropped"]["held_repeats"] = held
        info["source"] = b.source
    elif isinstance(b, DepthSpec):
        kind, name = "depth", b.name
        if b.pressure is not None:
            v, how = si.pressure_to_depth(v[:, 0], b.pressure.unit, b.pressure.water, b.pressure.surface)
            v = v[:, None]
            info["pressure"] = how
        else:
            v = v * LENGTH_UNITS_POS[b.unit] * (1.0 if b.positive == "down" else -1.0)
        columns = [f"depth_{name}"]
        if len(v) and np.nanmax(np.abs(v)) > si.MAX_PLAUSIBLE_DEPTH_M:
            raise IngestError(f"{path}: depth {name} reaches {np.nanmax(np.abs(v)):.0f} m: check its unit (or the "
                              f"pressure unit)")
    else:
        kind, name = "accelerometer", "acc"
        v = v * si.ACCEL_UNITS[b.unit]
        if b.mount_rpy_deg is not None and any(b.mount_rpy_deg):
            v = v @ Rotation.from_euler("xyz", b.mount_rpy_deg, degrees=True).as_matrix().T
        if spec.frame == "flu":
            v = v @ FLU_TO_FRD
        columns = ["acc_x", "acc_y", "acc_z"]
    if b.delay_s:
        t_n = t_n - b.delay_s
    if b.delay_s is not None and (b.delay_s or kind in ("position", "depth")):   # their 0 is said too: on time
        info["delay_s"] = b.delay_s
    if kind == "position" and len(t_n) > 2:
        # the tolerance comes from what the sensor type can do, not from these fixes: bad fixes would inflate it
        sigma = b.noise_m if b.noise_m is not None else si.FIX_NOISE_M[b.source]
        t_n, v, bad = si.drop_unreachable_fixes(t_n, v, b.max_speed_mps, 3.0 * np.sqrt(2.0) * sigma)
        if bad:
            info["dropped"]["unreachable"] = bad
            share = bad / (bad + len(t_n))
            if share > si.OUTLIER_SHARE_REFUSED:
                raise IngestError(f"{path}: position {name}: {bad} of {bad + len(t_n)} fixes ({100 * share:.0f}%) "
                                  f"are further from the last good fix than the vehicle could move at "
                                  f"{b.max_speed_mps:g} m/s: check the unit, the columns, and that x and y are one "
                                  f"system's (or raise max_speed_mps)")
            report["warnings"].append(f"position {name}: dropped {bad} fix(es) the vehicle couldn't have reached at "
                                      f"{b.max_speed_mps:g} m/s (outliers)")
    if kind == "depth" and len(t_n) > 2:
        dts = np.diff(t_n)
        far = dts >= 0.05
        if far.sum() > 10:
            rate = np.abs(np.diff(v[:, 0]))[far] / dts[far]
            if float(np.percentile(rate, 99)) > si.MAX_PLAUSIBLE_DEPTH_RATE_MPS:
                raise IngestError(f"{path}: depth {name} changes at {float(np.percentile(rate, 99)):.1f} m/s (99th "
                                  f"percentile): check its unit (a pressure in the wrong unit?) or that the column is "
                                  f"depth")
    info.update(kind=kind, name=name, columns=columns, native=si.native_stats(t_n, v))   # after outliers are gone
    info["_t"], info["_v"] = t_n, v
    return info


def _grid_rule_values(df, column, t_rows, finite, grid, resampled, native, path):
    """A drop_when rule's column on the final grid, as its last reading at each sample (NaN before its first): a log's
    span_exempt column is already that; a CSV's empty cells are no new reading, the same as a log's gaps between
    messages, so the last value is held there too."""
    x = _col(df, column, path)
    if native is not None:
        return x[finite] if finite is not None else x
    ok = np.isfinite(x)
    tr, xr = t_rows[ok], x[ok]
    if not len(tr):
        return np.full(len(grid), np.nan)
    idx = np.searchsorted(tr, grid + 1e-6, side="right") - 1
    return np.where(idx >= 0, xr[np.clip(idx, 0, len(xr) - 1)], np.nan)


def convert_file(spec: IngestSpec, path: str) -> tuple:
    """Returns (canonical DataFrame, per-file report dict)."""
    import sensor_ingest as si
    df, report, native = _read_source(spec, path)
    n = len(df)
    if n < 10:
        raise IngestError(f"{path}: only {n} rows")
    start_unix_s = None

    # time base
    if spec.source.format in LOG_FORMATS:
        import rosbag_reader as rb
        t = df[rb.TIME_COLUMN].to_numpy(dtype=float)
        rate = spec.resample_hz
        start_unix_s = (report.get("bag") or report.get("log"))["recording_start_unix_s"]
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
            missing = float(np.sum(dts[gaps] - med))
            report["time_gaps_total_s"] = round(missing, 2)
            report["warnings"].append(f"{len(gaps)} time gap(s) longer than 2.5 samples, {missing:.1f} s in all "
                                      f"({100 * missing / max(float(t[-1] - t[0]), 1e-9):.0f}% of the recording); "
                                      f"open-loop replay will integrate across them")
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
    vel = spec.body_velocity
    lin_dense = vel is not None and vel.has_linear and vel.linear_gaps == "interpolate"
    lin_kept = vel is not None and vel.has_linear and vel.linear_gaps == "keep"
    lin = ang = None
    if lin_dense:
        vc = vel.columns
        lin = np.stack([_col(df, vc[k], path) for k in "uvw"], 1) * LINEAR_UNITS[vel.linear_unit]
    if vel is not None and vel.has_angular:
        vc = vel.columns
        ang = np.stack([_col(df, vc[k], path) for k in "pqr"], 1) * ANGULAR_UNITS[vel.angular_unit]
    # a rotated sensor mount: readings in the sensor's axes -> the vehicle's (still in the spec's frame convention)
    mounts = {}
    for name, which in (("linear_mount_rpy_deg", "lin"), ("angular_mount_rpy_deg", "ang")):
        rpy = getattr(vel, name) if vel is not None else None
        if rpy is not None and any(rpy):
            M = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
            mounts[which] = M
            if which == "lin" and lin is not None:
                lin = lin @ M.T
            elif which == "ang" and ang is not None:
                ang = ang @ M.T
            report.setdefault("sensor_mounts", {})[name] = list(map(float, rpy))
    o = spec.orientation
    if o.quaternion is not None:
        q = np.stack([_col(df, getattr(o.quaternion, k), path) for k in "xyzw"], 1)
        norms = np.linalg.norm(q, axis=1)
        # bags: checked on the raw messages (the grid values were slerped, so are unit length by construction)
        src = report.get("bag") or report.get("log") or {}
        checked = np.array(src["quaternion_norm_range"]) if "quaternion_norm_range" in src else norms
        if np.any(np.abs(checked - 1) > 0.05):
            raise IngestError(f"{path}: quaternion norms range {checked.min():.3f}-{checked.max():.3f}; expected 1 "
                              f"(wrong columns, or a w-first/w-last mix-up?)")
        R = Rotation.from_quat(q / norms[:, None]).as_matrix()
    else:
        eu = np.stack([_col(df, getattr(o.euler, k), path) for k in ("roll", "pitch", "yaw")], 1)
        R = Rotation.from_euler("xyz", eu * ANGLE_UNITS[o.euler.unit]).as_matrix()
    if lin is not None and vel.linear_frame == "world":
        lin = np.einsum("nji,nj->ni", R, lin)  # R maps body to world, so R^T takes world velocity to body
        report["velocity_rotated_from_world_frame"] = True
    if spec.wrench.columns is not None:
        wc = spec.wrench.columns
        wrench = np.stack([_col(df, wc[k], path) for k in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")], 1)
        report["wrench_source"] = "logged forces"
    else:
        th = spec.wrench.thrusters
        cfg = _thruster_config(spec)
        pwm = np.stack([_col(df, c, path) for c in th.pwm_columns], 1)
        lo_hi = cfg.command_range
        what = "PWM" if lo_hi is None else "normalized commands"
        if spec.source.format != "csv" and not np.all(np.isfinite(pwm)):
            # a log message's command that isn't a number: PX4's actuator_motors sends NaN for a stopped motor (and
            # while disarmed), so those samples are taken as stopped, as PWM 0 is below
            nan = ~np.isfinite(pwm)
            report["warnings"].append(f"{int(nan.any(axis=1).sum())} sample(s) have a thruster command that isn't a "
                                      f"number (PX4 actuator_motors: NaN is a stopped motor), taken as stopped")
            pwm = np.where(nan, cfg.command_neutral, pwm)
        if lo_hi is None and np.any(pwm == 0):
            # PWM 0 is no signal: an autopilot writes it for an output it doesn't drive or while disarmed (PX4 does
            # both). An ESC with no signal stops its thruster, so those samples are taken as stopped (#94)
            off = pwm == 0
            dead = [th.pwm_columns[j] for j in range(pwm.shape[1]) if off[:, j].all()]
            some = int(off[:, [j for j in range(pwm.shape[1]) if not off[:, j].all()]].any(axis=1).sum()) \
                if len(dead) < pwm.shape[1] else 0
            if len(dead) == pwm.shape[1]:
                raise IngestError(f"{path}: every thruster command is 0 throughout: no output was driven (disarmed for "
                                  f"the whole log, or the wrong columns)")
            if dead:
                report["warnings"].append(f"{', '.join(dead)} are 0 throughout: no signal, so those thrusters are taken "
                                          f"as stopped. If they are thrusters that ran, the autopilot sent them on other "
                                          f"outputs: check the pwm_columns")
            if some:
                report["warnings"].append(f"{some} sample(s) have a command of 0 (no signal: disarmed, or an output "
                                          f"switched off), taken as stopped; drop disarmed stretches with exclude_samples "
                                          f"(PX4: column actuator_armed.armed, below: 0.5)")
            pwm = np.where(off, cfg.command_neutral, pwm)
        if lo_hi is None and (np.nanmin(pwm) < 900 or np.nanmax(pwm) > 2100):
            raise IngestError(f"{path}: PWM values span {np.nanmin(pwm):.0f}-{np.nanmax(pwm):.0f}; expected ESC "
                              f"microseconds (~1100-1900). If your commands are normalized (e.g. -1..1), add "
                              f"command: normalized (and command_range) to the thrusters block")
        if lo_hi is not None:
            slack = 0.05 * (lo_hi[1] - lo_hi[0])
            if np.nanmin(pwm) < lo_hi[0] - slack or np.nanmax(pwm) > lo_hi[1] + slack:
                raise IngestError(f"{path}: thruster commands span {np.nanmin(pwm):g} to {np.nanmax(pwm):g}, outside "
                                  f"command_range {list(lo_hi)}; set command_range to the range your commands use")
        if th.voltage_column:
            volts = _col(df, th.voltage_column, path)
            if not np.all(np.isfinite(volts)):
                raise IngestError(f"{path}: {th.voltage_column!r} has missing or non-numeric values")
            clipped = int(np.sum((volts < 12.0) | (volts > 20.0)))
            if clipped:
                report["warnings"].append(f"{clipped} sample(s) of {th.voltage_column!r} outside 12-20 V (the "
                                          f"published T200 range) were clipped to it")
            wrench = cfg.wrench_at_voltage(pwm, np.clip(volts, 12.0, 20.0))
            report["voltage_v"] = {"min": round(float(volts.min()), 2), "median": round(float(np.median(volts)), 2),
                                   "max": round(float(volts.max()), 2)}
            report["wrench_source"] = (f"computed from {what} ({cfg.label}, curve: {cfg.curve.label}, rescaled to "
                                       f"each sample's logged voltage)")
        else:
            wrench = cfg.wrench(pwm)
            report["wrench_source"] = f"computed from {what} ({cfg.label}, curve: {cfg.curve.label})"

    # frame conversion to FRD body / NED world
    if spec.frame == "flu":
        if lin is not None:
            lin = lin @ FLU_TO_FRD
        if ang is not None:
            ang = ang @ FLU_TO_FRD
        if spec.wrench.columns is not None:  # PWM-derived wrenches are already FRD
            wrench = np.hstack([wrench[:, :3] @ FLU_TO_FRD, wrench[:, 3:] @ FLU_TO_FRD])
        R = ENU_TO_NED @ R @ FLU_TO_FRD
    for p in spec.positions():
        if p.use != "check":
            continue
        if p.latlon is not None:
            ll = np.stack([_col(df, p.latlon.lat, path), _col(df, p.latlon.lon, path)], 1) * \
                (1.0 if p.latlon.unit == "deg" else 180.0 / np.pi)
            pos = np.hstack([si.latlon_to_ned(ll[:, 0], ll[:, 1], ll[0, 0], ll[0, 1]), np.zeros((len(ll), 1))])
        else:
            pos = np.stack([_col(df, getattr(p, k), path) if getattr(p, k) else np.zeros(n) for k in "xyz"], 1) * \
                LENGTH_UNITS_POS[p.unit]
            if spec.frame == "flu":
                pos = pos @ ENU_TO_NED.T
        if lin is not None:
            check = orientation_vs_position(t, pos, R, lin)
        else:   # no DVL: the thrust points where the vehicle goes in steady motion
            check = si.heading_vs_fixes(t, pos, R, wrench[:, :3], "thrust")
        report["orientation_vs_position"] = check
        if check.get("problem") and "orientation_vs_position" not in spec.skip_checks:
            raise IngestError(f"{path}: {check['problem']}")
        break

    # each sensor's own readings (before rows are dropped or resampled: they have their own times)
    channels = []
    for b in spec.sensor_blocks():
        ch = _sensor_channel(spec, b, *_block_readings(spec, b, df, t, native, path), path, report)
        ch["_block"] = b
        channels.append(ch)
    dvl_kept = None
    if lin_kept:
        vc = vel.columns
        tb, vb = _block_readings(spec, VelocityBlock([vc[k] for k in "uvw"]), df, t, native, path)
        vb = vb * LINEAR_UNITS[vel.linear_unit]
        if "lin" in mounts:
            vb = vb @ mounts["lin"].T
        dvl_kept = {"_t": tb, "_v": vb, "native": si.native_stats(tb, vb)}

    names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"] + (["Bu", "Bv", "Bw"] if lin is not None else []) + \
        (["Bp", "Bq", "Br"] if ang is not None else []) + ["qx", "qy", "qz", "qw"]
    data = np.hstack([wrench] + ([lin] if lin is not None else []) + ([ang] if ang is not None else [])
                     + [Rotation.from_matrix(R).as_quat()])
    drop = np.zeros(len(data), dtype=bool)
    for rule in spec.exclude_samples:
        x = _col(df, rule.column, path)
        hit = np.zeros_like(drop)
        if rule.above is not None:
            hit |= x > rule.above
        if rule.below is not None:
            hit |= x < rule.below
        report.setdefault("excluded_by_rule", {})[rule.column] = int(hit.sum())
        drop |= hit | ~np.isfinite(x)
    t_rows = t
    finite = np.all(np.isfinite(data), axis=1)
    finite_mask = None
    if not finite.all():
        report["warnings"].append(f"dropped {int((~finite).sum())} row(s) with NaN/inf")
        if (~finite).sum() > 0.05 * n:
            raise IngestError(f"{path}: {int((~finite).sum())} of {n} rows have NaN/inf in required quantities")
        data, t, drop = data[finite], t[finite], drop[finite]
        finite_mask = finite

    resampled = spec.resample_hz is not None and spec.source.format == "csv" and spec.time.column is not None
    if resampled:
        # t[0] + k / rate rather than np.arange(t[0], ...), which drifts at Unix-epoch magnitudes
        grid = t[0] + np.arange(int(np.ceil((t[-1] - t[0]) * spec.resample_hz - 1e-9))) / spec.resample_hz
        cols = [np.interp(grid, t, data[:, j]) for j in range(data.shape[1] - 4)]
        from rosbag_reader import slerp  # orientation: spherical interpolation, valid across sudden jumps
        data = np.hstack([np.stack(cols, 1), slerp(t, data[:, -4:], grid)])
        drop = np.interp(grid, t, drop.astype(float)) > 0  # a grid sample next to a dropped one is dropped too
        report["resampled_to_hz"] = spec.resample_hz
        t = grid

    out = pd.DataFrame(data, columns=names)
    Rg = Rotation.from_quat(out[["qx", "qy", "qz", "qw"]].values).as_matrix()
    if ang is None:   # no gyro logged: rates from the orientation itself
        derived = si.rates_from_orientation(t, Rg)
        for k, c in enumerate(("Bp", "Bq", "Br")):
            out.insert(out.columns.get_loc("qx"), c, derived[:, k])
        report["angular_rates_from"] = "the orientation (no gyro logged)"
        report["warnings"].append("no gyro rates logged: p, q, r are derived from the orientation (differentiated, so "
                                  "noisier and about half a sample late, and as good as the attitude filter). Log the "
                                  "gyro if you can (body_velocity.columns p, q, r)")
    if dvl_kept is not None:
        lin_g = si.place_on_grid(dvl_kept["_t"], dvl_kept["_v"], t)   # in the spec's frame convention
        if vel.linear_frame == "world":   # world (NED, or ENU with frame: flu) -> body FRD with the orientation
            lin_g = np.einsum("nji,nj->ni", Rg, lin_g @ ENU_TO_NED.T if spec.frame == "flu" else lin_g)
            report["velocity_rotated_from_world_frame"] = True
        elif spec.frame == "flu":
            lin_g = lin_g @ FLU_TO_FRD
        for k, c in enumerate(("Bu", "Bv", "Bw")):
            out.insert(out.columns.get_loc("Bp"), c, lin_g[:, k])
        got = int(np.all(np.isfinite(lin_g), axis=1).sum())
        report.setdefault("sensors", {})["dvl"] = {
            "kind": "dvl", "columns": ["Bu", "Bv", "Bw"], "native": dvl_kept["native"], "readings_on_grid": got,
            "coverage_pct": round(100.0 * got / max(len(out), 1), 1), "gaps": "kept (NaN where no reading)"}

    for ch in channels:
        vals = si.place_on_grid(ch.pop("_t"), ch.pop("_v"), t)
        blk = ch.pop("_block")
        by_rule = 0
        for rule in blk.drop_when:
            x = _grid_rule_values(df, rule.column, t_rows, finite_mask, t, resampled, native, path)
            hit = ~np.isfinite(x)
            if rule.above is not None:
                hit |= x > rule.above
            if rule.below is not None:
                hit |= x < rule.below
            had = np.all(np.isfinite(vals), axis=1)
            by_rule += int((hit & had).sum())
            vals[hit] = np.nan
        if blk.drop_when:
            ch["dropped"]["by_drop_when"] = by_rule
        for k, c in enumerate(ch["columns"]):
            out[c] = vals[:, k]
        got = int(np.all(np.isfinite(vals), axis=1).sum())
        ch["readings_on_grid"] = got
        ch["coverage_pct"] = round(100.0 * got / max(len(out), 1), 1)
        if got == 0:
            report["warnings"].append(f"{ch['kind']} {ch['name']}: no readings left in this recording"
                                      + (" after drop_when" if by_rule else ""))
        report.setdefault("sensors", {})[ch["name"] if ch["kind"] != "accelerometer" else "acc"] = \
            {k: v for k, v in ch.items() if not k.startswith("_")}

    if spec.exclude_samples:
        out[DROP_COLUMN] = drop
    Rq = Rotation.from_quat(out[["qx", "qy", "qz", "qw"]].values)
    jumps = int(np.sum(np.degrees((Rq[1:] * Rq[:-1].inv()).magnitude()) > MAX_PLAUSIBLE_ORIENTATION_JUMP_DEG))
    if jumps:
        report["warnings"].append(f"orientation jumps by more than {MAX_PLAUSIBLE_ORIENTATION_JUMP_DEG:g} deg between "
                                  f"consecutive samples {jumps} time(s): likely sensor glitches or heading resets; "
                                  f"they add error to the calibration")

    # unit sanity: the silent errors that ruin a calibration
    has_lin = "Bu" in out.columns
    lin_max = float(np.nanmax(np.abs(out[["Bu", "Bv", "Bw"]].values))) if has_lin and \
        np.isfinite(out[["Bu", "Bv", "Bw"]].values).any() else 0.0
    ang_max = float(np.abs(out[["Bp", "Bq", "Br"]].values).max())
    if has_lin:
        report["max_abs_linear_velocity_mps"] = round(lin_max, 4)
    report["max_abs_angular_velocity_radps"] = round(ang_max, 4)
    if ang is not None and ang_max > MAX_PLAUSIBLE_ANGULAR_RADPS:
        raise IngestError(f"{path}: angular rates reach {ang_max:.1f} rad/s after conversion -- implausible for "
                          f"an ROV; were they logged in deg/s while the spec says {vel.angular_unit}?")
    if lin_max > MAX_PLAUSIBLE_LINEAR_MPS:
        raise IngestError(f"{path}: linear speeds reach {lin_max:.1f} m/s after conversion -- check linear_unit")
    if ang_max < 1e-4 or (has_lin and lin_max < 1e-4):
        report["warnings"].append("velocities are essentially zero throughout -- nothing to calibrate from")
    _sensor_checks(spec, out, t, Rg, ang is not None, path, report)
    report["rows_out"] = int(len(out))
    report["sampling_rate_hz"] = float(rate)
    report["session_date"] = _session_date(spec, path, start_unix_s)
    return out, report


class VelocityBlock:
    """The DVL's u, v, w as a sensor block (linear_gaps: keep), for _block_readings."""
    def __init__(self, cols):
        self.cols = cols


def _sensor_checks(spec: IngestSpec, out: pd.DataFrame, t, R, gyro_logged: bool, path: str, report: dict) -> None:
    """The checks that catch a sensor's silent errors (#96). Each problem stops ingestion unless its check is listed
    in skip_checks; the numbers are kept in the report either way."""
    import sensor_ingest as si
    checks = {}
    if gyro_logged:
        checks["gyro_vs_orientation"] = si.gyro_vs_orientation(t, R, out[["Bp", "Bq", "Br"]].to_numpy())
    if "acc_x" in out.columns:
        checks["accelerometer"] = si.accelerometer_gravity(R, out[["acc_x", "acc_y", "acc_z"]].to_numpy(),
                                                           out[["Bp", "Bq", "Br"]].to_numpy())
    depth_cols = [c for c in out.columns if c.startswith("depth_")]
    down_cols = [c for c in out.columns if c.startswith("pos_") and c.endswith("_d")]
    for dc in depth_cols:
        for pc in down_cols:
            checks.setdefault("depth_vs_position", {})[f"{dc} vs {pc}"] = si.depth_vs_position(
                out[dc].to_numpy(), out[pc].to_numpy())
        if not down_cols:   # nothing vertical to compare with: the thrust says which way is down
            checks.setdefault("depth_vs_thrust", {})[dc] = si.depth_vs_thrust(
                t, out[dc].to_numpy(), R, out[["Fx", "Fy", "Fz"]].to_numpy())
    for p in spec.positions():
        if p.use != "calibrate":
            continue
        cols = [f"pos_{p.name}_n", f"pos_{p.name}_e"]
        pos = out[cols].to_numpy()
        if "Bu" in out.columns and np.isfinite(out["Bu"]).mean() > 0.9:
            body, what = out[["Bu", "Bv", "Bw"]].to_numpy(), "body velocity"
        else:
            body, what = out[["Fx", "Fy", "Fz"]].to_numpy(), "thrust"
        checks.setdefault("heading_vs_position", {})[p.name] = si.heading_vs_fixes(t, pos, R, np.nan_to_num(body),
                                                                                    what)
    for name, res in checks.items():
        report.setdefault("sensor_checks", {})[name] = res
        if "warning" in res:
            report["warnings"].append(f"{name}: {res['warning']}")
        report["warnings"] += [f"{name}: {k}: {r['warning']}" for k, r in res.items()
                               if isinstance(r, dict) and "warning" in r]
        problems = [res["problem"]] if "problem" in res else \
            [f"{k}: {r['problem']}" for k, r in res.items() if isinstance(r, dict) and "problem" in r]
        if problems and name not in spec.skip_checks:
            raise IngestError(f"{path}: {'; '.join(problems)}. If you're sure it's right, add "
                              f"`skip_checks: [{name}]` to the spec")
        if problems:
            report["warnings"].append(f"{name}: {'; '.join(problems)} (check skipped by skip_checks)")


def torque_reference_for(spec: IngestSpec) -> Optional[list]:
    """The torque_reference_offset_m calibration should apply to these files' torques, when the
    ingest spec determines it: the BlueROV2 preset reproduces the reference dataset's torque point
    (8.5 cm below the thrusters); a layout or allocation file gives torques about the point its
    positions are measured from (offset 0). Logged forces: unknown here (None = vehicle_spec.yaml)."""
    if spec.wrench.thrusters is None:
        return None
    return [float(x) for x in _thruster_config(spec).torque_reference_offset_m]


DROP_COLUMN = "_excluded"


def split_excluded(out: pd.DataFrame, rate_hz: float, min_segment_s: float) -> list:
    """exclude_samples: the stretches between dropped samples, each at least min_segment_s long."""
    if DROP_COLUMN not in out.columns:
        return [out]
    keep = ~out[DROP_COLUMN].to_numpy(dtype=bool)
    out = out.drop(columns=[DROP_COLUMN])
    pieces, start = [], None
    for i, k in enumerate(np.append(keep, False)):
        if k and start is None:
            start = i
        elif not k and start is not None:
            if (i - start) / rate_hz >= min_segment_s:
                pieces.append(out.iloc[start:i].reset_index(drop=True))
            start = None
    return pieces


AXIS_PAIRS = (("Fx", "Bu", "surge"), ("Fy", "Bv", "sway"), ("Fz", "Bw", "heave"), ("Tx", "Bp", "roll"),
              ("Ty", "Bq", "pitch"), ("Tz", "Br", "yaw"))
REVERSED_CORRELATION = -0.2
WEAK_CORRELATION = 0.2
DIRECTION_CHECKED_AXES = ("surge", "sway", "heave", "yaw")


def force_direction_check(frames: list, unreachable: list) -> dict:
    """Does each axis's force push the way the vehicle moves? Pooled over every converted file, the correlation of
    force with velocity on each axis. Thrust mostly balances drag, so on an axis the thrusters drive it is clearly
    positive; clearly negative means the force is reversed: a mirrored thruster layout, wrong `reversed:` flags, a
    frame mix-up, or (with world-frame velocity) a mirrored heading. Found by hand on DFKI's BlueROV2, where the
    old preset gave yaw -0.65 (PROJECT_STATUS.md #62, #63)."""
    d = pd.concat(frames, ignore_index=True)
    out = {}
    for f, v, name in AXIS_PAIRS:
        if v not in d.columns:   # no sensor sees this axis's motion (#96)
            out[name] = None
            continue
        x, y = d[f], d[v]
        ok = x.notna() & y.notna()
        if not ok.all():
            x, y = x[ok], y[ok]
        if ok.sum() < 3 or x.std() < 1e-9 or y.std() < 1e-9:
            out[name] = None
            continue
        out[name] = round(float(np.corrcoef(x, y)[0, 1]), 3)
    skip = {a.split()[0] for a in unreachable}  # "pitch torque" -> "pitch"
    # roll and pitch are reported, not judged: restoring moments (buoyancy above weight) dominate them, so their
    # torque and rate can anti-correlate with the forces the right way round (-0.22 on synthetic test logs)
    reversed_axes = [a for a, c in out.items() if c is not None and c < REVERSED_CORRELATION and a not in skip
                     and a in DIRECTION_CHECKED_AXES]
    # near zero: not reversed, but force and motion don't move together. A mirrored heading with world-frame
    # velocity does this: on DFKI, ArduSub's mirrored yaw took surge from +0.79 to -0.05 and sway from +0.57 to
    # +0.08 (PROJECT_STATUS.md #63); refusing at -0.2 alone would have let it through
    weak_axes = [a for a, c in out.items() if c is not None and REVERSED_CORRELATION <= c < WEAK_CORRELATION
                 and a not in skip and a in DIRECTION_CHECKED_AXES]
    return {"correlation_force_vs_velocity": out, "reversed_axes": reversed_axes, "weak_axes": weak_axes}


def orientation_vs_position(t, pos, R, lin_body) -> dict:
    """Does the orientation turn the body velocity into the motion the positions show? Horizontal velocity from the
    position track against the body velocity rotated into the world, as complex numbers: a correct heading gives a
    ratio near 1; a heading offset rotates it; a mirrored heading matches the conjugate instead. Needs body-frame
    velocity: world-frame velocity was rotated with this same orientation, so it can't test it."""
    dt = np.gradient(t)
    k = max(1, int(round(0.5 / max(float(np.median(dt)), 1e-6))))  # differentiate over ~1 s to beat position noise
    if len(t) < 4 * k:
        return {}
    vp = (pos[2 * k:] - pos[:-2 * k]) / (t[2 * k:] - t[:-2 * k])[:, None]
    vb = np.einsum("nij,nj->ni", R, lin_body)[k:-k]
    a, b = vp[:, 0] + 1j * vp[:, 1], vb[:, 0] + 1j * vb[:, 1]
    moving = np.abs(a) > 0.05
    if moving.sum() < 20:
        return {"note": "too little horizontal motion to check"}
    a, b = a[moving], b[moving]

    def fit(x):
        c = np.vdot(x, a) / np.vdot(x, x)  # least-squares complex gain: rotation + scale
        return float(np.angle(c, deg=True)), float(1 - np.linalg.norm(a - c * x) ** 2 / np.linalg.norm(a) ** 2)
    off, r2 = fit(b)
    moff, mr2 = fit(np.conj(b))
    out = {"heading_offset_deg": round(off, 1), "match_r2": round(r2, 3), "mirrored_match_r2": round(mr2, 3)}
    if mr2 > 0.5 and mr2 > r2 + 0.2:
        out["problem"] = (f"the orientation's heading is mirrored relative to the positions (it turns the wrong way: "
                          f"the velocity matches the position track only when the heading's sign is flipped, "
                          f"R^2 {mr2:.2f} against {r2:.2f}). Check the heading source's frame (e.g. a compass in a "
                          f"different convention) or set frame: correctly")
    elif r2 >= 0.5 and abs(off) > 20:
        out["problem"] = (f"the orientation's heading is about {off:.0f} degrees off the direction the positions show "
                          f"the vehicle moving (a steady offset, R^2 {r2:.2f} once it's removed). Check the heading "
                          f"source (a compass offset or declination, or a different world frame from the positions)")
    elif r2 < 0.5 and mr2 < 0.5:
        out["problem"] = (f"the body velocity, turned by the orientation, doesn't follow the position track (R^2 "
                          f"{r2:.2f}, {mr2:.2f} if the heading were mirrored). Check that velocity, orientation and "
                          f"positions are in the frames the spec says, and on one clock")
    return out


LINEAR_AXES_SEEN_BY = {"surge": ("dvl", "horizontal position"), "sway": ("dvl", "horizontal position"),
                       "heave": ("dvl", "depth", "vertical position")}


def velocity_proxies(df: pd.DataFrame, rate_hz: float) -> pd.DataFrame:
    """For the force-direction check on a vehicle with no DVL: body velocity from its position fixes (surge, sway) and
    depth (heave), over 2-6 s between readings so their noise doesn't swamp the motion. Only for that sign check;
    calibration never uses it."""
    import sensor_ingest as si
    if "Bu" in df.columns and df["Bu"].notna().sum() >= 20:   # a DVL, gaps and all: its own readings (pairwise)
        return df
    out = df.drop(columns=[c for c in ("Bu", "Bv", "Bw") if c in df.columns])
    t = np.arange(len(df)) / rate_hz
    R = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].values).as_matrix()
    hor = [c[:-2] for c in df.columns if c.startswith("pos_") and c.endswith("_n")]
    if hor:
        mids, v = si.fix_velocities(t, df[[hor[0] + "_n", hor[0] + "_e"]].to_numpy(), max_gap_s=6.0, min_dt_s=2.0)
        if len(mids):
            vw = np.zeros((len(mids), 3))
            vw[:, :2] = v
            b = np.einsum("nji,nj->ni", R[mids], vw)
            for k, c in enumerate(("Bu", "Bv")):
                out[c] = np.nan
                out.loc[mids, c] = b[:, k]
    vert = [c for c in df.columns if c.startswith("depth_")] + \
           [c for c in df.columns if c.startswith("pos_") and c.endswith("_d")]
    if vert:
        mids, v = si.fix_velocities(t, df[[vert[0]]].to_numpy(), max_gap_s=6.0, min_dt_s=2.0)
        if len(mids):
            vw = np.zeros((len(mids), 3))
            vw[:, 2] = v[:, 0]
            out["Bw"] = np.nan
            out.loc[mids, "Bw"] = np.einsum("nji,nj->ni", R[mids], vw)[:, 2]
    return out


def sensors_summary(reports: list, frames: list) -> dict:
    """Per sensor, over every recording: readings, rate, coverage, measured noise; and which axes the sensors see
    (#96). Rotation is always seen (the gyro, or rates from the orientation)."""
    sensors = {}
    for rep in reports:
        for name, ch in rep.get("sensors", {}).items():
            agg = sensors.setdefault(name, {"kind": ch["kind"], "columns": ch["columns"], "readings": 0,
                                            "rates_hz": [], "noise_std": [], "coverage_pct": [], "dropped": {}})
            agg["readings"] += int(ch.get("readings_on_grid", 0))
            nat = ch.get("native", {})
            if nat.get("rate_hz"):
                agg["rates_hz"].append(nat["rate_hz"])
            if nat.get("noise_std"):
                agg["noise_std"].append(nat["noise_std"])
            agg["coverage_pct"].append(ch.get("coverage_pct", 0.0))
            for k, v in ch.get("dropped", {}).items():
                agg["dropped"][k] = agg["dropped"].get(k, 0) + int(v)
            if ch.get("native", {}).get("filtered"):
                agg["filtered"] = True
            if "source" in ch:
                agg["source"] = ch["source"]
            if "delay_s" in ch:   # said in the ingest spec: calibration takes it as given (measurements.given_delays)
                agg["delay_s"] = ch["delay_s"]
    for agg in sensors.values():
        agg["rate_hz"] = round(float(np.median(agg.pop("rates_hz"))), 3) if agg["rates_hz"] else None
        ns = agg.pop("noise_std")
        agg["noise_std"] = [round(float(x), 6) for x in np.median(np.array(ns), axis=0)] if ns else None
        cov = agg.pop("coverage_pct")
        agg["coverage_pct"] = round(float(np.mean(cov)), 1) if cov else 0.0
    cols = set().union(*(f.columns for f in frames)) if frames else set()
    have = set()
    if "Bu" in cols and any(f["Bu"].notna().any() for f in frames if "Bu" in f.columns):
        have.add("dvl")
    if any(c.startswith("pos_") and c.endswith("_n") for c in cols):
        have.add("horizontal position")
    if any(c.startswith("depth_") for c in cols):
        have.add("depth")
    if any(c.startswith("pos_") and c.endswith("_d") for c in cols):
        have.add("vertical position")
    seen = {ax: sorted(set(by) & have) for ax, by in LINEAR_AXES_SEEN_BY.items()}
    gyro = "gyro" if not any(r.get("angular_rates_from") for r in reports) else "rates from the orientation"
    seen.update({ax: [gyro] for ax in ("roll", "pitch", "yaw")})
    if any(c.startswith("acc_") for c in cols):
        have.add("accelerometer")
    return {"sensors": sensors, "have": sorted(have), "axes_seen_by": seen,
            "axes_not_seen": [ax for ax, by in seen.items() if not by]}


def ingest(spec: IngestSpec, out_dir: str) -> dict:
    if spec.source.format == "rosbag":
        import rosbag_reader as rb
        files = rb.bag_paths(spec.source.files)
    elif spec.source.format in LOG_FORMATS:
        import importlib
        files = importlib.import_module(LOG_FORMATS[spec.source.format]).log_paths(spec.source.files)
    else:
        files = sorted(glob.glob(spec.source.files))
    if not files:
        raise IngestError(f"no files match {spec.source.files!r}")
    os.makedirs(out_dir, exist_ok=True)
    reports, rates, counters = [], [], {}
    frames = []
    for path in files:
        df_all, rep0 = convert_file(spec, path)
        pieces = split_excluded(df_all, rep0["sampling_rate_hz"], spec.min_segment_s)
        if spec.exclude_samples:
            kept = sum(len(p) for p in pieces)
            rep0["kept_after_exclusions"] = f"{kept} of {len(df_all)} samples, in {len(pieces)} piece(s)"
        for j, df in enumerate(pieces):
            rep = dict(rep0) if len(pieces) == 1 else {**rep0, "piece": j + 1, "rows_out": int(len(df))}
            frames.append(df)
            date = rep["session_date"]
            counters[date] = counters.get(date, 0) + 1
            name = f"{spec.vehicle}-{date}-run{counters[date]}.csv"
            df.to_csv(os.path.join(out_dir, name), index=False)
            rep["output"] = name
            reports.append(rep)
            rates.append(rep["sampling_rate_hz"])
        if not pieces:
            rep0["output"] = None
            rep0["warnings"].append(f"nothing left after exclude_samples (no stretch of {spec.min_segment_s:g} s)")
            reports.append(rep0)
    if not frames:
        raise IngestError("nothing left to calibrate on after exclude_samples; loosen the rules or min_segment_s")
    if max(rates) - min(rates) > 0.01 * min(rates):
        raise IngestError(f"files have different sampling rates ({sorted(set(round(r, 3) for r in rates))} Hz); "
                          f"set resample_hz so every run shares one rate")
    summary = {"vehicle": spec.vehicle, "n_files": len(files), "n_recordings": len(frames),
               "sampling_rate_hz": round(rates[0], 4),
               "sessions": {d: c for d, c in sorted(counters.items())},
               "torque_reference_offset_m": torque_reference_for(spec), "files": reports,
               "next_step": f"python strathe.py calibrate --data {out_dir} --name {spec.vehicle}"}
    sens = sensors_summary(reports, frames)
    if sens["sensors"] or sens["axes_not_seen"]:
        summary["sensors"] = sens
    if sens["axes_not_seen"]:
        summary.setdefault("warnings", []).append(
            f"no sensor sees {', '.join(sens['axes_not_seen'])} motion (no DVL, position fixes or depth for "
            f"{'those axes' if len(sens['axes_not_seen']) > 1 else 'that axis'}): calibration can't learn "
            f"{'their' if len(sens['axes_not_seen']) > 1 else 'its'} drag or added mass from these logs and keeps "
            f"your vehicle description's values (calibration/SENSORS.md)")
    unreach = _thruster_config(spec).unreachable_axes() if spec.wrench.thrusters is not None else []
    fd = force_direction_check([velocity_proxies(f, rates[0]) for f in frames], unreach)
    if "dvl" not in sens["have"]:
        fd["velocity_from"] = "position fixes and depth (no DVL)"
    summary["force_direction"] = fd
    if fd["reversed_axes"]:
        msg = (f"the force on {', '.join(fd['reversed_axes'])} pushes against the way the vehicle moves (correlation "
               + ", ".join(f"{a} {fd['correlation_force_vs_velocity'][a]:+.2f}" for a in fd["reversed_axes"])
               + "; on an axis the thrusters drive it should be clearly positive). The thruster direction or layout "
                 "looks reversed: check each thruster's `direction` and `reversed`, the PWM column order, `frame:`, "
                 "and (for world-frame velocity) the heading. Calibrating with reversed forces fits nonsense.")
        if "force_direction" in spec.skip_checks:
            summary.setdefault("warnings", []).append(msg + " (check skipped by skip_checks)")
        else:
            with open(os.path.join(out_dir, "ingest_report.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            raise IngestError(msg + " If you're sure the forces are right, add `skip_checks: [force_direction]` "
                                    "to the spec.")
    if fd["weak_axes"]:
        summary.setdefault("warnings", []).append(
            f"force and velocity barely move together on {', '.join(fd['weak_axes'])} ("
            + ", ".join(f"{a} {fd['correlation_force_vs_velocity'][a]:+.2f}" for a in fd["weak_axes"])
            + "; on an axis the thrusters drive they should be clearly positive). If the velocity is world-frame "
              "(linear_frame: world), the heading used to turn it is probably wrong (mirrored or offset); "
              "otherwise check the thruster layout and frame")
    if len(counters) == 1:
        summary.setdefault("warnings", []).append(
            f"every recording is from one session ({next(iter(counters))}). Calibration will hold out recordings "
            f"within it (or the last part of a single recording), which is an easier check than a session it never "
            f"saw: record a second session for the honest number (calibration/ONE_SESSION.md)")
    if spec.wrench.thrusters is not None:
        import thruster_model as tm
        cfg = _thruster_config(spec)
        summary["thrusters"] = tm.describe(cfg) + ("; thrust rescaled to each sample's logged voltage "
                                                   f"({spec.wrench.thrusters.voltage_column})"
                                                   if spec.wrench.thrusters.voltage_column else "")
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
    ap.add_argument("--no-next-step", action="store_true", help=argparse.SUPPRESS)  # inside `strathe calibrate`
    args = ap.parse_args(argv)
    try:
        spec = load_ingest_spec(args.spec)
        s = ingest(spec, args.out)
    except IngestError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    n_warn = sum(len(f["warnings"]) for f in s["files"]) + len(s.get("warnings", []))
    if s.get("thrusters"):
        print(f"Thrusters: {s['thrusters']}")
    vel = spec.body_velocity
    if vel is not None and vel.has_linear:
        print(f"Velocity from: {', '.join(vel.columns[k] for k in 'uvwpqr' if k in vel.columns)} ({vel.linear_frame} "
              f"frame" + (", rotated into the body frame with the orientation" if vel.linear_frame == "world" else "")
              + (", DVL gaps kept" if vel.linear_gaps == "keep" else "") + ")")
    for name, ch in s.get("sensors", {}).get("sensors", {}).items():
        noise = (", smoothed by the logging system, so its noise can't be measured from the log" if ch.get("filtered")
                 else (", noise " + "/".join(f"{x:.3g}" for x in ch["noise_std"])) if ch.get("noise_std") else "")
        print(f"Sensor {name} ({ch['kind']}): {ch['readings']} readings at {ch['rate_hz']} Hz{noise}, "
              f"{ch['coverage_pct']:.0f}% of samples")
    if "sensors" in s:
        seen = s["sensors"]["axes_seen_by"]
        print("Axes your sensors see: " + ", ".join(f"{a} ({' + '.join(b)})" for a, b in seen.items() if b)
              + (f"; not seen: {', '.join(s['sensors']['axes_not_seen'])}" if s["sensors"]["axes_not_seen"] else ""))
    c = s["force_direction"]["correlation_force_vs_velocity"]
    print("Force vs velocity, per axis (clearly positive where the thrusters drive the motion): "
          + ", ".join(f"{a} {v:+.2f}" for a, v in c.items() if v is not None))
    for w in s.get("warnings", []):
        print(f"WARNING: {w}")
    print(f"Ingested {s['n_files']} file(s)"
          + (f" into {s['n_recordings']} recording(s) after exclude_samples" if spec.exclude_samples else "")
          + f" at {s['sampling_rate_hz']} Hz across {len(s['sessions'])} session date(s) "
          f"-> {args.out} ({n_warn} warning(s); see ingest_report.json)")
    if not args.no_next_step:
        print(f"Next: {s['next_step']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
