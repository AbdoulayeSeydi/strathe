"""
ulog_reader.py -- read the fields an ingest spec names straight out of PX4 logs (.ulg), and put them on one uniform
time grid (PROJECT_STATUS.md #94; calibration/LOG_FORMATS.md).

Used by data_ingestion.py when an ingest spec says `source: {format: px4}`. Read with pyulog (pip install pyulog).

Columns are `<topic>.<field>`, with `[i]` after the topic to pick one instance of a multi-instance topic, and the
field as pyulog names it (arrays as `q[0]`, nested fields with dots):
    vehicle_attitude.q[0]                 attitude quaternion, w first (PX4: w, x, y, z), body FRD -> world NED
    vehicle_angular_velocity.xyz[0]       body roll rate, rad/s (FRD)
    vehicle_local_position.vx             north velocity, m/s (world NED)
    actuator_outputs[0].output[2]         what output 3 was sent (PWM, or normalized on newer PX4)
    actuator_motors.control[0]            normalized motor command, -1..1 (PX4 1.14+)

PX4 is forward-right-down / north-east-down already, so `frame: frd`. Its velocity is in the world frame: set
`body_velocity.linear_frame: world`. Its quaternion is w, x, y, z: map them by name in `orientation.quaternion`.

Time: each message's `timestamp` (microseconds since the autopilot booted). A log with GPS has its UTC time
(sensor_gps / vehicle_gps_position.time_utc_usec), and then the session date can come from the recording; underwater
there is usually none (`session_date: {from: filename}` or `{from: constant}`). A file cut off at power-down is read
up to the damage, and the report says so.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Sequence

import numpy as np

from rosbag_reader import series_to_grid

_REF = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)(?:\[(\d+)\])?\.(.+)$")
LOG_SUFFIXES = (".ulg", ".ULG")
COLUMN_KINDS = ("scalar", "array element", "multi-instance topic")   # what a column reference can name
GPS_TOPICS = ("sensor_gps", "vehicle_gps_position")


class ULogError(ValueError):
    pass


def split_reference(ref: str) -> tuple:
    """'actuator_outputs[0].output[2]' -> ('actuator_outputs', 0, 'output[2]'); 'vehicle_attitude.q[0]' ->
    ('vehicle_attitude', None, 'q[0]')."""
    m = _REF.match(ref)
    if not m:
        raise ULogError(f"{ref!r} isn't a PX4 log column: use <topic>.<field>, e.g. vehicle_attitude.q[0], or "
                        f"<topic>[instance].<field>, e.g. actuator_outputs[0].output[2]")
    return m.group(1), (None if m.group(2) is None else int(m.group(2))), m.group(3)


def _utc_offset_us(ulog) -> Optional[float]:
    """UTC minus boot time (µs) from the GPS topic, when the log has a fix."""
    for d in ulog.data_list:
        if d.name in GPS_TOPICS and "time_utc_usec" in d.data:
            utc, t = np.asarray(d.data["time_utc_usec"], float), np.asarray(d.data["timestamp"], float)
            ok = utc > 1e15   # a real date, not 0 before the first fix
            if ok.any():
                return float(np.median(utc[ok] - t[ok]))
    return None


def read_log(path: str, columns: Sequence[str], resample_hz: float, hold_columns: Sequence[str] = (),
             quaternion_groups: Sequence[Sequence[str]] = (), angle_columns: Optional[dict] = None) -> tuple:
    """Returns (DataFrame with the time column + one column per reference on a uniform grid, report)."""
    try:
        from pyulog import ULog
    except ImportError as e:
        raise ULogError("reading PX4 logs needs pyulog: pip install pyulog") from e
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise ULogError(f"{path}: empty or missing log file")
    wanted: dict = {}
    for ref in dict.fromkeys(columns):
        topic, inst, field = split_reference(ref)
        wanted.setdefault((topic, inst), []).append((ref, field))
    topics = sorted({t for t, _ in wanted} | set(GPS_TOPICS))
    try:
        ulog = ULog(path, topics)
    except Exception as e:
        raise ULogError(f"{path}: can't read as a PX4 log (.ulg) ({type(e).__name__}: {e})") from e
    report = {"topics": {}, "warnings": []}
    if getattr(ulog, "file_corruption", False):
        report["warnings"].append("the log is damaged (cut off at power-down?); it was read up to the damage")
    if ulog.dropouts:
        lost = sum(d.duration for d in ulog.dropouts)
        report["warnings"].append(f"the logger dropped data {len(ulog.dropouts)} time(s), {lost} ms in all; values are "
                                  f"interpolated across the gaps")
    present: dict = {}
    for d in ulog.data_list:
        present.setdefault(d.name, {})[d.multi_id] = d
    series, clock = {}, {}
    offset = _utc_offset_us(ulog)
    for (topic, inst), refs in wanted.items():
        if topic not in present:
            have = _all_topics(path)
            raise ULogError(f"{path}: topic {topic!r} isn't in the log (it has: {have[:60]})")
        ids = sorted(present[topic])
        if inst is None and len(ids) > 1:
            raise ULogError(f"{path}: {topic} is logged for {len(ids)} instances {ids}; pick one, e.g. "
                            f"{topic}[{ids[0]}].<field>")
        d = present[topic].get(ids[0] if inst is None else inst)
        if d is None:
            raise ULogError(f"{path}: {topic} has no instance {inst} (instances: {ids})")
        name = f"{topic}[{inst}]" if inst is not None else topic
        t = np.asarray(d.data["timestamp"], dtype=float)
        t = (t + offset) / 1e6 if offset is not None else t / 1e6
        vals = {}
        for ref, field in refs:
            if field not in d.data:
                fields = [f for f in d.data if f != "timestamp"]
                raise ULogError(f"{path}: {topic} has no field {field!r} (has: {fields[:40]})")
            v = np.asarray(d.data[field])
            if v.dtype.kind not in "biuf":
                raise ULogError(f"{path}: {ref} isn't a number")
            vals[ref] = v.astype(float)
        series[name] = (t, vals)
        clock[name] = "UTC (from the log's GPS)" if offset is not None else "time since boot (no GPS time in the log)"
    df, report = series_to_grid(path, series, resample_hz, hold_columns, quaternion_groups, report, clock, ULogError,
                                angle_columns=angle_columns)
    report["clock"] = next(iter(clock.values()))
    if offset is None:
        report["recording_start_unix_s"] = None
    return df, report


def _all_topics(path: str) -> list:
    from pyulog import ULog
    try:
        return sorted({d.name for d in ULog(path).data_list})
    except Exception:
        return []


def log_paths(pattern: str) -> list:
    import glob
    return [h for h in sorted(glob.glob(pattern)) if os.path.isfile(h) and h.endswith(LOG_SUFFIXES)]
