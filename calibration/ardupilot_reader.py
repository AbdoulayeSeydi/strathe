"""
ardupilot_reader.py -- read the fields an ingest spec names straight out of
ArduPilot / ArduSub logs, and put them on one uniform time grid.

Used by data_ingestion.py when an ingest spec says `source: {format: ardupilot}`.
Two log types, both read with pymavlink (no ArduPilot tools needed):
  - DataFlash logs from the autopilot's SD card (`.bin`, also `.log`)
  - MAVLink telemetry logs a ground station records (`.tlog`, e.g. QGroundControl)

Columns are `<MESSAGE>.<field>`, with `[i]` to pick one instance of a
multi-instance message:
    RCOU.C1                     .bin: PWM sent to servo/motor output 1
    ATT.Roll                    .bin: roll, degrees
    IMU[0].GyrX                 .bin: first IMU's x rate, rad/s
    XKF1[0].VN                  .bin: EKF core 0 north velocity, m/s (world frame)
    SERVO_OUTPUT_RAW.servo1_raw .tlog: PWM on output 1
    ATTITUDE.roll               .tlog: roll, radians
    LOCAL_POSITION_NED.vx       .tlog: north velocity, m/s (world frame)

ArduPilot logs velocity in the world (NED) frame, not the body frame; set
`body_velocity.linear_frame: world` in the spec and ingestion rotates it
into the body frame with the logged attitude.

Every message type is put on one grid at `resample_hz` over the span every
type covers (series_to_grid in rosbag_reader.py): continuous values are
interpolated (angles unwrapped first), PWM is held at the last command.
Timestamps are the log's own clock: GPS-synchronised time when the log has
GPS, otherwise time since the autopilot booted (then give session dates with
`session_date: {from: filename}` or `{from: constant}`). Corrupt bytes, which
SD-card logs cut off at power-down often end with, are skipped and counted.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Sequence

import numpy as np

from rosbag_reader import series_to_grid

_REF = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)(\[(\d+)\])?\.([A-Za-z_][A-Za-z0-9_]*)$")
INSTANCE_FIELDS = ("I", "C", "Instance", "Id", "port")
LOG_SUFFIXES = (".bin", ".BIN", ".log", ".LOG", ".tlog", ".TLOG")


class ArduPilotLogError(ValueError):
    pass


def split_reference(ref: str) -> tuple:
    """'IMU[0].GyrX' -> ('IMU', 0, 'GyrX'); 'RCOU.C1' -> ('RCOU', None, 'C1')."""
    m = _REF.match(ref)
    if not m:
        raise ArduPilotLogError(f"{ref!r} isn't an ArduPilot log column: use <MESSAGE>.<field>, e.g. RCOU.C1, "
                                f"ATT.Roll or IMU[0].GyrX")
    return m.group(1), (None if m.group(3) is None else int(m.group(3))), m.group(4)


def _instance(msg) -> Optional[int]:
    fmt = getattr(msg, "fmt", None)
    name = getattr(fmt, "instance_field", None) if fmt is not None else None
    fields = getattr(msg, "_fieldnames", None) or []
    for f in ([name] if name else []) + [f for f in INSTANCE_FIELDS if f in fields]:
        v = getattr(msg, f, None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return None


def read_log(path: str, columns: Sequence[str], resample_hz: float, hold_columns: Sequence[str] = (),
             quaternion_groups: Sequence[Sequence[str]] = (), angle_columns: Optional[dict] = None) -> tuple:
    """Returns (DataFrame with the time column + one column per reference on a uniform grid, report)."""
    try:
        from pymavlink import mavutil
    except ImportError as e:  # pragma: no cover - pymavlink is in requirements.txt
        raise ArduPilotLogError("reading ArduPilot logs needs pymavlink (pip install -r requirements.txt)") from e
    wanted: dict = {}
    for ref in dict.fromkeys(columns):
        mtype, inst, field = split_reference(ref)
        wanted.setdefault((mtype, inst), []).append((ref, field))
    types = sorted({k[0] for k in wanted})
    series = {f"{t}[{i}]" if i is not None else t: ([], {ref: [] for ref, _ in refs})
              for (t, i), refs in wanted.items()}
    seen_types: dict = {}
    seen_instances: dict = {}
    report = {"topics": {}, "warnings": []}
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise ArduPilotLogError(f"{path}: empty or missing log file")
    import contextlib
    import io
    chatter = io.StringIO()  # pymavlink prints a line per damaged spot; count them instead of flooding the terminal
    try:
        with contextlib.redirect_stdout(chatter):
            conn = mavutil.mavlink_connection(path, dialect="ardupilotmega", robust_parsing=True)
    except Exception as e:
        raise ArduPilotLogError(f"{path}: can't open as an ArduPilot .bin or MAVLink .tlog log "
                                f"({type(e).__name__}: {e})") from e
    bad = 0
    boot_clock = True
    try:
        while True:
            try:
                with contextlib.redirect_stdout(chatter):
                    m = conn.recv_match(type=types + ["BAD_DATA"], blocking=False)
            except Exception as e:  # a corrupt record deep in the file: keep what was read
                report["warnings"].append(f"stopped reading at a corrupt record ({type(e).__name__}: {e})")
                break
            if m is None:
                break
            mtype = m.get_type()
            if mtype == "BAD_DATA":
                bad += 1
                continue
            seen_types[mtype] = seen_types.get(mtype, 0) + 1
            inst = _instance(m)
            seen_instances.setdefault(mtype, set()).add(inst)
            t = getattr(m, "_timestamp", None)
            if t is None:
                continue
            if t > 1e8:
                boot_clock = False
            for key in ((mtype, inst), (mtype, None)):
                if key not in wanted or (key[1] is None and inst is not None and (mtype, inst) in wanted):
                    continue
                name = f"{mtype}[{key[1]}]" if key[1] is not None else mtype
                times, vals = series[name]
                row = []
                for ref, field in wanted[key]:
                    if not hasattr(m, field):
                        have = [f for f in (getattr(m, "_fieldnames", None) or [])]
                        raise ArduPilotLogError(f"{path}: {mtype} has no field {field!r} (has: {have})")
                    try:
                        row.append((ref, float(getattr(m, field))))
                    except (TypeError, ValueError):
                        raise ArduPilotLogError(f"{path}: {ref} isn't a number") from None
                times.append(float(t))
                for ref, v in row:
                    vals[ref].append(v)
                break
    finally:
        try:
            conn.close()
        except Exception:
            pass
    missing = [t for t in types if t not in seen_types]
    if missing:
        raise ArduPilotLogError(f"{path}: message type(s) {missing} not in the log (it has: "
                                f"{sorted(seen_types)[:40]})")
    for (t, i), _ in wanted.items():
        name = f"{t}[{i}]" if i is not None else t
        if i is not None and not series[name][0]:
            raise ArduPilotLogError(f"{path}: {t} has no instance {i} (instances: "
                                    f"{sorted(x for x in seen_instances.get(t, set()) if x is not None)})")
        multi = [x for x in seen_instances.get(t, set()) if x is not None]
        if i is None and len(multi) > 1:
            raise ArduPilotLogError(f"{path}: {t} is logged for {len(multi)} instances {sorted(multi)}; pick one, "
                                    f"e.g. {t}[{min(multi)}].<field>")
    skipped = chatter.getvalue().count("Skipped")
    if bad or skipped:
        report["warnings"].append(f"skipped {bad + skipped} corrupt record(s) or damaged spot(s); the rest of the log "
                                  f"was used")
    if path.lower().endswith(".tlog"):
        clock = "ground-station recording clock"
    else:
        clock = "time since boot (no GPS time in the log)" if boot_clock else "log clock (GPS time)"
    df, report = series_to_grid(path, series, resample_hz, hold_columns, quaternion_groups, report,
                                {k: clock for k in series}, ArduPilotLogError, angle_columns=angle_columns,
                                source_word="message type")
    report["clock"] = clock
    if boot_clock:
        report["recording_start_unix_s"] = None
    return df, report


def log_paths(pattern: str) -> list:
    import glob
    return [h for h in sorted(glob.glob(pattern)) if os.path.isfile(h) and h.endswith(LOG_SUFFIXES)]
