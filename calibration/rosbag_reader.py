"""
rosbag_reader.py -- read the fields an ingest spec names straight out of a
ROS 1 (.bag) or ROS 2 (bag folder: .db3 or .mcap) recording, without a ROS
install, and put them on one uniform time grid.

Used by data_ingestion.py when an ingest spec says `source: {format: rosbag}`.
Columns are named `<topic>.<field path>`, for example
    /mavros/local_position/velocity_body.twist.linear.x
    /mavros/imu/data.orientation.w
    /mavros/rc/out.channels[2]
(ROS topic names can't contain '.', so everything after the first '.' is the
field path inside the message).

Topics arrive at different rates and times, so every column is interpolated
onto one grid at `resample_hz`, covering only the span where every topic has
data:
  - continuous signals (velocity, rates): linear interpolation
  - orientation quaternions: spherical interpolation (slerp), which stays a
    valid rotation even across the sudden jumps real IMU/odometry logs contain
    (the reference dataset has 239 jumps over 20 degrees in 0.1 s); their raw
    norms are reported so a wrong column mapping is still caught
  - thruster commands (PWM): held at the last command, as the ESC does
Timestamps come from each message's header.stamp when it has one (the time the
data was measured), otherwise from the time the bag recorded it.

Message definitions: ROS 1 bags and ROS 2 .mcap bags carry their own. ROS 2
.db3 bags don't, so non-standard types (e.g. mavros_msgs/msg/RCOut) need their
.msg files listed in `source.message_types`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

TIME_COLUMN = "__bag_time_s__"
_INDEX = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(\[(\d+)\])?$")


class BagError(ValueError):
    pass


def split_reference(ref: str) -> tuple:
    """'/a/b.twist.linear.x' -> ('/a/b', ['twist', 'linear', 'x'])."""
    if not ref.startswith("/") or "." not in ref:
        raise BagError(f"{ref!r} isn't a bag column: use <topic>.<field>, e.g. /mavros/imu/data.orientation.x")
    topic, path = ref.split(".", 1)
    parts = path.split(".")
    for p in parts:
        if not _INDEX.match(p):
            raise BagError(f"{ref!r}: can't parse field {p!r} (use names and [index], e.g. channels[0])")
    return topic, parts


def _field(msg, parts: Sequence[str], ref: str) -> float:
    obj = msg
    for p in parts:
        m = _INDEX.match(p)
        name, idx = m.group(1), m.group(3)
        if not hasattr(obj, name):
            have = [a for a in dir(obj) if not a.startswith("_")]
            raise BagError(f"{ref!r}: message has no field {name!r} (has: {have})")
        obj = getattr(obj, name)
        if idx is not None:
            i = int(idx)
            if i >= len(obj):
                raise BagError(f"{ref!r}: index {i} but the array has {len(obj)} element(s)")
            obj = obj[i]
    try:
        return float(obj)
    except (TypeError, ValueError):
        raise BagError(f"{ref!r} is a {type(obj).__name__}, not a number; name a numeric field") from None


def _stamp(msg) -> Optional[float]:
    h = getattr(msg, "header", None)
    st = getattr(h, "stamp", None) if h is not None else None
    if st is None:
        return None
    sec = getattr(st, "sec", None)
    nsec = getattr(st, "nanosec", None)
    if sec is None or nsec is None:
        return None
    t = float(sec) + float(nsec) * 1e-9
    return t if t > 0 else None


def _typestore(message_types: Sequence[str]):
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)
    add = {}
    for path in message_types:
        p = Path(path)
        if p.suffix != ".msg" or p.parent.name != "msg":
            raise BagError(f"message_types entry {path!r} should be a <package>/msg/<Name>.msg file")
        name = f"{p.parent.parent.name}/msg/{p.stem}"
        add.update(get_types_from_msg(p.read_text(encoding="utf-8"), name))
    if add:
        ts.register(add)
    return ts


def _open(path: str, message_types: Sequence[str]):
    from rosbags.highlevel import AnyReader
    p = Path(path)
    if p.is_file() and p.suffix in (".db3", ".mcap"):
        p = p.parent  # a ROS 2 bag is the folder with metadata.yaml
    if p.is_dir() and not (p / "metadata.yaml").exists():
        raise BagError(f"{path}: not a ROS 2 bag folder (no metadata.yaml)")
    return AnyReader([p], default_typestore=_typestore(message_types))


def read_bag(path: str, columns: Sequence[str], resample_hz: float, stamp: str = "header",
             hold_columns: Sequence[str] = (), quaternion_groups: Sequence[Sequence[str]] = (),
             message_types: Sequence[str] = ()) -> tuple:
    """Returns (DataFrame with TIME_COLUMN + one column per reference on a uniform grid, report)."""
    wanted: dict = {}
    for ref in dict.fromkeys(columns):
        topic, parts = split_reference(ref)
        wanted.setdefault(topic, []).append((ref, parts))
    series = {topic: ([], {ref: [] for ref, _ in refs}) for topic, refs in wanted.items()}
    report = {"topics": {}, "warnings": []}
    try:
        reader = _open(path, message_types)
        reader.open()
    except BagError:
        raise
    except Exception as e:  # rosbags raises its own error types for unreadable bags
        raise BagError(f"{path}: can't open as a ROS bag ({type(e).__name__}: {e})") from e
    try:
        present = {c.topic: c for c in reader.connections}
        missing = sorted(set(wanted) - set(present))
        if missing:
            raise BagError(f"{path}: topic(s) {missing} not in the bag (it has: {sorted(present)})")
        conns = [c for c in reader.connections if c.topic in wanted]
        header_used = {t: 0 for t in wanted}
        for conn, bag_ns, raw in reader.messages(connections=conns):
            try:
                msg = reader.deserialize(raw, conn.msgtype)
            except Exception as e:
                raise BagError(f"{path}: can't decode {conn.topic} ({conn.msgtype}): {e}. For a ROS 2 .db3 bag "
                               f"with custom message types, list their .msg files in source.message_types") from e
            t = _stamp(msg) if stamp == "header" else None
            if t is None:
                t = bag_ns * 1e-9
            else:
                header_used[conn.topic] += 1
            times, vals = series[conn.topic]
            times.append(t)
            for ref, parts in wanted[conn.topic]:
                vals[ref].append(_field(msg, parts, ref))
    finally:
        reader.close()

    frames = {}
    for topic, (times, vals) in series.items():
        t = np.asarray(times, dtype=float)
        if len(t) < 2:
            raise BagError(f"{path}: topic {topic} has {len(t)} message(s); need at least 2")
        order = np.argsort(t, kind="stable")
        t = t[order]
        keep = np.r_[True, np.diff(t) > 0]
        if not keep.all():
            report["warnings"].append(f"{topic}: dropped {int((~keep).sum())} message(s) with a repeated timestamp")
        t = t[keep]
        cols = {ref: np.asarray(v, dtype=float)[order][keep] for ref, v in vals.items()}
        dts = np.diff(t)
        med = float(np.median(dts))
        gaps = int(np.sum(dts > 2.5 * med))
        report["topics"][topic] = {"messages": int(len(t)), "rate_hz": round(1.0 / med, 3),
                                   "timestamps": "header.stamp" if stamp == "header" and header_used[topic]
                                   else "bag receive time", "gaps": gaps}
        if gaps:
            report["warnings"].append(f"{topic}: {gaps} gap(s) longer than 2.5 messages; values are interpolated "
                                      f"across them")
        frames[topic] = (t, cols)

    start = max(t[0] for t, _ in frames.values())
    end = min(t[-1] for t, _ in frames.values())
    if end - start < 10.0 / resample_hz:
        raise BagError(f"{path}: the topics overlap for only {max(end - start, 0):.2f} s; they need a common span")
    # start + k / rate, not np.arange(start, ...): arange steps by (start + step) - start, which at
    # Unix-epoch magnitudes is rounded (~2e-8 s) and drifts ~1e-5 s over a few minutes
    # (+1e-3: Unix-epoch stamps carry ~2e-7 s rounding, which must not drop the final sample)
    grid = start + np.arange(int(np.floor((end - start) * resample_hz + 1e-3)) + 1) / resample_hz
    grid = np.minimum(grid, end)
    out = {TIME_COLUMN: grid}
    quat_members = {r for g in quaternion_groups for r in g}
    for topic, (t, cols) in frames.items():
        for g in quaternion_groups:
            present = [r for r in g if r in cols]
            if not present:
                continue
            if len(present) != len(g):
                raise BagError(f"{path}: the quaternion's x, y, z, w must come from one topic; {topic} has only "
                               f"{present}")
            q = np.stack([cols[r] for r in g], 1)
            norms = np.linalg.norm(q, axis=1)
            report["quaternion_norm_range"] = [float(norms.min()), float(norms.max())]
            if np.any(norms < 1e-6):
                raise BagError(f"{path}: {topic} has all-zero quaternions (an IMU with no orientation estimate "
                               f"publishes zeros); use a topic that carries orientation")
            qi = slerp(t, q / norms[:, None], grid)
            # keep the raw norm range visible to the unit-quaternion check downstream
            for j, r in enumerate(g):
                out[r] = qi[:, j]
        for ref, v in cols.items():
            if ref in quat_members:
                continue
            if ref in hold_columns:
                # 1 us tolerance so a grid point that lands on a message's own stamp (to rounding) uses it
                idx = np.clip(np.searchsorted(t, grid + 1e-6, side="right") - 1, 0, len(t) - 1)
                out[ref] = v[idx]
            else:
                out[ref] = np.interp(grid, t, v)
    report["span_s"] = round(float(grid[-1] - grid[0]), 3)
    report["recording_start_unix_s"] = float(start)
    return pd.DataFrame(out), report


def slerp(t: np.ndarray, q: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Unit quaternions (x, y, z, w) at times t -> at grid times, by spherical interpolation."""
    from scipy.spatial.transform import Rotation, Slerp
    return Slerp(t, Rotation.from_quat(q))(np.clip(grid, t[0], t[-1])).as_quat()


def bag_paths(pattern: str) -> list:
    """Glob that returns ROS 1 .bag files and ROS 2 bag folders (or files inside them) as-is."""
    import glob
    hits = sorted(glob.glob(pattern))
    return [h for h in hits if os.path.isdir(h) or os.path.splitext(h)[1] in (".bag", ".db3", ".mcap")]
