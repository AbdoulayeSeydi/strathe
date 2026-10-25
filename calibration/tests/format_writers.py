"""
tests/format_writers.py -- write recordings as PX4 (.ulg), MOOS-IvP (.alog) and LCM logs, for the readers' tests and
the round-trip proof (PROJECT_STATUS.md #94; stress/run_94.py). Test-only: Strathe reads these formats, it never
writes them.

The ULog writer follows PX4's published file format (docs.px4.io, "ULog File Format") and doesn't use pyulog, so
pyulog reading what it writes checks the writer independently of Strathe's reader.

A recording is given as topics: {name: Topic(times_s, {field: values}, rate tag, instance)}. Each topic keeps its own
times, as a vehicle's logs do (attitude faster than velocity, commands held between updates).
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Topic:
    times_s: np.ndarray                       # Unix seconds (or seconds since boot for a PX4 log without GPS)
    fields: dict                              # {field name: values}; "q[0]".."q[3]" for arrays
    instance: int = 0
    string_fields: dict = field(default_factory=dict)   # MOOS: {VAR: [strings]} posted as key=value text


# --- PX4 ULog ----------------------------------------------------------------------------------------------------------
_MAGIC = b"ULog\x01\x12\x35"


def _msg(kind: bytes, payload: bytes) -> bytes:
    return struct.pack("<HB", len(payload), kind[0]) + payload


def _ulog_fields(fields: dict) -> list:
    """{'q[0]': .., 'q[1]': .., 'vx': ..} -> [('float', 'q', 2), ('float', 'vx', 0)] (arrays grouped; double precision
    for the round trip, so nothing is lost to float32)."""
    out, seen = [], {}
    for name in fields:
        base, _, idx = name.partition("[")
        if idx:
            seen[base] = max(seen.get(base, 0), int(idx.rstrip("]")) + 1)
            if base not in [o[1] for o in out]:
                out.append(("double", base, None))
        else:
            out.append(("double", base, 0))
    return [(t, b, seen.get(b, 0)) for t, b, _ in out]


def write_ulog(path: str, topics: dict, utc_offset_s: Optional[float] = None, info: Optional[dict] = None,
               start_us: int = 0) -> None:
    """Topics' times are seconds since boot. With utc_offset_s, a sensor_gps topic carries UTC (time_utc_usec), as a
    log with a GPS fix does."""
    topics = dict(topics)
    if utc_offset_s is not None:
        any_t = next(iter(topics.values())).times_s
        t = np.linspace(any_t[0], any_t[-1], 20)
        topics["sensor_gps"] = Topic(t, {"time_utc_usec": (t + utc_offset_s) * 1e6})
    out = bytearray(_MAGIC + bytes([1]) + struct.pack("<Q", start_us))
    out += _msg(b"B", bytes(8) + bytes(8) + struct.pack("<QQQ", 0, 0, 0))
    for key, value in (info or {"sys_name": "PX4", "ver_hw": "STRATHE_TEST"}).items():
        v = value.encode()
        k = f"char[{len(v)}] {key}".encode()
        out += _msg(b"I", bytes([len(k)]) + k + v)
    layouts = {}
    for key, tp in topics.items():   # a key "name#n" logs the topic again as another instance
        name = key.split("#")[0]
        if name in layouts:
            continue
        fl = _ulog_fields(tp.fields) if name != "sensor_gps" else [("uint64_t", "time_utc_usec", 0)]
        spec = "uint64_t timestamp;" + "".join(f"{t}{f'[{n}]' if n else ''} {b};" for t, b, n in fl)
        out += _msg(b"F", f"{name}:{spec}".encode())
        layouts[name] = fl
    data = []
    for msg_id, (key, tp) in enumerate(topics.items()):
        name = key.split("#")[0]
        out += _msg(b"A", struct.pack("<BH", tp.instance, msg_id) + name.encode())
        fl = layouts[name]
        for k, t in enumerate(tp.times_s):
            row = struct.pack("<Q", int(round(t * 1e6)))
            for typ, base, n in fl:
                if typ == "uint64_t":
                    row += struct.pack("<Q", int(round(tp.fields[base][k])))
                elif n:
                    row += struct.pack(f"<{n}d", *(tp.fields[f"{base}[{i}]"][k] for i in range(n)))
                else:
                    row += struct.pack("<d", tp.fields[base][k])
            data.append((int(round(t * 1e6)), msg_id, row))
    for _, msg_id, row in sorted(data, key=lambda x: (x[0], x[1])):   # logged in time order, as PX4 does
        out += _msg(b"D", struct.pack("<H", msg_id) + row)
    open(path, "wb").write(bytes(out))


# --- MOOS-IvP alog -----------------------------------------------------------------------------------------------------
def write_alog(path: str, topics: dict, logstart: float, sources: Optional[dict] = None) -> None:
    """Each numeric field is its own MOOS variable (NAME = the field); string_fields post key=value text."""
    lines = []
    for name, tp in topics.items():
        src = (sources or {}).get(name, "pStratheTest")
        for f, vals in tp.fields.items():
            lines += [(t - logstart, f, src, repr(float(v))) for t, v in zip(tp.times_s, vals)]
        for var, texts in tp.string_fields.items():
            lines += [(t - logstart, var, src, s) for t, s in zip(tp.times_s, texts)]
    lines.sort(key=lambda x: x[0])
    with open(path, "w", encoding="utf-8") as f:
        f.write("%%------------------------------------------------\n%% LOG FILE:       strathe_test.alog\n"
                f"%% FILE OPENED ON  test\n%% LOGSTART           {logstart:.6f}\n"
                "%%------------------------------------------------\n")
        for t, var, src, val in lines:
            f.write(f"{t:.6f}   {var:<24} {src:<16} {val}\n")


# --- LCM ---------------------------------------------------------------------------------------------------------------
def lcm_type_text(package: str, name: str, fields: dict, with_utime: bool = True) -> str:
    """An .lcm definition for a channel's fields: doubles, arrays for name[i], and one level of nested types for
    sub.x (struct name_sub)."""
    arrays, scalars, nested = {}, [], {}
    for f in fields:
        if "." in f:
            sub, leaf = f.split(".", 1)
            nested.setdefault(sub, []).append(leaf)
            continue
        base, _, idx = f.partition("[")
        if idx:
            arrays[base] = max(arrays.get(base, 0), int(idx.rstrip("]")) + 1)
        elif f not in scalars:
            scalars.append(f)
    subs = "".join(f"struct {name}_{sub}\n{{\n" + "".join(f"    double {x};\n" for x in leaves) + "}\n\n"
                   for sub, leaves in nested.items())
    body = ("    int64_t utime;\n" if with_utime else "") + "".join(f"    double {s};\n" for s in scalars) + \
        "".join(f"    double {b}[{n}];\n" for b, n in arrays.items()) + \
        "".join(f"    {name}_{sub} {sub};\n" for sub in nested)
    return f"package {package};\n\n{subs}struct {name}\n{{\n{body}}}\n"


def write_lcm(path: str, types_dir: str, topics: dict, with_utime: bool = True, receive_delay_s: float = 0.0) -> list:
    """One channel per topic; writes the .lcm type files into types_dir and returns their paths. The message's own
    utime is the topic's time; the receive time is that plus receive_delay_s."""
    import importlib
    import sys
    import lcm_reader
    os.makedirs(types_dir, exist_ok=True)
    files = []
    for name, tp in topics.items():
        tname = name.lower()
        p = os.path.join(types_dir, f"{tname}.lcm")
        open(p, "w").write(lcm_type_text("strathe_test", tname, tp.fields, with_utime))
        files.append(p)
    types = lcm_reader.load_types(files)
    by_name = {cls.__name__: cls for cls in types.values()}
    import lcm
    log = lcm.EventLog(path, "w", overwrite=True)
    events = []
    for name, tp in topics.items():
        cls = by_name[name.lower()]
        for k, t in enumerate(tp.times_s):
            m = cls()
            if with_utime:
                m.utime = int(round(t * 1e6))
            arrays = {}
            for f, vals in tp.fields.items():
                if "." in f:   # a nested type's field
                    sub, leaf = f.split(".", 1)
                    if not isinstance(getattr(m, sub), by_name[f"{name.lower()}_{sub}"]):
                        setattr(m, sub, by_name[f"{name.lower()}_{sub}"]())
                    setattr(getattr(m, sub), leaf, float(vals[k]))
                    continue
                base, _, idx = f.partition("[")
                if idx:
                    arrays.setdefault(base, {})[int(idx.rstrip("]"))] = float(vals[k])
                else:
                    setattr(m, f, float(vals[k]))
            for b, d in arrays.items():
                setattr(m, b, [d[i] for i in range(len(d))])
            events.append((int(round((t + receive_delay_s) * 1e6)), name, m.encode()))
    for ts, ch, data in sorted(events, key=lambda e: e[0]):
        log.write_event(ts, ch, data)
    log.close()
    return files
