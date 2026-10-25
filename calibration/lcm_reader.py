"""
lcm_reader.py -- read the fields an ingest spec names straight out of LCM logs (lcm-logger's event logs), and put
them on one uniform time grid (PROJECT_STATUS.md #94; calibration/LOG_FORMATS.md).

Used by data_ingestion.py when an ingest spec says `source: {format: lcm, lcm_types: [my_types/*.lcm]}`. Needs the
`lcm` Python package (pip install lcm; it ships lcm-gen). Your message definitions (.lcm files) are turned into
Python with lcm-gen once and cached under ~/.strathe/lcm_types; each message is decoded by its type's fingerprint, so
the channel's name doesn't have to say its type.

Columns are `<CHANNEL>.<field>`, with `[i]` for array elements and dots into nested types:
    STATE_ESTIMATE.vel_body[0]        an array element
    THRUSTER_CMD.pwm[3]
    NAV.pose.orientation[2]           a field of a nested type
A channel name with dots in it works too: the longest channel in the log that the reference starts with is used.

Time: `stamp: header` (the default) uses the message's own `utime` field when it has one (microseconds, Unix time,
the LCM convention), else the time lcm-logger received it; `stamp: receive` always uses the receive time. Both are
Unix time, so the session date can come from the recording.
"""
from __future__ import annotations

import glob
import hashlib
import importlib
import os
import re
import shutil
import subprocess
import sys
from typing import Optional, Sequence

from rosbag_reader import series_to_grid

CACHE = os.path.join(os.path.expanduser("~"), ".strathe", "lcm_types")
COLUMN_KINDS = ("scalar", "array element", "nested type")   # what a column reference can name
_PART = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)((?:\[\d+\])*)")


class LcmLogError(ValueError):
    pass


def _lcm_gen() -> str:
    exe = shutil.which("lcm-gen") or os.path.join(os.path.dirname(sys.executable), "lcm-gen")
    if not os.path.exists(exe):
        raise LcmLogError("reading LCM logs needs lcm-gen, which comes with the lcm package: pip install lcm")
    return exe


def load_types(type_files: Sequence[str]) -> dict:
    """{packed fingerprint (8 bytes): class} for every struct in the .lcm files, generated once and cached."""
    paths = sorted({p for pat in type_files for p in glob.glob(os.path.expanduser(pat))})
    if not paths:
        raise LcmLogError(f"no LCM type files match {list(type_files)}: give your message definitions as "
                          f"source: {{lcm_types: [path/to/*.lcm]}}")
    h = hashlib.sha256(b"".join(open(p, "rb").read() for p in paths)).hexdigest()[:16]
    out = os.path.join(CACHE, h)
    if not os.path.isdir(out):
        tmp = out + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        r = subprocess.run([_lcm_gen(), "--python", "--ppath", tmp, *paths], capture_output=True, text=True)
        if r.returncode:
            shutil.rmtree(tmp, ignore_errors=True)
            raise LcmLogError(f"lcm-gen couldn't read your type files: {(r.stderr or r.stdout).strip()[-600:]}")
        os.replace(tmp, out)
    # generated modules are imported by their package names: a type set imported before with the same names (another
    # vehicle's, or an edited .lcm file) must not be reused, so those modules are dropped and this set's folder comes
    # first on the path
    pkgs = {n for n in os.listdir(out) if os.path.isdir(os.path.join(out, n))}
    for name in [m for m in sys.modules if m.split(".")[0] in pkgs]:
        del sys.modules[name]
    sys.path[:] = [out] + [p for p in sys.path if not p.startswith(CACHE)]
    importlib.invalidate_caches()
    types = {}
    for py in glob.glob(os.path.join(out, "**", "*.py"), recursive=True):
        rel = os.path.relpath(py, out)[:-3].replace(os.sep, ".")
        if rel.endswith("__init__"):
            continue
        mod = importlib.import_module(rel)
        cls = getattr(mod, rel.split(".")[-1], None)
        if cls is not None and hasattr(cls, "_get_packed_fingerprint"):
            types[cls._get_packed_fingerprint()] = cls
    if not types:
        raise LcmLogError(f"no message types in {paths}")
    return types


def _get(msg, path: str, ref: str):
    v = msg
    for name, idx in _PART.findall(path):
        if not hasattr(v, name):
            have = [s for s in getattr(v, "__slots__", [])]
            raise LcmLogError(f"{ref}: the message has no field {name!r} (has: {have})")
        v = getattr(v, name)
        for i in re.findall(r"\d+", idx):
            v = v[int(i)]
    return v


def read_log(path: str, columns: Sequence[str], resample_hz: float, hold_columns: Sequence[str] = (),
             quaternion_groups: Sequence[Sequence[str]] = (), angle_columns: Optional[dict] = None,
             type_files: Sequence[str] = (), stamp: str = "header") -> tuple:
    """Returns (DataFrame with the time column + one column per reference on a uniform grid, report)."""
    try:
        import lcm
    except ImportError as e:
        raise LcmLogError("reading LCM logs needs the lcm package: pip install lcm") from e
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise LcmLogError(f"{path}: empty or missing log file")
    types = load_types(type_files)
    try:
        log = lcm.EventLog(path, "r")
    except Exception as e:
        raise LcmLogError(f"{path}: can't open as an LCM log ({type(e).__name__}: {e})") from e
    events, counts, unknown, damaged = {}, {}, {}, 0
    while True:
        try:
            ev = log.read_next_event()
        except Exception:   # a damaged tail: keep what was read
            damaged += 1
            break
        if ev is None:
            break
        counts[ev.channel] = counts.get(ev.channel, 0) + 1
        events.setdefault(ev.channel, []).append((ev.timestamp, ev.data))
    log.close()
    channels = sorted(counts, key=len, reverse=True)
    wanted: dict = {}
    for ref in dict.fromkeys(columns):
        ch = next((c for c in channels if ref.startswith(c + ".")), None)
        if ch is None:
            raise LcmLogError(f"{path}: {ref!r} doesn't start with a channel in the log (channels: {sorted(counts)})")
        wanted.setdefault(ch, []).append((ref, ref[len(ch) + 1:]))
    series, clock = {}, {}
    report = {"topics": {}, "warnings": []}
    for ch, refs in wanted.items():
        t, vals = [], {ref: [] for ref, _ in refs}
        used_utime = None
        for ts, data in events[ch]:
            cls = types.get(data[:8])
            if cls is None:
                unknown[ch] = unknown.get(ch, 0) + 1
                continue
            try:
                msg = cls.decode(data)
            except Exception:
                unknown[ch] = unknown.get(ch, 0) + 1
                continue
            row = []
            for ref, field in refs:
                v = _get(msg, field, ref)
                try:
                    row.append(float(v))
                except (TypeError, ValueError):
                    raise LcmLogError(f"{path}: {ref} isn't a number") from None
            own = getattr(msg, "utime", None) if stamp == "header" else None
            used_utime = own is not None if used_utime is None else used_utime
            t.append((own if own is not None else ts) / 1e6)
            for (ref, _), v in zip(refs, row):
                vals[ref].append(v)
        if not t:
            raise LcmLogError(f"{path}: no message on {ch} matches your type files (its fingerprint isn't any "
                              f"type in them): give the .lcm file that defines it")
        series[ch] = (t, vals)
        clock[ch] = "the message's utime field" if used_utime else "lcm-logger's receive time"
    for ch, n in unknown.items():
        report["warnings"].append(f"{ch}: skipped {n} message(s) whose type isn't in your type files")
    if damaged:
        report["warnings"].append("the log ends in a damaged event; it was read up to there")
    df, report = series_to_grid(path, series, resample_hz, hold_columns, quaternion_groups, report, clock, LcmLogError,
                                angle_columns=angle_columns, source_word="channel")
    report["clock"] = ", ".join(sorted(set(clock.values())))
    return df, report


def log_paths(pattern: str) -> list:
    """Every file the pattern matches: lcm-logger names its logs lcmlog-YYYY-MM-DD.NN, with no fixed extension (and
    .lcm is also the extension of type definitions, so point `files:` at the logs, `lcm_types:` at the types)."""
    return [h for h in sorted(glob.glob(pattern)) if os.path.isfile(h)]
