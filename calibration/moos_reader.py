"""
moos_reader.py -- read the variables an ingest spec names straight out of MOOS-IvP logs (.alog, written by pLogger),
and put them on one uniform time grid (PROJECT_STATUS.md #94; calibration/LOG_FORMATS.md).

Used by data_ingestion.py when an ingest spec says `source: {format: moos}`. An .alog is text, so nothing extra is
needed. Each line is `<seconds since LOGSTART> <VARIABLE> <source app> <value>`; the header's `%% LOGSTART <unix
time>` gives the clock, so the session date can come from the recording.

Columns:
    NAV_SPEED                     a numeric variable
    NODE_REPORT_LOCAL.SPD         one key of a `key=value,key=value` posting (NODE_REPORT, DESIRED_*, your own)
    DESIRED_THRUST@pMarinePID     only what one app posted (when several post the same variable)
A string posting under a numeric column, or a key missing from a posting, is skipped and counted.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Sequence

import numpy as np

from rosbag_reader import series_to_grid

LOG_SUFFIXES = (".alog", ".ALOG")
COLUMN_KINDS = ("scalar", "key=value posting", "one app's postings")   # what a column reference can name
_REF = re.compile(r"^([A-Za-z0-9_]+)(?:\.([A-Za-z0-9_]+))?(?:@([A-Za-z0-9_\-]+))?$")


class MoosLogError(ValueError):
    pass


def split_reference(ref: str) -> tuple:
    """'NODE_REPORT_LOCAL.SPD@pNodeReporter' -> ('NODE_REPORT_LOCAL', 'SPD', 'pNodeReporter')."""
    m = _REF.match(ref)
    if not m:
        raise MoosLogError(f"{ref!r} isn't a MOOS log column: use VARIABLE, VARIABLE.key (a key=value posting) or "
                           f"VARIABLE@app, e.g. NAV_SPEED or NODE_REPORT_LOCAL.SPD")
    return m.group(1), m.group(2), m.group(3)


def _key(value: str, key: str) -> Optional[str]:
    for part in value.split(","):
        k, _, v = part.partition("=")
        if k.strip() == key:
            return v.strip()
    return None


def read_log(path: str, columns: Sequence[str], resample_hz: float, hold_columns: Sequence[str] = (),
             quaternion_groups: Sequence[Sequence[str]] = (), angle_columns: Optional[dict] = None) -> tuple:
    """Returns (DataFrame with the time column + one column per reference on a uniform grid, report)."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise MoosLogError(f"{path}: empty or missing log file")
    refs = {ref: split_reference(ref) for ref in dict.fromkeys(columns)}
    by_var: dict = {}
    for ref, (var, key, src) in refs.items():
        by_var.setdefault(var, []).append((ref, key, src))
    times = {ref: [] for ref in refs}
    vals = {ref: [] for ref in refs}
    start, seen, sources = None, {}, {}
    skipped = {"not a number": 0, "key missing": 0, "unreadable line": 0}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("%%"):
                m = re.match(r"%%\s*LOGSTART\s+([\d.]+)", line)
                if m:
                    start = float(m.group(1))
                continue
            parts = line.rstrip("\n").split(None, 3)
            if len(parts) < 4:
                if line.strip():
                    skipped["unreadable line"] += 1
                continue
            try:
                t = float(parts[0])
            except ValueError:
                skipped["unreadable line"] += 1
                continue
            var, src, value = parts[1], parts[2], parts[3].strip()
            seen[var] = seen.get(var, 0) + 1
            if var not in by_var:
                continue
            sources.setdefault(var, set()).add(src)
            for ref, key, want_src in by_var[var]:
                if want_src and src != want_src:
                    continue
                raw = value if key is None else _key(value, key)
                if raw is None:
                    skipped["key missing"] += 1
                    continue
                try:
                    v = float(raw)
                except ValueError:
                    skipped["not a number"] += 1
                    continue
                times[ref].append(t)
                vals[ref].append(v)
    if start is None:
        raise MoosLogError(f"{path}: no '%% LOGSTART' line: is this a MOOS .alog written by pLogger?")
    for ref, (var, key, src) in refs.items():
        if var not in seen:
            raise MoosLogError(f"{path}: variable {var!r} isn't in the log (it has: {sorted(seen)[:60]})")
        if not times[ref]:
            why = (f"no posting from {src!r} (posted by: {sorted(sources.get(var, []))})" if src
                   else f"no posting has the key {key!r}" if key else "no numeric posting")
            raise MoosLogError(f"{path}: {ref}: {why}")
    report = {"topics": {}, "warnings": []}
    for what, n in skipped.items():
        if n:
            report["warnings"].append(f"skipped {n} posting(s): {what}")
    # each column is its own source: MOOS variables post independently, at their own times
    series = {ref: (np.asarray(times[ref]) + start, {ref: vals[ref]}) for ref in refs}
    multi = [ref for ref, (var, _, src) in refs.items() if not src and len(sources.get(var, ())) > 1]
    if multi:
        report["warnings"].append(f"{', '.join(multi)}: posted by several apps ({', '.join(sorted(sources[refs[multi[0]][0]]))}); "
                                  f"pick one with VARIABLE@app")
    # quaternion components from one source: group them so series_to_grid slerps them together
    for g in quaternion_groups:
        if all(r in series for r in g):
            t0 = series[g[0]][0]
            if all(len(series[r][0]) == len(t0) and np.allclose(series[r][0], t0) for r in g):
                series[g[0]] = (t0, {r: series[r][1][r] for r in g})
                for r in g[1:]:
                    del series[r]
    df, report = series_to_grid(path, series, resample_hz, hold_columns, quaternion_groups, report,
                                {k: "MOOS log clock (LOGSTART + seconds)" for k in series}, MoosLogError,
                                angle_columns=angle_columns, source_word="variable")
    report["clock"] = "MOOS log clock (LOGSTART + seconds)"
    return df, report


def log_paths(pattern: str) -> list:
    import glob
    return [h for h in sorted(glob.glob(pattern)) if os.path.isfile(h) and h.endswith(LOG_SUFFIXES)]
