"""
fin_check.py -- `strathe fin-check`: does a rudder do what the logs say, without calibrating anything (PROJECT_STATUS.md #97).

For logs that can't be calibrated (a glider has no propeller thrust) or before a long fit: each fin channel against the rate it
causes, at the short lag that fits best. Says whether the sign is right, how fast the vehicle responds, and how much it turns per
degree. Reads prepared recordings (`strathe ingest` makes them), or a Slocum glider's logs as they come: `dbd2asc` text, .asc files.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from data_ingestion import fin_direction

AXIS = {"rudder": ("Br", "turn"), "elevator": ("Bq", "pitch")}
MAX_LAG_S = 3.0
SLOCUM_MIN_DEPTH_M = 2.0     # at the surface a glider is on its GPS and its fin does nothing useful
SLOCUM_GAP_SAMPLES = 3       # a longer gap than this many samples ends a stretch: a rate can't be worked out across it


def read_slocum(path: str) -> tuple:
    """A Slocum log in dbd2asc's text form: (stretches, hz). Each stretch is a frame on an even grid with fin_rudder and fincmd_rudder
    (m_fin, c_fin, rad) and Br, the turn rate from m_heading: gliders log no gyro rate. Submerged only."""
    with open(path, encoding="latin-1") as f:
        lines = f.read().splitlines()
    tags = next((int(x.split()[1]) for x in lines[:20] if x.startswith("num_ascii_tags:")), None)
    if tags is None or len(lines) < tags + 4:
        raise ValueError(f"{path}: not a Slocum dbd2asc file: no num_ascii_tags header")
    names = lines[tags].split()
    missing = [c for c in ("m_present_time", "m_fin", "m_heading", "m_depth") if c not in names]
    if missing:
        raise ValueError(f"{path}: this log has no {', '.join(missing)}: fin-check needs a log with the fin and heading in it")
    rows = np.array([[float(x) for x in l.split()] for l in lines[tags + 3:] if l.strip()], dtype=float)
    col = lambda c: rows[:, names.index(c)] if c in names else np.full(len(rows), np.nan)
    t, fin, hdg, depth = col("m_present_time"), col("m_fin"), col("m_heading"), col("m_depth")
    good = np.isfinite(t) & np.isfinite(fin) & np.isfinite(hdg)
    if good.sum() < 3:
        return [], None
    dt = float(np.median(np.diff(t[good])))
    hz = 1.0 / dt
    cmd = col("c_fin")
    keep = good & (np.interp(t, t[np.isfinite(depth)], depth[np.isfinite(depth)]) > SLOCUM_MIN_DEPTH_M)
    idx = np.flatnonzero(keep)
    out = []
    for part in np.split(idx, np.flatnonzero(np.diff(t[idx]) > SLOCUM_GAP_SAMPLES * dt) + 1):
        if len(part) < 3:
            continue
        grid = np.arange(t[part[0]], t[part[-1]], dt)
        if len(grid) < 3:
            continue
        heading = np.interp(grid, t[part], np.unwrap(hdg[part]))
        frame = {"fin_rudder": np.interp(grid, t[part], fin[part]), "Br": np.gradient(heading, dt)}
        c = np.isfinite(cmd[part])
        if c.sum() >= 2:
            frame["fincmd_rudder"] = np.interp(grid, t[part][c], cmd[part][c])
        out.append(pd.DataFrame(frame))
    return out, hz


def check_channel(angle: np.ndarray, rate: np.ndarray, hz: float, word: str, key: str) -> dict:
    """Sign, lag and gain of one fin channel against the rate it should cause."""
    t = np.arange(len(angle)) / hz
    r = fin_direction(t, rate, angle, np.ones(len(angle), dtype=bool), word, key)
    if "note" in r:
        return r
    lags = range(0, int(round(MAX_LAG_S * hz)) + 1)
    corr = [float(np.corrcoef(angle[:len(angle) - k or None], rate[k:])[0, 1]) for k in lags]
    k = int(np.argmax(np.abs(corr)))
    a, b = angle[:len(angle) - k or None], rate[k:]
    r["lag_s"] = round(k / hz, 2)
    r["rate_deg_s_per_deg"] = round(float(np.polyfit(a, b, 1)[0]), 4)
    return r


def check_folder(data: str, hz: Optional[float] = None) -> dict:
    files = sorted(glob.glob(os.path.join(data, "**", "*.csv"), recursive=True))
    slocum = sorted(glob.glob(os.path.join(data, "**", "*.asc"), recursive=True))
    if not files and not slocum:
        raise FileNotFoundError(f"no recordings under {data}: prepare them with `strathe ingest`, or give a Slocum glider's .asc logs")
    if slocum and files:
        raise ValueError(f"{data} has both prepared recordings and Slocum .asc logs: check them separately")
    frames = []
    if slocum:
        read = [read_slocum(f) for f in slocum]
        frames = [d for parts, _ in read for d in parts]
        rates = [r for _, r in read if r]
        if not frames:
            raise ValueError(f"no submerged stretch with a fin and a heading in the logs under {data}")
        hz = hz or float(np.median(rates))
    elif hz is None:
        rep = os.path.join(data, "ingest_report.json")
        hz = json.load(open(rep)).get("sampling_rate_hz") if os.path.exists(rep) else None
    if not hz:
        raise ValueError("give --sampling-rate-hz: there's no ingest_report.json to read it from")
    frames = frames or [pd.read_csv(f) for f in files]
    cols = sorted({c for d in frames for c in d.columns if c.startswith(("fin_", "fincmd_"))})
    if not cols:
        raise ValueError(f"no fin channels in {data}: add a fins: block to the ingest spec (FINS.md)")
    out = {"recordings": len(files or slocum), "channels": {}}
    for c in cols:
        ch = c.split("_", 1)[1]
        rate_col, word = AXIS.get(ch, ("Br", "turn"))
        parts = [d for d in frames if c in d.columns and rate_col in d.columns]
        if not parts:
            continue
        a = np.concatenate([d[c].to_numpy(dtype=float) for d in parts])
        r = np.concatenate([d[rate_col].to_numpy(dtype=float) for d in parts])
        key = {"rudder": "positive_turns", "elevator": "positive_pitches"}.get(ch, "the channel's sign")
        out["channels"][ch] = {**check_channel(a, r, float(hz), ch, key), "against": word,
                               "swing_deg": round(float(np.degrees(np.nanstd(a))), 2)}
    return out


def words(res: dict) -> list:
    L = [f"{res['recordings']} recording(s)."]
    for ch, r in res["channels"].items():
        if "note" in r:
            L.append(f"{ch}: {r['note']}.")
        elif "problem" in r:
            L.append(f"❌ {ch}: {r['problem']}.")
        else:
            L.append(f"✅ {ch}: the sign is right (correlation {r['correlation']:+.2f} with {r['against']} rate), the vehicle responds "
                     f"after {r['lag_s']:.1f} s, {r['rate_deg_s_per_deg']:.3f} per unit of fin angle.")
    return L
