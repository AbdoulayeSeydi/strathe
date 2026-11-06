"""
fin_check.py -- `strathe fin-check`: does a rudder do what the logs say, without calibrating anything (PROJECT_STATUS.md #97).

For logs that can't be calibrated (a glider has no propeller thrust) or before a long fit: each fin channel against the rate it
causes, at the short lag that fits best. Says whether the sign is right, how fast the vehicle responds, and how much it turns per
degree. Reads prepared recordings: `strathe ingest` makes them.
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
    if not files:
        raise FileNotFoundError(f"no recordings under {data}: prepare them with `strathe ingest`")
    if hz is None:
        rep = os.path.join(data, "ingest_report.json")
        hz = json.load(open(rep)).get("sampling_rate_hz") if os.path.exists(rep) else None
    if not hz:
        raise ValueError("give --sampling-rate-hz: there's no ingest_report.json to read it from")
    frames = [pd.read_csv(f) for f in files]
    cols = sorted({c for d in frames for c in d.columns if c.startswith(("fin_", "fincmd_"))})
    if not cols:
        raise ValueError(f"no fin channels in {data}: add a fins: block to the ingest spec (FINS.md)")
    out = {"recordings": len(files), "channels": {}}
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
