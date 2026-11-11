"""
stress/prepare_ooi_97.py -- OOI's REMUS logs as Strathe recordings, for the real rows of #97. Not shipped data: run it on extracts made from
rawdata.oceanobservatories.org, one file per mission. OOI data: credit the NSF and the WHOI OOI Program; its terms are in FINS.md's sources.

The state and fin messages are about 1 Hz and share their rows; the DVL's forward, starboard and vertical velocity is about 0.8 Hz and only
reads near the bottom. Recordings are the stretches running at depth with DVL lock, gaps under 5 s filled, at least 120 s long, at 1 Hz.
Rudder, elevator and aileron come as percent of full scale with the opposite sense to Strathe's, as fin_direction shows on these logs.
The depth sensor goes in as depth_pressure: a velocity fit doesn't read it, and validation scores depth against it rather than the DVL's
velocity integrated (the 2024 mission's DVL read vertical speed 1.5 times its depth sensor's rate, #97).
The aileron is the roll autopilot's (about 1.5% per degree of roll): each fin's own angle in the same message is the three mixed, as the y3
preset mixes them, so leaving it out hides the roll control from the fit (#97).
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

SRC = os.path.expanduser("~/strathe_fin_data/ooi_extract")
OUT = os.path.expanduser("~/strathe_fin_data/ooi_prepared")
FULL_SCALE_DEG = 15.0   # unknown for these fins: the fit's lift slope takes it (its limits go down to 0.1x for these logs, fin_rows_97.remus600_spec)
MIN_S, MAX_GAP_S = 120, 5


def mission(path: str, out: str) -> int:
    z = np.load(path)
    st, fm, dv = z["m1000"], z["m1089"], z["m1141"]
    n = min(len(st), len(fm))
    st, fm = st[:n], fm[:n]
    t = st[:, 0] + np.arange(n) * 1e-6
    keep = np.isfinite(dv[:, 7]) & (np.abs(dv[:, 7]) < 5) & (np.abs(dv[:, 8]) < 5) & (np.abs(dv[:, 9]) < 5)
    td = dv[keep, 0]
    grid = np.arange(t[0], t[-1], 1.0)
    gap = np.abs(grid[:, None] * 0 + 0)  # placeholder shape
    nearest = np.searchsorted(td, grid).clip(1, len(td) - 1)
    gap = np.minimum(np.abs(td[nearest] - grid), np.abs(td[nearest - 1] - grid))
    lin = np.stack([np.interp(grid, td, dv[keep, c]) for c in (7, 8, 9)], 1)
    it = lambda col, arr=st: np.interp(grid, t, arr[:, col])
    rpm, depth = it(1), it(2)
    pitch, roll = np.radians(it(4)), np.radians(it(5))
    psi = np.interp(grid, t, np.unwrap(np.radians(st[:, 6])))
    q_rate, p_rate = np.radians(it(8)), np.radians(it(9))
    r_rate = np.gradient(psi, grid)
    rud, ele, ail = (-it(c, fm) / 100 * np.radians(FULL_SCALE_DEG) for c in (3, 2, 4))
    ok = (gap <= MAX_GAP_S) & (rpm > 300) & (depth > 3)
    stem = os.path.basename(path)[:-4]
    day = dt.datetime.fromtimestamp(float(t[0]), dt.timezone.utc).strftime("%d-%m-%Y")
    start, runs = None, 0
    for i, k in enumerate(np.r_[ok, False]):
        if k and start is None:
            start = i
        elif not k and start is not None:
            if i - start >= MIN_S:
                s = slice(start, i)
                qq = Rotation.from_euler("ZYX", np.c_[psi[s], pitch[s], roll[s]]).as_quat()
                d = pd.DataFrame({"Fx": 0.0, "Fy": 0.0, "Fz": 0.0, "Tx": 0.0, "Ty": 0.0, "Tz": 0.0, "Bu": lin[s, 0], "Bv": lin[s, 1], "Bw": lin[s, 2],
                                  "Bp": p_rate[s], "Bq": q_rate[s], "Br": r_rate[s], "qx": qq[:, 0], "qy": qq[:, 1], "qz": qq[:, 2], "qw": qq[:, 3],
                                  "fin_rudder": rud[s], "fin_elevator": ele[s], "fin_aileron": ail[s], "rpm": rpm[s],
                                  "depth_pressure": depth[s]})
                runs += 1
                d.to_csv(f"{out}/{stem.replace('_', '')}-{day}-run{runs}.csv", index=False)
            start = None
    return runs


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for p in sorted(glob.glob(f"{SRC}/*.npz")):
        name = os.path.basename(p)[:-4]
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        print(name, mission(p, d), "recordings", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
