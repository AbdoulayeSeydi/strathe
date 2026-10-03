#!/usr/bin/env python3
"""
prepare_dfki_tether.py -- turn the DFKI ROV-tether dataset into one CSV per recording, the shape
`strathe calibrate --ingest-spec` reads. A worked example for labs whose logs come as one CSV per
sensor, at different rates (PROJECT_STATUS.md #62).

    python prepare_dfki_tether.py <unzipped rov_tether_dataset_v1.2 folder> <out folder>

The dataset (Khanmeh, Wehbe, Simetti, Indiveri; Scientific Data 2025; Zenodo 10.5281/zenodo.17360027,
CC BY 4.0): a 6-thruster BlueROV2 piloted in DFKI's 23 x 19 x 8 m test basin, tracked by a 12-camera
Qualisys motion-capture system, with a tether whose pull at the surface is measured. One folder per
run, one CSV per source.

What this script does, and why each step is needed:
- One time base: everything is put on the autopilot's own ~10 Hz timestamps (rc-out.csv, the PWM
  ArduSub sent to each thruster). Strathe's CSV reader takes one table per recording.
- Velocity: from the origin marker's track in qtm_rov.csv (250 Hz, seen ~100% of the time), as the
  slope of a straight line fitted over +-0.1 s. velocity.csv (the motion-capture velocity streamed live)
  is the same quantity in the basin's frame, but it misses up to a third of each run; where both exist
  they agree to ~2 cm/s at zero lag. It is written in a NED-like world frame (x = the basin's x, z down);
  the ingest spec says `linear_frame: world`, so Strathe rotates it into the body frame with the
  attitude below. The origin marker sits on the vehicle's top, not at its centre of mass.
- Heading: from the motion-capture markers (back-right -> front-right is the body x axis).
  ArduSub's own yaw is mirrored relative to the basin's frame, offset by ~175 degrees and drifts by up
  to 15 degrees near the basin's steel (the paper says the same), so on its own it can't rotate the
  basin-frame velocity. The markers drop out often, though, so heading = ArduSub's yaw corrected by its
  offset from the markers, tracked where the markers are seen (5 s running median) and carried through
  marker gaps of up to MAX_HEADING_GAP_S. Roll and pitch come from ArduSub, which the paper calls
  continuous and precise (the markers drop out and flip roll).
- Body rates: ArduSub's ATTITUDE roll/pitch/yaw speeds (its gyros, FRD). The motion-capture angular
  velocity spikes to hundreds of rad/s when markers drop out.
- Sessions: the dataset has no recording dates. Calibration checks itself on whole sessions it never
  saw, so the runs are labelled as two sessions: the 11 limited-cable runs and the 3 free-piloting
  (unlimited-cable) runs, with placeholder dates 2025-01-01 and 2025-01-02. The default split then
  holds out the free-piloting runs.
- Samples more than MAX_HEADING_GAP_S from a marker heading are dropped by splitting the recording
  there.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

SESSIONS = {"limited cable": "2025-01-01", "unlimited cable": "2025-01-02"}  # placeholder dates, see above
MAX_HEADING_GAP_S = 5.0
OFFSET_WINDOW_S = 5.0
MIN_PIECE_S = 30.0


def marker_heading(qtm: pd.DataFrame) -> tuple:
    """(time, heading rad) where both right-side markers are seen; heading is the body x axis in the
    basin's frame, counter-clockwise from its x axis seen from above (the basin's z points up)."""
    fr = qtm[["rov_front_right_x", "rov_front_right_y"]].to_numpy()
    br = qtm[["rov_back_right_x", "rov_back_right_y"]].to_numpy()
    ok = np.isfinite(fr).all(1) & np.isfinite(br).all(1)
    d = fr[ok] - br[ok]
    return qtm["time"].to_numpy()[ok], np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def window_slope(t_src, x_src, t_out, half_width, min_samples=10):
    """Least-squares slope of x_src (n, k) against t_src over [t - half_width, t + half_width] for each t
    in t_out; NaN where fewer than min_samples samples fall in the window."""
    out = np.full((len(t_out), x_src.shape[1]), np.nan)
    lo, hi = np.searchsorted(t_src, t_out - half_width), np.searchsorted(t_src, t_out + half_width)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a >= min_samples:
            tt = t_src[a:b] - t_src[a:b].mean()
            out[i] = tt @ (x_src[a:b] - x_src[a:b].mean(0)) / (tt @ tt)
    return out


def window_mean(t_src, x_src, t_out, half_width):
    """Mean of x_src over [t - half_width, t + half_width) for each t in t_out (x_src may be 2-D)."""
    c = np.vstack([np.zeros((1,) + x_src.shape[1:]), np.cumsum(x_src, axis=0)])
    lo = np.searchsorted(t_src, t_out - half_width)
    hi = np.searchsorted(t_src, t_out + half_width)
    n = np.maximum(hi - lo, 1)[:, None] if x_src.ndim > 1 else np.maximum(hi - lo, 1)
    out = (c[hi] - c[lo]) / n
    out[hi == lo] = np.nan
    return out


def prepare_run(run_dir: str) -> list:
    """One run folder -> [DataFrame] (more than one when long marker dropouts split it)."""
    rc = pd.read_csv(os.path.join(run_dir, "rc-out.csv"))
    att = pd.read_csv(os.path.join(run_dir, "attitude.csv"))
    qtm = pd.read_csv(os.path.join(run_dir, "qtm_rov.csv"))
    ten = pd.read_csv(os.path.join(run_dir, "tension.csv")).sort_values("time")

    t = rc["time"].to_numpy()
    th, hd = marker_heading(qtm)
    keep = (t > max(th[0], att["time"].iloc[0]) + 0.1) & (t < min(th[-1], att["time"].iloc[-1]) - 0.1)
    # nearest heading sample: drop times more than MAX_HEADING_GAP_S from one
    k = np.clip(np.searchsorted(th, t), 1, len(th) - 1)
    gap = np.minimum(np.abs(t - th[k - 1]), np.abs(th[k] - t))
    keep &= gap <= MAX_HEADING_GAP_S
    dt = np.median(np.diff(t))
    origin = qtm[["rov_origin_x", "rov_origin_y", "rov_origin_z"]].to_numpy()
    seen = np.isfinite(origin).all(1)
    v_world = window_slope(qtm["time"].to_numpy()[seen], origin[seen], t, 0.1)
    keep &= np.isfinite(v_world).all(1)

    out = pd.DataFrame({"time_s": t})
    # basin frame (z up) -> NED-like world (x = basin x, y = -basin y, z down): a half turn about x
    out["vel_n"], out["vel_e"], out["vel_d"] = v_world[:, 0], -v_world[:, 1], -v_world[:, 2]
    out["roll_rad"] = np.interp(t, att["time"], att["roll"])
    out["pitch_rad"] = np.interp(t, att["time"], att["pitch"])
    # ArduSub yaw (clockwise from magnetic-ish north) = -heading + offset; track the offset where the
    # markers are seen and use it to correct ArduSub's yaw everywhere
    yaw_ap = np.unwrap(att["yaw"].to_numpy())
    off = np.unwrap(np.interp(th, att["time"], yaw_ap) + hd)
    step = max(1, int(round(OFFSET_WINDOW_S / np.median(np.diff(th)))))
    off_s = pd.Series(off).rolling(step, center=True, min_periods=1).median().to_numpy()
    yaw = np.interp(t, att["time"], yaw_ap) - np.interp(t, th, off_s)   # = -heading, clockwise from above
    out["yaw_rad"] = np.angle(np.exp(1j * yaw))
    out["p_rps"] = np.interp(t, att["time"], att["roll_speed"])
    out["q_rps"] = np.interp(t, att["time"], att["pitch_speed"])
    out["r_rps"] = np.interp(t, att["time"], att["yaw_speed"])
    for i in range(1, 7):
        out[f"pwm{i}"] = rc[f"channel{i}"].to_numpy()
    # for reference only (not used by calibration): the tether's pull, measured at the surface pulley
    tw = window_mean(ten["time"].to_numpy(), ten[["Tension_horizontal_Fy", "Tenion_vertical_Fx"]].to_numpy(),
                     t, dt / 2)
    out["tether_horizontal_n"], out["tether_vertical_n"] = tw[:, 0], tw[:, 1]

    # split at every dropped sample; keep pieces long enough to calibrate on
    pieces, start = [], None
    for i, ok in enumerate(np.append(keep, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            piece = out.iloc[start:i].reset_index(drop=True)
            if piece["time_s"].iloc[-1] - piece["time_s"].iloc[0] >= MIN_PIECE_S:
                pieces.append(piece)
            start = None
    return pieces


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    src, dest = argv
    os.makedirs(dest, exist_ok=True)
    n = 0
    for group, date in SESSIONS.items():
        gdir = os.path.join(src, group)
        if not os.path.isdir(gdir):
            print(f"not found: {gdir} (expected the unzipped rov_tether_dataset_v1.2 folder)")
            return 1
        for run in sorted(os.listdir(gdir)):
            pieces = prepare_run(os.path.join(gdir, run))
            for j, piece in enumerate(pieces):
                name = f"{date}_{group.split()[0]}_{run}" + (f"_part{j + 1}" if len(pieces) > 1 else "") + ".csv"
                piece.to_csv(os.path.join(dest, name), index=False)
                n += 1
                print(f"{name}: {len(piece)} samples, {piece['time_s'].iloc[-1] - piece['time_s'].iloc[0]:.0f} s")
    print(f"{n} recordings written to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
