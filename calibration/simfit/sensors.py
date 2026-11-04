"""
simfit/sensors.py -- fit-sim and export-sim from a vehicle's own sensors (PROJECT_STATUS.md #96, SENSORS.md).

fit-sim replays the lab's simulator over windows of each recording, free-running from the vehicle's state at the start,
and scores it against the logs. With a DVL at every sample that's the logged velocity (core.py, as before). Without
one, this module:
- starts each window from the vehicle's estimated velocity (state_smoother.py: the DVL where it read, else a Kalman
  smoother over every sensor; no physics by default, since the simulator's model may be of any vehicle), and the
  logged gyro rates;
- scores the simulator's own motion against what each sensor read: its body rates against the gyro, its velocity
  against the DVL where it read, and its path, integrated with the logged attitude, against the position
  fixes and depth, each window's unknown starting position removed exactly (measurements.py);
- measures drift against the position fixes (or depth), not against a velocity.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

PENALTY = 10.0


def start_nu(run) -> np.ndarray:
    """(N, 6): the velocity each window starts from: anchors (u, v, w) and the gyro (p, q, r)."""
    return np.hstack([run.anchor_lin, run.real_nu[:, 3:6]])


def set_starts(runs: list, params=None, vehicle=None, noise: Optional[dict] = None) -> dict:
    """Every run's window starts (state_smoother.set_anchors), physics-free unless a Strathe calibration is given."""
    import state_smoother as sm
    sigma, _ = sm.choose_accel_sigma(runs, params, vehicle, noise) if runs else (0.1, None)
    return sm.set_anchors(runs, params, vehicle, sigma, noise)


def _pred(ch, V, dt, rel, R):
    from measurements import path_in_window, predicted
    path = path_in_window(V, R, dt) if ch.kind in ("position", "depth") else None
    return predicted(ch, V, R, path, dt, rel)


def window_parts(run, a: int, n: int, V: np.ndarray) -> dict:
    """{label: residuals} for one simulator window: each sensor's readings in (a, a+n) against the simulated vehicle,
    positions and depth taken about their window mean. Empty for a channel with too few readings."""
    from measurements import MIN_FIXES_PER_WINDOW
    # the logged attitude, as a calibration's path does (measurements.path_in_window): a simulator's own heading, off by
    # a few degrees after 30 s, would turn the path away from the fixes and take the blame out on surge
    R = Rotation.from_quat(run.orientation_quat[a:a + n]).as_matrix()
    out = {}
    for ch in run.channels:
        if ch.kind == "accelerometer":
            continue
        seg = ch.values[a:a + n]
        ok = np.all(np.isfinite(seg), axis=1)
        if ch.kind in ("gyro", "dvl"):
            ok[0] = False
        rel = np.flatnonzero(ok)
        if len(rel) < (1 if ch.kind in ("gyro", "dvl") else MIN_FIXES_PER_WINDOW):
            continue
        r = _pred(ch, V, run.dt, rel, R) - seg[rel]
        if ch.kind in ("position", "depth"):
            r = r - r.mean(axis=0)
        for j, lab in enumerate(ch.labels):
            out[lab] = r[:, j]
    return out


def still_parts(run, a: int, n: int) -> dict:
    """The same, for predicting no motion: what each label's residuals are scaled by."""
    out = {}
    for ch in run.channels:
        if ch.kind == "accelerometer":
            continue
        seg = ch.values[a:a + n]
        ok = np.all(np.isfinite(seg), axis=1)
        if ch.kind in ("gyro", "dvl"):
            ok[0] = False
        v = seg[ok]
        if not len(v) or (ch.kind in ("position", "depth") and len(v) < 2):
            continue
        if ch.kind in ("position", "depth"):
            v = v - v.mean(axis=0)
        for j, lab in enumerate(ch.labels):
            out[lab] = -v[:, j]
    return out


def scales(runs_wins: list) -> dict:
    pooled = {}
    for run, wins in runs_wins:
        for a, n in wins:
            for lab, r in still_parts(run, a, n).items():
                pooled.setdefault(lab, []).append(r)
    return {lab: max(float(np.sqrt(np.mean(np.concatenate(v) ** 2))), 1e-3) for lab, v in pooled.items()}


def layout(runs_wins: list) -> list:
    """Per window: [(label, length)], fixed, so the residual vector keeps its length at every evaluation."""
    return [[[(lab, len(r)) for lab, r in still_parts(run, a, n).items()] for a, n in wins] for run, wins in runs_wins]


def residuals(run, wins, preds, sc: dict, lay: list) -> list:
    """The optimizer's residuals for one recording's windows: each sensor axis over its still RMS and the square root
    of its readings in the window (a blown-up window: PENALTY in the same shape)."""
    out = []
    for (a, n), (V, _), lw in zip(wins, preds, lay):
        if np.isnan(V).any():
            out.append(np.full(sum(k for _, k in lw), PENALTY))
            continue
        parts = window_parts(run, a, n, V)
        for lab, k in lw:
            r = parts.get(lab)
            if r is None or len(r) != k:
                r = np.full(k, PENALTY)
            else:
                r = r / sc[lab] / math.sqrt(max(k, 1))
            out.append(r)
    return out


def _fix_channel(run):
    for ch in run.channels:
        if ch.kind == "position" and "n" in ch.axes:
            return ch, [("ned".index(x)) for x in ch.axes]
    for ch in run.channels:
        if ch.kind == "depth":
            return ch, [2]
    return None, []


def score(runs_wins: list, preds: list, horizons) -> dict:
    """core.score's numbers from the sensors: drift against the position fixes (or depth) nearest each horizon's end,
    velocity RMSE against the DVL where it read (NaN on u, v, w with no DVL) and the gyro, attitude against the logged
    orientation; and each sensor axis's own RMSE and skill (1 - RMSE / predicting no motion)."""
    from measurements import path_in_window
    vel = {h: [] for h in horizons}
    drift = {h: [] for h in horizons}
    att = {h: [] for h in horizons}
    sens, still = {}, {}
    total = diverged = 0
    for (run, wins), pr in zip(runs_wins, preds):
        ch, axes = _fix_channel(run)
        tol = max(1, int(round(0.5 / run.dt)))
        for (a, n), (V, Q) in zip(wins, pr):
            total += 1
            if np.isnan(V).any():
                diverged += 1
                continue
            R = Rotation.from_quat(Q).as_matrix()
            P = path_in_window(V, R, run.dt)
            rq = run.orientation_quat[a:a + n]
            e = np.degrees(np.abs((Rotation.from_quat(Q).as_euler("xyz") - Rotation.from_quat(rq).as_euler("xyz")
                                   + np.pi) % (2 * np.pi) - np.pi))
            truth_v = run.real_nu[a:a + n]
            for h in horizons:
                m = int(round(h / run.dt))
                if m >= n:
                    continue
                vel[h].append((V[1:m + 1] - truth_v[1:m + 1]) ** 2)
                att[h].append(e[m])
                if ch is not None:
                    vals = ch.values[a:a + n]
                    ok = np.all(np.isfinite(vals), axis=1)
                    k0 = next((k for k in range(0, min(tol + 1, n)) if ok[k]), None)
                    k1 = next((k for k in sorted(range(max(0, m - tol), min(n, m + tol + 1)), key=lambda k: abs(k - m))
                               if ok[k]), None)
                    if k0 is not None and k1 is not None and k1 > k0:
                        drift[h].append(float(np.linalg.norm((P[k1] - P[k0])[axes] - (vals[k1] - vals[k0]))))
            for lab, r in window_parts(run, a, n, V).items():
                sens.setdefault(lab, []).append(r)
            for lab, r in still_parts(run, a, n).items():
                still.setdefault(lab, []).append(r)
    out = {"windows": total, "diverged": diverged, "drift_against": None if not runs_wins else
           (lambda c: None if c[0] is None else c[0].name)(_fix_channel(runs_wins[0][0]))}
    for h in horizons:
        if att[h]:
            with np.errstate(invalid="ignore"):
                vr = np.sqrt(np.nanmean(np.concatenate(vel[h]), axis=0))
            # NaN, not None, where nothing measured it: the report formats it as nan and keeps going
            out[f"{h}s"] = {"drift_median_m": round(float(np.median(drift[h])), 3) if drift[h] else float("nan"),
                            "drift_p90_m": round(float(np.percentile(drift[h], 90)), 3) if drift[h] else float("nan"),
                            "velocity_rmse": [round(float(x), 4) if np.isfinite(x) else float("nan") for x in vr],
                            "attitude_error_median_deg": np.median(np.array(att[h]), axis=0).round(2).tolist()}
    out["sensors"] = {}
    for lab, rs in sens.items():
        rm = float(np.sqrt(np.mean(np.concatenate(rs) ** 2)))
        st = float(np.sqrt(np.mean(np.concatenate(still[lab]) ** 2))) if lab in still else float("nan")
        out["sensors"][lab] = {"rmse": round(rm, 5), "skill": round(1 - rm / st, 4) if st > 0 else None}
    return out
