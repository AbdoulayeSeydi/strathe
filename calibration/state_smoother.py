"""
state_smoother.py -- the vehicle's body velocity at every sample, estimated from whatever its sensors read
(PROJECT_STATUS.md #96, SENSORS.md).

Calibration replays each recording in short windows, and each window starts from a velocity. With a DVL reading at the
window's start that's the reading. Without one (no DVL, or a dropout) it comes from here: an extended Kalman filter run
forward through the recording, then a Rauch-Tung-Striebel smoother run back, so every estimate uses the readings both
before and after it.

    state:    body velocity u, v, w; and for each position or depth sensor its own position (n, e, d, or d alone),
              so sensors with different origins (a USBL's frame, GPS's first fix) don't have to agree
    process:  Strathe's own physics (fast_dynamics.FastDynamics, the same equations as the calibration's replays) with
              the current parameters, driven by the logged thrust, the logged roll and pitch and the logged gyro
              rates; position advances with the velocity turned by the logged orientation. Process noise: an
              acceleration the physics doesn't explain, accel_sigma (m/s^2) per axis
    readings: the DVL (body velocity), each position fix, each depth reading, with their noise

The smoother only supplies where each window starts. The calibration's residuals are always against the real sensor
readings (measurements.py), never against this estimate, so the fit isn't scored against itself. Calibration runs it
again with the fitted parameters and refits (--anchor-rounds): better physics, better starting velocities.

accel_sigma is chosen by the data: the value whose innovations are most likely over the training recordings, from a
log-spaced grid. A filter that blows up (a parameter set that makes the physics unstable) falls back to the DVL where
it read and zero elsewhere, and says so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

ACCEL_SIGMA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
DEFAULT_NOISE = {"dvl": 0.01, "position": 0.2, "depth": 0.02}
NOISE_FLOOR = {"dvl": 0.002, "position": 0.005, "depth": 0.002}
INITIAL_VELOCITY_STD = 0.3
UNKNOWN_POSITION_VAR = 1e8
POSITION_PROCESS_STD = 1e-3
JACOBIAN_STEP = 1e-4


@dataclass
class SmoothResult:
    velocity: np.ndarray            # (N, 3) smoothed body velocity
    loglik: float                   # log-likelihood of the innovations (for choosing accel_sigma)
    nis_mean: float                 # mean normalized innovation squared per reading (about 1 when consistent)
    n_readings: int
    accel_sigma: float
    fallback: Optional[str] = None  # why the estimate fell back to DVL-or-zero, if it did


def _blocks(channels) -> list:
    """[(channel index, kind, state axes)] for the channels that enter the state: DVL (velocity), position, depth."""
    out = []
    for ci, ch in enumerate(channels):
        if ch.kind == "position":
            out.append((ci, "position", [("ned".index(a)) for a in ch.axes]))
        elif ch.kind == "depth":
            out.append((ci, "depth", [2]))
        elif ch.kind == "dvl":
            out.append((ci, "dvl", None))
    return out


def smooth(run, params, vehicle, accel_sigma: float = 0.1, noise: Optional[dict] = None) -> SmoothResult:
    """Smoothed body velocity over one run (see module docstring). noise: {channel name: reading std (its units)}.
    params None: no physics, the velocity over ground held from sample to sample (a simulator's model of any
    vehicle, fit-sim)."""
    from dynamics_model import SimulationConfig
    from fast_dynamics import FastDynamics
    from objective import with_thruster_lag
    from scipy.spatial.transform import Rotation

    noise = noise or {}
    kinematic = params is None   # no physics (fit-sim on any vehicle): the velocity holds, the readings move it
    run_l = run if kinematic else with_thruster_lag(run, params)
    n, dt = run.n_samples, run.dt
    cfg = SimulationConfig(dt=dt)
    ns, sdt = cfg.n_substeps, cfg.sub_dt
    acc = None
    if not kinematic:
        try:
            dyn = FastDynamics(params, vehicle)
        except (ValueError, np.linalg.LinAlgError) as e:
            return _fallback(run, accel_sigma, f"the parameters don't give a valid mass matrix ({e})")
        acc = dyn.acceleration_trig
    R = Rotation.from_quat(run.orientation_quat).as_matrix()
    F = run_l.forces_torques.tolist()
    RP = run.roll_pitch.tolist()
    W = run.real_nu[:, 3:6].tolist()
    CB = None if run.current_body is None else np.asarray(run.current_body, dtype=float).tolist()
    CT = None if run_l.controls is None else np.asarray(run_l.controls, dtype=float).tolist()

    blocks = _blocks(run.channels)
    pos_blocks = [(ci, kind, ax) for ci, kind, ax in blocks if kind in ("position", "depth")]
    offs, dim = [], 3
    for _, _, ax in pos_blocks:
        offs.append(dim)
        dim += len(ax)

    def sigma_of(ch):
        s = noise.get(ch.name, DEFAULT_NOISE[ch.kind])
        return max(float(np.max(np.atleast_1d(s))), NOISE_FLOOR[ch.kind])

    def propagate(v, k):
        if kinematic:   # the velocity over ground holds; in the body it turns as the vehicle turns
            return R[k + 1].T @ (R[k] @ np.asarray(v, dtype=float)) if k + 1 < n else np.array(v, dtype=float)
        tau = F[k]
        roll, pitch = RP[k]
        sr, cr, sp, cp = math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch)
        w = W[k]
        cb = None if CB is None else CB[k]
        ctl = None if CT is None else CT[k]
        x = [v[0], v[1], v[2], w[0], w[1], w[2]]
        for _ in range(ns):
            vd = acc(x, tau, sr, cr, sp, cp, cb, ctl)
            x = [x[0] + vd[0] * sdt, x[1] + vd[1] * sdt, x[2] + vd[2] * sdt, w[0], w[1], w[2]]
        return np.array(x[:3])

    x = np.zeros(dim)
    P = np.zeros((dim, dim))
    P[:3, :3] = np.eye(3) * INITIAL_VELOCITY_STD ** 2
    for o, (_, _, ax) in zip(offs, pos_blocks):
        P[o:o + len(ax), o:o + len(ax)] = np.eye(len(ax)) * UNKNOWN_POSITION_VAR
    dvl = next((run.channels[ci] for ci, kind, _ in blocks if kind == "dvl"), None)
    if dvl is not None and np.all(np.isfinite(dvl.values[0])):
        x[:3] = dvl_over_ground(run, dvl)[0]
    q_pos = POSITION_PROCESS_STD ** 2

    xf = np.empty((n, dim))
    Pf = np.empty((n, dim, dim))
    xp = np.empty((n, dim))
    Pp = np.empty((n, dim, dim))
    Fk = np.empty((n, dim, dim))
    loglik, nis, nread = 0.0, 0.0, 0
    I = np.eye(dim)
    sig = {ci: sigma_of(run.channels[ci]) for ci, _, _ in blocks}
    for k in range(n):
        if k > 0:
            v = x[:3]
            try:
                f0 = propagate(v, k - 1)
                J = np.empty((3, 3))
                for j in range(3):
                    dv = v.copy()
                    dv[j] += JACOBIAN_STEP
                    J[:, j] = (propagate(dv, k - 1) - f0) / JACOBIAN_STEP
            except (OverflowError, ValueError, ZeroDivisionError) as e:
                return _fallback(run, accel_sigma, f"the physics blew up at sample {k} ({type(e).__name__})")
            if not (np.all(np.isfinite(f0)) and np.all(np.isfinite(J))):
                return _fallback(run, accel_sigma, f"the physics blew up at sample {k}")
            Fm = np.eye(dim)
            Fm[:3, :3] = J
            vw = R[k - 1] @ v
            xn = x.copy()
            xn[:3] = f0
            for o, (_, _, ax) in zip(offs, pos_blocks):
                xn[o:o + len(ax)] += dt * vw[ax]
                Fm[o:o + len(ax), :3] = dt * R[k - 1][ax, :]
            Q = np.zeros((dim, dim))
            Q[:3, :3] = np.eye(3) * accel_sigma ** 2 * dt
            for o, (_, _, ax) in zip(offs, pos_blocks):
                Q[o:o + len(ax), o:o + len(ax)] = np.eye(len(ax)) * q_pos
            x = xn
            P = Fm @ P @ Fm.T + Q
            Fk[k - 1] = Fm
        xp[k], Pp[k] = x, P
        # the readings at this sample
        rows, z, rv = [], [], []
        for (ci, kind, ax) in blocks:
            val = run.channels[ci].values[k]
            if not np.all(np.isfinite(val)):
                continue
            s2 = sig[ci] ** 2
            if kind == "dvl":
                ch = run.channels[ci]
                over_ground = val
                if ch.track == "water" and run.current_body is not None:
                    over_ground = val + run.current_body[k]   # the state is over ground
                if ch.lever_arm_m is not None:   # the DVL reads v + w x r: take the rotation's part off
                    over_ground = over_ground - np.cross(run.real_nu[k, 3:6], ch.lever_arm_m)
                for j in range(3):
                    h = np.zeros(dim)
                    h[j] = 1.0
                    rows.append(h); z.append(over_ground[j]); rv.append(s2)
            else:
                o = offs[[b[0] for b in pos_blocks].index(ci)]
                for j in range(len(ax)):
                    h = np.zeros(dim)
                    h[o + j] = 1.0
                    rows.append(h); z.append(val[j]); rv.append(s2)
        if rows:
            H = np.array(rows)
            innov = np.array(z) - H @ x
            S = H @ P @ H.T + np.diag(rv)
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return _fallback(run, accel_sigma, f"a singular update at sample {k}")
            K = P @ H.T @ Sinv
            x = x + K @ innov
            IKH = I - K @ H
            P = IKH @ P @ IKH.T + K @ np.diag(rv) @ K.T   # Joseph form: stays symmetric positive
            # an unknown position's first reading isn't a surprise: only readings of a known state count
            known = np.array([np.diag(Pp[k])[np.flatnonzero(h)[0]] < 0.1 * UNKNOWN_POSITION_VAR for h in rows])
            if known.any():
                Sk = S[np.ix_(known, known)]
                ik = innov[known]
                sol = np.linalg.solve(Sk, ik)
                loglik += -0.5 * (float(ik @ sol) + float(np.linalg.slogdet(2 * np.pi * Sk)[1]))
                nis += float(ik @ sol)
                nread += int(known.sum())
        xf[k], Pf[k] = x, P
    # Rauch-Tung-Striebel, backward
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        try:
            C = Pf[k] @ Fk[k].T @ np.linalg.inv(Pp[k + 1])
        except np.linalg.LinAlgError:
            C = np.zeros((dim, dim))
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
    vel = xs[:, :3]
    if not np.all(np.isfinite(vel)):
        return _fallback(run, accel_sigma, "the smoothed estimate isn't finite")
    return SmoothResult(velocity=vel, loglik=loglik, nis_mean=(nis / nread) if nread else float("nan"),
                        n_readings=nread, accel_sigma=accel_sigma)


def dvl_over_ground(run, ch) -> np.ndarray:
    """A DVL's readings as the body velocity over ground at the reference point (what a window starts from): a
    water-tracking DVL plus the current, minus the rotation's part at the DVL's mount. NaN where it didn't read."""
    v = np.array(ch.values, dtype=float)
    if ch.track == "water" and run.current_body is not None:
        v = v + run.current_body
    if ch.lever_arm_m is not None:
        v = v - np.cross(run.real_nu[:, 3:6], ch.lever_arm_m)
    return v


def _fallback(run, accel_sigma, why) -> SmoothResult:
    v = np.zeros((run.n_samples, 3))
    dvl = next((ch for ch in run.channels if ch.kind == "dvl"), None)
    if dvl is not None:
        ok = np.all(np.isfinite(dvl.values), axis=1)
        v[ok] = dvl_over_ground(run, dvl)[ok]
    return SmoothResult(velocity=v, loglik=float("-inf"), nis_mean=float("nan"), n_readings=0,
                        accel_sigma=accel_sigma, fallback=why)


# a late position sensor (#96): the fixes shifted earlier in time until the smoother, with the description's physics
# driven by the logged thrust, explains them best. Measured through calibrate: fixes 1 s late on a known truth were
# found 0.8 s late but gained only 1.2 per fix (B-latency-1), under the warning's bar, so the report says it without
# warning; real fixes with no delay (the tank's and DFKI's, from motion capture) were found 0.0 s late; URI's real GPS
# read 2 s, gaining 0.05 and 0.9 per fix, not confirmed by its DVL
DELAY_SCAN_S = (0.0, 0.5, 1.0, 1.5, 2.0)
DELAY_REFINE_S = 0.2
DELAY_WINDOW_S = 60.0
DELAY_SAID_S = 0.5            # said from this delay
# and only when it explains each fix this much better (log-likelihood) than no delay. First 2.0, which B-latency-1
# failed (1.21); 0.5 was then chosen from the numbers above and tested on fresh recordings (B-latency-fresh-*)
DELAY_GAIN_PER_FIX = 0.5
# a late depth sensor, the same way (2026-11-03). Depth is logged at 1 to 10 Hz, so its bar is per second of the
# window, not per reading. Chosen from the scan alone on the reference recordings, depth late by 0 / 0.3 / 0.5 / 1 /
# 2 s with full, horizontal or no fixes: on time 0.0 per second every time; 0.5 s late 0.67-0.97; and from real depth
# that's on time: the tank and DFKI's Bar30 0.0 s, URI's 1 Hz depth 0.5 s at 0.003 per second. Half the weakest late
# case, tested on fresh recordings (B-depthlag-*)
DELAY_GAIN_PER_S_DEPTH = 0.3
DELAY_KINDS = ("position", "depth")
# Two or more training runs (#98): every run's best DELAY_WINDOW_S window is scored and the log-likelihoods added, so
# a small delay that each run shows weakly adds up. Chosen on the 12 Jan known truth (16 recordings, 3 seeds) and the
# real on-time sensors (the tank's, DFKI's and KTH's motion capture and depth, URI's GPS and depth), before any proof
# row ran (PROJECT_STATUS.md #98): on time, the pooled best was 0.0 s with no gain in every case from 4 runs (the
# largest false gain anywhere, URI's real depth from 2 runs, 0.4); USBL-like 1 Hz fixes late by 0.3 / 0.4 s were
# found 0.2 / 0.3 s late with a pooled gain of 6.5-146, depth late by 0.1-0.3 s found exactly with 1.3-142.
DELAY_FINE_STEP_S = 0.1       # one sample at 10 Hz: finer than that stays `delay_s:`
DELAY_FINE_SPAN_S = 0.4       # fine steps this far either side of the coarse scan's best
DELAY_MAX_RUNS = 8            # the runs with the most readings: 8 already separates 0.3 s from none, at half the cost
DELAY_POOLED_GAIN = 3.0       # pooled log-likelihood gain over no delay (7x the largest false gain seen)
DELAY_FLOOR_S = {"position": 0.2, "depth": 0.1}   # the smallest found delay taken off
DELAY_LOO_AGREE_S = 0.2       # leaving out the run that gains most, the rest must find the delay within this
# Recordings that disagree (after #98's first proof run): how much better each run fits with its own best delay than
# with the pooled one, per run. On every consistent known truth in A0 (0 to 0.4 s late, 3 seeds, fixes and depth) at
# most 1.01; on DFKI's real Bar30, logged apart from its motion capture and merged by time, 3.86 (its 8 longest runs)
# and 4.97 (all 57). Over this, one delay doesn't fit them, so none is taken off and the report says so. Chosen on A0's
# numbers after D-real-dfki failed (PROJECT_STATUS.md #98)
DELAY_DISAGREE_PER_RUN = 2.0
# Fixes further apart than this (s, on average over the scanned windows) are too slow to show a delay near the floor:
# A0 measured the floor at 1 Hz, and at 0.2 Hz a 0.3 s delay wasn't seen even over 600 s windows (#98 fix
# experiment), so with nothing taken off the report says to give `delay_s:`.
DELAY_SLOW_FIX_S = 1.5


def _window(r, name: str) -> tuple:
    """(start sample, readings) of the DELAY_WINDOW_S window of run r with the most of sensor `name`'s readings."""
    ch = next((c for c in (r.channels or []) if c.name == name), None)
    if ch is None:
        return 0, 0
    L = min(r.n_samples, int(round(DELAY_WINDOW_S * r.sample_rate_hz)))
    has = np.isfinite(ch.values[:, 0]).astype(float)
    counts = np.convolve(has, np.ones(L), "valid")
    a = int(np.argmax(counts))
    return a, int(counts[a])


def _loglik(r, name: str, a: int, delay: float, params, vehicle, accel_sigma: float, noise) -> float:
    from dataclasses import replace
    from measurements import slice_run
    L = min(r.n_samples, int(round(DELAY_WINDOW_S * r.sample_rate_hz)))
    k = int(round(delay * r.sample_rate_hz))
    chans = [replace(c, values=shifted_values(c.values, k)) if c.name == name else c for c in r.channels]
    res = smooth(slice_run(replace(r, channels=chans), a, a + L), params, vehicle, accel_sigma, noise)
    return res.loglik if res.fallback is None else float("-inf")


def _best(lls: Sequence[dict]) -> float:
    """The delay with the highest summed log-likelihood over these runs' scans (all on the same grid)."""
    tot = {d: sum(ll[d] for ll in lls) for d in lls[0]}
    return max(tot, key=tot.get)


def shifted_values(values: np.ndarray, k: int) -> np.ndarray:
    """Readings k samples earlier: a fix logged at sample i+k is where the vehicle was at i (k < 0: later)."""
    if not k:
        return values
    v = np.full_like(values, np.nan)
    if k > 0:
        v[:-k] = values[k:]
    else:
        v[-k:] = values[:k]
    return v


def shift_fixes(runs: Sequence, name: str, delay_s: float) -> None:
    """Take delay_s off position or depth sensor `name`'s times in every run (in place). The shift so far is kept on
    the channel, so shifting a run again to the same delay (a reloaded calibration) changes nothing."""
    from dataclasses import replace
    for r in runs:
        if not r.channels:
            continue
        for i, c in enumerate(r.channels):
            if c.name != name or c.kind not in DELAY_KINDS:
                continue
            k = int(round((delay_s - c.shifted_s) * r.sample_rate_hz))
            if k:
                r.channels[i] = replace(c, values=shifted_values(c.values, k), shifted_s=float(delay_s))
                r.__dict__.pop("_plans", None)   # measurements.plan_for's windows were built from the old fixes


def fix_delays(runs: Sequence, params, vehicle, accel_sigma: float, noise: Optional[dict] = None) -> dict:
    """{position or depth sensor: {kind, delay_s, gain_per_fix, gain_per_s, fixes, late, ...}}: how late each sensor's
    readings look. With two or more runs that have at least 10 of its readings, pooled over up to DELAY_MAX_RUNS of
    them (_pooled_delay); with one, from that run's best window, as before #98 (_one_run_delay). fixes and gain_per_fix
    count readings, a depth sensor's too."""
    out = {}
    kinds = {ch.name: ch.kind for r in runs for ch in (r.channels or []) if ch.kind in DELAY_KINDS}
    for name in sorted(kinds):
        usable = [(r, *_window(r, name)) for r in runs]
        usable = sorted([u for u in usable if u[2] >= 10], key=lambda u: -u[2])[:DELAY_MAX_RUNS]
        if len(usable) >= 2:
            out[name] = _pooled_delay(name, kinds[name], usable, params, vehicle, accel_sigma, noise)
        else:
            out[name] = _one_run_delay(name, kinds[name], runs, params, vehicle, accel_sigma, noise)
    return out


def _pooled_delay(name: str, kind: str, usable: list, params, vehicle, accel_sigma: float, noise) -> dict:
    """Every usable run scored on one grid: the coarse DELAY_SCAN_S, then DELAY_FINE_STEP_S steps within
    DELAY_FINE_SPAN_S of its pooled best. late: the pooled best is at least DELAY_FLOOR_S[kind], gains at least
    DELAY_POOLED_GAIN over no delay, and the runs left after dropping the one that gains most find it within
    DELAY_LOO_AGREE_S (so no single run, or single unlucky window, decides it)."""
    lls = [{d: _loglik(r, name, a, d, params, vehicle, accel_sigma, noise) for d in DELAY_SCAN_S} for r, a, _ in usable]
    if not all(np.isfinite(v) for ll in lls for v in ll.values()):
        return {"kind": kind, "note": "the smoother failed on a delay tried, so no delay can be told"}
    c = _best(lls)
    top = max(DELAY_SCAN_S)
    fine = sorted({round(c + i * DELAY_FINE_STEP_S, 3) for i in range(-int(round(DELAY_FINE_SPAN_S / DELAY_FINE_STEP_S)),
                                                                     int(round(DELAY_FINE_SPAN_S / DELAY_FINE_STEP_S)) + 1)
                   if 0.0 <= c + i * DELAY_FINE_STEP_S <= top + 1e-9} - set(DELAY_SCAN_S))
    for (r, a, _), ll in zip(usable, lls):
        for d in fine:
            ll[d] = _loglik(r, name, a, d, params, vehicle, accel_sigma, noise)
    if not all(np.isfinite(v) for ll in lls for v in ll.values()):
        return {"kind": kind, "note": "the smoother failed on a delay tried, so no delay can be told"}
    best = _best(lls)
    n = sum(u[2] for u in usable)
    secs = sum(min(r.n_samples, int(round(DELAY_WINDOW_S * r.sample_rate_hz))) / r.sample_rate_hz for r, _, _ in usable)
    total = float(sum(ll[best] - ll[0.0] for ll in lls))
    gains = [ll[best] - ll[0.0] for ll in lls]
    rest = [ll for i, ll in enumerate(lls) if i != int(np.argmax(gains))]
    loo = _best(rest)
    disagree = float(sum(max(ll.values()) - ll[best] for ll in lls) / len(lls))
    late = bool(best >= DELAY_FLOOR_S[kind] - 1e-9 and total >= DELAY_POOLED_GAIN
                and abs(loo - best) <= DELAY_LOO_AGREE_S + 1e-9 and disagree <= DELAY_DISAGREE_PER_RUN)
    return {"kind": kind, "delay_s": round(float(best), 2), "gain_per_fix": round(total / n, 3),
            "gain_per_s": round(total / secs, 3), "fixes": int(n), "at_the_scan_edge": bool(best >= top),
            "late": late, "pooled": True, "runs": len(usable), "gain_total": round(total, 2),
            "loo_delay_s": round(float(loo), 2), "floor_s": DELAY_FLOOR_S[kind],
            "disagreement_per_run": round(disagree, 2),
            "own_delays_s": sorted(round(float(max(ll, key=ll.get)), 2) for ll in lls),
            "fix_interval_s": round(secs / n, 2) if n else None}


def _one_run_delay(name: str, kind: str, runs: Sequence, params, vehicle, accel_sigma: float, noise) -> dict:
    """One run with enough readings: its best window, scanned as before #98 (DELAY_SCAN_S, then DELAY_REFINE_S either
    side), and taken off from DELAY_SAID_S at DELAY_GAIN_PER_FIX (DELAY_GAIN_PER_S_DEPTH for depth)."""
    from dataclasses import replace
    from measurements import slice_run
    kinds = {name: kind}
    out = {}
    for name in (name,):
        best_run, best_a, best_n = None, 0, 0
        for r in runs:
            ch = next((c for c in r.channels if c.name == name), None)
            if ch is None:
                continue
            L = min(r.n_samples, int(round(DELAY_WINDOW_S * r.sample_rate_hz)))
            has = np.isfinite(ch.values[:, 0]).astype(float)
            counts = np.convolve(has, np.ones(L), "valid")
            a = int(np.argmax(counts))
            if counts[a] > best_n:
                best_run, best_a, best_n = r, a, int(counts[a])
        if best_run is None or best_n < 10:
            out[name] = {"kind": kinds[name], "note": "too few readings to tell a delay"}
            continue
        r = best_run
        L = min(r.n_samples, int(round(DELAY_WINDOW_S * r.sample_rate_hz)))

        def loglik(delay):
            k = int(round(delay * r.sample_rate_hz))
            chans = [replace(c, values=shifted_values(c.values, k)) if c.name == name else c for c in r.channels]
            res = smooth(slice_run(replace(r, channels=chans), best_a, best_a + L), params, vehicle, accel_sigma,
                         noise)
            return res.loglik if res.fallback is None else float("-inf")
        ll = {d: loglik(d) for d in DELAY_SCAN_S}
        d0 = max(ll, key=ll.get)
        for d in (d0 - DELAY_REFINE_S, d0 + DELAY_REFINE_S):
            if 0.0 <= d <= max(DELAY_SCAN_S) and d not in ll:
                ll[round(d, 3)] = loglik(d)
        best = max(ll, key=ll.get)
        gain = (ll[best] - ll[0.0]) / best_n if np.isfinite(ll[0.0]) and np.isfinite(ll[best]) else float("nan")
        per_s = gain * best_n / (L / r.sample_rate_hz) if np.isfinite(gain) else float("nan")
        clear = gain >= DELAY_GAIN_PER_FIX if kinds[name] == "position" else per_s >= DELAY_GAIN_PER_S_DEPTH
        out[name] = {"kind": kinds[name], "delay_s": round(float(best), 2), "gain_per_fix": round(float(gain), 3),
                     "gain_per_s": round(float(per_s), 3), "fixes": best_n,
                     "at_the_scan_edge": bool(best >= max(DELAY_SCAN_S)),
                     "late": bool(best >= DELAY_SAID_S and np.isfinite(gain) and clear), "pooled": False, "runs": 1}
    return out[name]


def choose_accel_sigma(runs: Sequence, params, vehicle, noise: Optional[dict] = None,
                       grid: Sequence[float] = ACCEL_SIGMA_GRID) -> tuple:
    """The grid value whose innovations are most likely over the runs (summed log-likelihood). Returns
    (sigma, {sigma: loglik}). With no reading of a known state anywhere (only a gyro), there's nothing to choose
    from: the middle of the grid."""
    scores = {}
    for s in grid:
        scores[s] = float(sum(smooth(r, params, vehicle, s, noise).loglik for r in runs))
    finite = {s: v for s, v in scores.items() if np.isfinite(v) and v != 0.0}
    if not finite:
        return float(grid[len(grid) // 2]), scores
    return max(finite, key=finite.get), scores


def set_anchors(runs: Sequence, params, vehicle, accel_sigma: float, noise: Optional[dict] = None) -> dict:
    """Each run's window-start velocities: the DVL's reading where it read, the smoothed estimate elsewhere.
    Returns a summary {runs, fallbacks: [...], nis_mean}."""
    fallbacks, nis = [], []
    for r in runs:
        res = smooth(r, params, vehicle, accel_sigma, noise)
        a = res.velocity.copy()
        dvl = next((ch for ch in r.channels if ch.kind == "dvl"), None)
        if dvl is not None:
            ok = np.all(np.isfinite(dvl.values), axis=1)
            a[ok] = dvl_over_ground(r, dvl)[ok]
        r.anchor_lin = a
        if res.fallback:
            fallbacks.append(f"{r.filepath}: {res.fallback}")
        if np.isfinite(res.nis_mean):
            nis.append(res.nis_mean)
    return {"runs": len(runs), "fallbacks": fallbacks, "nis_mean": float(np.mean(nis)) if nis else None,
            "accel_sigma": accel_sigma}
