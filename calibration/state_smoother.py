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
    """Smoothed body velocity over one run (see module docstring). noise: {channel name: reading std (its units)}."""
    from dynamics_model import SimulationConfig
    from fast_dynamics import FastDynamics
    from objective import with_thruster_lag
    from scipy.spatial.transform import Rotation

    noise = noise or {}
    run_l = with_thruster_lag(run, params)
    n, dt = run.n_samples, run.dt
    cfg = SimulationConfig(dt=dt)
    ns, sdt = cfg.n_substeps, cfg.sub_dt
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
        tau = F[k]
        roll, pitch = RP[k]
        sr, cr, sp, cp = math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch)
        w = W[k]
        cb = None if CB is None else CB[k]
        x = [v[0], v[1], v[2], w[0], w[1], w[2]]
        for _ in range(ns):
            vd = acc(x, tau, sr, cr, sp, cp, cb)
            x = [x[0] + vd[0] * sdt, x[1] + vd[1] * sdt, x[2] + vd[2] * sdt, w[0], w[1], w[2]]
        return np.array(x[:3])

    x = np.zeros(dim)
    P = np.zeros((dim, dim))
    P[:3, :3] = np.eye(3) * INITIAL_VELOCITY_STD ** 2
    for o, (_, _, ax) in zip(offs, pos_blocks):
        P[o:o + len(ax), o:o + len(ax)] = np.eye(len(ax)) * UNKNOWN_POSITION_VAR
    dvl = next((run.channels[ci] for ci, kind, _ in blocks if kind == "dvl"), None)
    if dvl is not None and np.all(np.isfinite(dvl.values[0])):
        x[:3] = dvl.values[0]
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
                for j in range(3):
                    h = np.zeros(dim)
                    h[j] = 1.0
                    rows.append(h); z.append(val[j]); rv.append(s2)
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


def _fallback(run, accel_sigma, why) -> SmoothResult:
    v = np.zeros((run.n_samples, 3))
    dvl = next((ch for ch in run.channels if ch.kind == "dvl"), None)
    if dvl is not None:
        ok = np.all(np.isfinite(dvl.values), axis=1)
        v[ok] = dvl.values[ok]
    return SmoothResult(velocity=v, loglik=float("-inf"), nis_mean=float("nan"), n_readings=0,
                        accel_sigma=accel_sigma, fallback=why)


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
            a[ok] = dvl.values[ok]
        r.anchor_lin = a
        if res.fallback:
            fallbacks.append(f"{r.filepath}: {res.fallback}")
        if np.isfinite(res.nis_mean):
            nis.append(res.nis_mean)
    return {"runs": len(runs), "fallbacks": fallbacks, "nis_mean": float(np.mean(nis)) if nis else None,
            "accel_sigma": accel_sigma}
