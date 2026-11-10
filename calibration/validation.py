"""
How the calibrated simulation compares with the real vehicle, on the recordings the calibration never saw
(PROJECT_STATUS.md #65). `strathe calibrate` runs it after the fit; `strathe validate` runs it on any result.

The calibration itself is scored on velocity, with the logged orientation fed in (dynamics_model.py's teacher
forcing). This adds what that score can't show:

1. **Position drift.** Predicted velocity, turned into a path with the logged orientation, against the logged
   velocity integrated the same way, from the real state at the start of each window (5-60 s). So it measures the
   drift that velocity error alone causes. Baseline: assuming the vehicle doesn't move (drift = distance travelled).
2. **Orientation.** A free-running replay: the model turns and tilts on its own (the fit's own equations and
   step, fed its own attitude), from the real state every 10 s, driven by the logged forces.
   Roll, pitch and yaw error after 3 and 10 s. Baseline: holding the starting attitude.
3. **Response.** The velocity change over the next second (= the average acceleration over it), predicted against
   measured, from every moment. Baseline: assuming the velocity stays the same. And command steps: when a force or
   torque jumps and then holds, how fast (time to 63% of the change) and how far the velocity responds, model
   against real.
4. **Plots** of all of it, in <result>/validation/, and validation/validation.json with the numbers. The report
   (calibration_report.py) adds a section from them.

    python validation.py --result calibration_output_my_rov --data-dir <logs> --spec vehicle_spec.yaml
"""
from __future__ import annotations

import argparse
from typing import Optional
import json
import math
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from data_split import discover_csv_files, split_by_date
from dynamics_model import CalibrationParameters, SimulationConfig, TrajectoryInputs, simulate_trajectory
from objective import DatasetConfig, load_run
from optimizer import initial_parameters, load_vehicle_spec

DRIFT_HORIZONS_S = (5, 10, 20, 30, 60)
REPORT_DRIFT_S = (10, 30, 60)
ORIENT_WINDOW_S = 10.0
ORIENT_REPORT_S = (3, 10)
RESPONSE_S = 1.0
FIN_WINDOW_S = 30.0   # how long a fin vehicle flies itself before it's compared
FIN_SHORT_WINDOW_S = 5.0     # and over a horizon an unstable vehicle allows
UNSTABLE_DOUBLING_S = 6.0    # errors doubling faster than this: five doublings or more in FIN_WINDOW_S, so a 30 s flight says little about the model
STEADY_RATE_RAD_S = 0.01     # an axis's body rate under this: that axis held steady, where a deployment's trim is compared
MIN_TRIM_SAMPLES = 30        # fewer steady samples than this in a deployment, and its trim isn't estimated
TRIM_AOA_BIN_RAD = math.radians(2.0)   # a deployment's trim is compared with training's only at angles of attack both flew at
MIN_BIN_SAMPLES = 5          # samples each side needs in an angle-of-attack bin for it to count
STEP_PRE_S, STEP_POST_S = 1.0, 3.0
AXES = ["u", "v", "w", "p", "q", "r"]
AXIS_WORDS = {"u": "forward (surge)", "v": "sideways (sway)", "w": "up/down (heave)", "p": "roll rate",
              "q": "pitch rate", "r": "turning rate (yaw)"}
UNITS = {"u": "m/s", "v": "m/s", "w": "m/s", "p": "rad/s", "q": "rad/s", "r": "rad/s"}
MODELS = ("calibrated", "default")


# ------------------------------------------------------------------ simulation pieces

_LAGGED = {}


def _lagged(run, params):
    """The run with the model's thruster lag applied to its whole recording (objective.with_thruster_lag), cached:
    every window of a recording then starts from the thrusters' real state, as in the calibration."""
    T = float(params.thruster_time_constant_s)
    Ts = float(params.fin_servo_T) if run_has_servo(run) else 0.0
    if T <= 0 and not run_has_servo(run):
        return run
    key = (id(run), T, Ts)
    hit = _LAGGED.get(key)
    if hit is None or hit[0] is not run:
        from objective import with_thruster_lag
        hit = _LAGGED[key] = (run, with_thruster_lag(run, params))
    return hit[1]


def run_has_servo(run) -> bool:
    return getattr(run, "servo", None) is not None


def start_velocity(run, a: int) -> np.ndarray:
    """The real state a replay starts from: the logged velocity, or with the sensor suite (#96) the window start the
    calibration uses (the DVL where it read, the state smoother's estimate elsewhere) and the gyro."""
    if run.channels is None:
        return run.real_nu[a].copy()
    return np.r_[run.anchor_lin[a], run.real_nu[a, 3:6]]


def window_replay(params, vehicle, run, a: int, b: int):
    """Velocity over samples a..b-1, from the real velocity at a, the way the calibration replays it (logged
    orientation fed in), with the model's thruster lag. None if the simulation diverges."""
    run = _lagged(run, params)
    traj = TrajectoryInputs(nu0=start_velocity(run, a), forces_torques=run.forces_torques[a:b],
                            roll_pitch_series=run.roll_pitch[a:b],
                            current_body_series=None if run.current_body is None else run.current_body[a:b],
                            control_series=None if run.controls is None else run.controls[a:b])
    try:
        nu = simulate_trajectory(params, vehicle, SimulationConfig(dt=run.dt), traj).nu
    except (FloatingPointError, ValueError):
        return None
    return nu if np.all(np.isfinite(nu)) else None


def path_from_velocity(nu_lin: np.ndarray, quat: np.ndarray, dt: float) -> np.ndarray:
    """Positions (n,3) from body-frame linear velocity and body->world orientation, starting at 0."""
    vw = Rotation.from_quat(quat).apply(nu_lin)
    return np.vstack([np.zeros(3), np.cumsum(vw[:-1] * dt, axis=0)])


def control_at(run, i: int):
    """Sample i's fin angles and the like, or None for a vehicle without them."""
    c = getattr(run, "controls", None)
    return None if c is None else c[i].tolist()


def world_current(run, i: int):
    """Sample i's water current in world axes (a list), or None in still water."""
    cb = getattr(run, "current_body", None)
    if cb is None or not np.any(cb[i]):
        return None
    return Rotation.from_quat(run.orientation_quat[i]).apply(cb[i]).tolist()


def current_in_body(cw, qx, qy, qz, qw) -> list:
    """A world current in the model's own body axes, R^T cw, for a model integrating its own attitude."""
    r20, r21, r22 = 2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)
    r00, r01, r02 = 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)
    r10, r11, r12 = 2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)
    return [r00 * cw[0] + r10 * cw[1] + r20 * cw[2], r01 * cw[0] + r11 * cw[1] + r21 * cw[2],
            r02 * cw[0] + r12 * cw[1] + r22 * cw[2]]


def free_run(params, vehicle, run, a: int, n: int, step_s=None, full: bool = False):
    """Roll, pitch, yaw (n,3, rad) with the model integrating its own orientation from the real state at a, driven
    by the logged forces. The calibration's own physics and step: FastDynamics.acceleration_trig (what the fit
    integrates, fed the model's own attitude instead of the logged one) in Forward Euler steps of
    SimulationConfig.physics_dt. The attitude is a quaternion, advanced by the body rates each step. None if it
    diverges. full: also (n,6) velocities and (n,) depth change, with the model's own attitude turning its velocity."""
    from fast_dynamics import FastDynamics
    run = _lagged(run, params)
    acc = FastDynamics(params, vehicle).acceleration_trig
    cfg = SimulationConfig(dt=run.dt) if step_s is None else SimulationConfig(dt=run.dt, physics_dt=step_s)
    sub, h = cfg.n_substeps, cfg.sub_dt
    qx, qy, qz, qw = (float(x) for x in run.orientation_quat[a])
    v = [float(x) for x in start_velocity(run, a)]
    out = np.empty((n, 4))
    vel, sink, z = np.empty((n, 6)), np.empty(n), 0.0
    for k in range(n):
        out[k] = (qx, qy, qz, qw)
        vel[k], sink[k] = v, z
        i = a + k
        tau = run.forces_torques[i].tolist()
        cw = world_current(run, i)
        ctl = control_at(run, i)
        for _ in range(sub):
            # world-down in body axes = -(third row of R): exactly what the logged roll/pitch give the fit
            r20, r21, r22 = 2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)
            cb = None if cw is None else current_in_body(cw, qx, qy, qz, qw)
            # acceleration_trig uses sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll): pass them with cp = 1
            vd = acc(v, tau, r21, r22, -r20, 1.0, cb, ctl)
            v = [v[j] + vd[j] * h for j in range(6)]
            if not all(math.isfinite(x) and abs(x) < 1e3 for x in v):
                return None
            z += (r20 * v[0] + r21 * v[1] + r22 * v[2]) * h
            wx, wy, wz = v[3] * h, v[4] * h, v[5] * h
            th = math.sqrt(wx * wx + wy * wy + wz * wz)
            if th > 1e-12:
                c, s_ = math.cos(th / 2), math.sin(th / 2) / th
                dx, dy, dz = wx * s_, wy * s_, wz * s_
                qw, qx, qy, qz = (qw * c - qx * dx - qy * dy - qz * dz, qw * dx + qx * c + qy * dz - qz * dy,
                                  qw * dy - qx * dz + qy * c + qz * dx, qw * dz + qx * dy - qy * dx + qz * c)
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    rpy = Rotation.from_quat(out).as_euler("xyz")
    return (rpy, vel, sink) if full else rpy


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _summary(x) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if not len(x):
        return {"n": 0, "median": None, "p25": None, "p75": None, "p90": None}
    return {"n": int(len(x)), "median": float(np.median(x)), "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)), "p90": float(np.percentile(x, 90))}


# ------------------------------------------------------------------ the four checks

FIX_TOLERANCE_S = 0.5   # a position fix counts for a window's start or end within this much of it


def drift_reference(run):
    """What a window's real displacement is measured with: the logged velocity integrated (the original check), or
    with the sensor suite (#96) the vehicle's own position fixes, independent of the model and of any velocity (a
    horizontal fix, else depth). (label, columns of the channel, its readings (N, k) or None)."""
    if run.channels is None:
        return "the logged velocity, integrated", (0, 1, 2), None
    for ch in run.channels:
        if ch.kind == "position" and "n" in ch.axes:
            return f"the position fixes ({ch.name})", tuple("ned".index(a) for a in ch.axes), ch.values
    for ch in run.channels:
        if ch.kind == "depth":
            return f"depth ({ch.name})", (2,), ch.values
    return None, (), None


def _nearest_fix(vals, i, lo, hi, tol):
    """The valid reading nearest sample i within [lo, hi) and tol samples, or None."""
    best = None
    for k in range(max(lo, i - tol), min(hi, i + tol + 1)):
        if np.all(np.isfinite(vals[k])) and (best is None or abs(k - i) < abs(best - i)):
            best = k
    return best


def trajectory_drift(models: dict, runs: list, vehicle, horizons_s=DRIFT_HORIZONS_S) -> dict:
    """Position error at the end of non-overlapping windows of each horizon, per model, plus the no-motion
    baseline (distance actually travelled). With the sensor suite, against the position fixes (or depth): the
    displacement between the fixes nearest the window's start and end, against the simulated displacement between
    the same samples."""
    out = {"horizons_s": list(horizons_s), "no_motion": [], **{m: [] for m in models}, "diverged": {m: [] for m in models}}
    against = None
    for H in horizons_s:
        ends = {m: [] for m in models}
        travelled, div = [], {m: 0 for m in models}
        for run in runs:
            label, axes, fixes = drift_reference(run)
            if label is None:
                continue
            against = against or label
            n = int(round(H * run.sample_rate_hz))
            tol = int(round(FIX_TOLERANCE_S * run.sample_rate_hz))
            for a in range(0, run.n_samples - n, n):
                b = a + n + 1
                if fixes is None:
                    ia, ib = 0, n
                    real = path_from_velocity(run.real_nu[a:b, :3], run.orientation_quat[a:b], run.dt)
                    disp = real[-1]
                else:
                    ka, kb = _nearest_fix(fixes, a, a, b, tol), _nearest_fix(fixes, b - 1, a, b, tol)
                    if ka is None or kb is None or kb <= ka:
                        continue
                    ia, ib = ka - a, kb - a
                    disp = fixes[kb] - fixes[ka]
                travelled.append(float(np.linalg.norm(disp)))
                for m, p in models.items():
                    nu = window_replay(p, vehicle, run, a, b)
                    if nu is None:
                        div[m] += 1
                        continue
                    pred = path_from_velocity(nu[:, :3], run.orientation_quat[a:b], run.dt)
                    d_pred = (pred[ib] - pred[ia])[list(axes)] if fixes is not None else pred[-1]
                    ends[m].append(float(np.linalg.norm(d_pred - disp)))
        out["no_motion"].append(_summary(travelled))
        for m in models:
            out[m].append(_summary(ends[m]))
            out["diverged"][m].append(div[m])
    out["against"] = against
    return out


def orientation_error(models: dict, runs: list, vehicle, window_s=ORIENT_WINDOW_S, report_s=ORIENT_REPORT_S) -> dict:
    """Roll/pitch/yaw error (deg) after report_s seconds of free-running replay, per model, plus holding the
    starting attitude."""
    errs = {m: {s: [] for s in report_s} for m in (*models, "hold_start")}
    div = {m: 0 for m in models}
    windows = 0
    for run in runs:
        n = int(round(window_s * run.sample_rate_hz)) + 1
        logged = Rotation.from_quat(run.orientation_quat).as_euler("xyz")
        for a in range(0, run.n_samples - n + 1, n - 1):
            windows += 1
            real = logged[a:a + n]
            for s in report_s:
                k = int(round(s * run.sample_rate_hz))
                errs["hold_start"][s].append(np.degrees(np.abs(_wrap(real[k] - real[0]))))
            for m, p in models.items():
                ang = free_run(p, vehicle, run, a, n)
                if ang is None:
                    div[m] += 1
                    continue
                for s in report_s:
                    k = int(round(s * run.sample_rate_hz))
                    errs[m][s].append(np.degrees(np.abs(_wrap(ang[k] - real[k]))))
    out = {"window_s": window_s, "report_s": list(report_s), "windows": windows, "diverged": div}
    for m, by_s in errs.items():
        out[m] = {name: [_summary([e[i] for e in by_s[s]]) for s in report_s]
                  for i, name in enumerate(("roll", "pitch", "yaw"))}
    return out


def response_error(models: dict, runs: list, vehicle, horizon_s=RESPONSE_S) -> dict:
    """RMS error of the velocity change over the next horizon_s, from every window start, per axis; baseline:
    the velocity stays the same (a change of zero)."""
    sq = {m: [] for m in models}
    base = []
    div = {m: 0 for m in models}
    for run in runs:
        n = int(round(horizon_s * run.sample_rate_hz))
        for a in range(0, run.n_samples - n, n):
            real_change = run.real_nu[a + n] - run.real_nu[a]   # NaN on u, v, w where no DVL read (#96)
            base.append(real_change ** 2)
            for m, p in models.items():
                nu = window_replay(p, vehicle, run, a, a + n + 1)
                if nu is None:
                    div[m] += 1
                    continue
                sq[m].append((nu[-1] - run.real_nu[a + n]) ** 2)

    def rms(rows):
        if not rows:
            return np.full(6, np.nan)
        a = np.array(rows)
        with np.errstate(invalid="ignore"):
            return np.array([np.sqrt(np.nanmean(a[:, j])) if np.isfinite(a[:, j]).any() else np.nan
                             for j in range(6)])
    persistence = rms(base)
    out = {"horizon_s": horizon_s, "windows": len(base), "diverged": div,
           "same_velocity_rmse": persistence.tolist()}
    for m in models:
        rm = rms(sq[m])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[m] = {"rmse": rm.tolist(), "skill_vs_same_velocity": (1 - rm / persistence).tolist()}
    return out


def find_steps(F: np.ndarray, rate: float, thresh: float) -> list:
    """Sample indices where a command jumps by at least thresh and holds: steady for STEP_PRE_S before and
    STEP_POST_S after (spread under 15% of the jump). At least STEP_POST_S apart."""
    pre, post = int(round(STEP_PRE_S * rate)), int(round(STEP_POST_S * rate))
    steps, k = [], pre
    while k < len(F) - post:
        before, after = F[k - pre:k], F[k:k + post]
        d = float(np.median(after) - np.median(before))
        if abs(d) >= thresh and before.std() < 0.15 * abs(d) and after.std() < 0.15 * abs(d):
            steps.append(k)
            k += post
        else:
            k += 1
    return steps


def _rise(dv: np.ndarray, final: float, rate: float):
    if final == 0:
        return None
    hit = np.flatnonzero(dv * np.sign(final) >= 0.63 * abs(final))
    return None if not len(hit) else float(hit[0] / rate)


def step_responses(models: dict, runs: list, vehicle) -> dict:
    """Per axis: the command steps found, and for each the time to 63% of the velocity change and its size, real
    against each model (replayed from the real velocity at the step)."""
    out = {}
    allF = np.concatenate([r.forces_torques for r in runs])
    for j, ax in enumerate(AXES):
        span = float(np.percentile(allF[:, j], 95) - np.percentile(allF[:, j], 5))
        rec = {"steps": 0, "real_rise_s": [], "examples": []}
        for m in models:
            rec[f"{m}_rise_s"], rec[f"{m}_size_ratio"] = [], []
        if span < 1e-6 or not any(np.isfinite(r.real_nu[:, j]).any() for r in runs):   # no DVL: not measured (#96)
            out[ax] = rec
            continue
        for run in runs:
            rate = run.sample_rate_hz
            post, tail = int(round(STEP_POST_S * rate)), max(1, int(round(0.5 * rate)))
            d_ = np.diff(run.real_nu[:, j])
            if not np.isfinite(d_).any():
                continue
            noise = float(np.nanstd(d_))
            for k in find_steps(run.forces_torques[:, j], rate, 0.3 * span):
                dv_real = run.real_nu[k:k + post + 1, j] - run.real_nu[k, j]
                if not np.all(np.isfinite(dv_real)):   # a DVL gap across the step
                    continue
                final_real = float(dv_real[-tail:].mean())
                d = float(np.median(run.forces_torques[k:k + post, j]) - np.median(run.forces_torques[k - post // 3:k, j]))
                if abs(final_real) < 3 * noise or np.sign(final_real) != np.sign(d):
                    continue  # the vehicle didn't clearly respond (or something else moved it)
                rec["steps"] += 1
                rec["real_rise_s"].append(_rise(dv_real, final_real, rate))
                ex = {"file": os.path.basename(run.filepath), "start": int(k), "real": dv_real.tolist()}
                for m, p in models.items():
                    nu = window_replay(p, vehicle, run, k, k + post + 1)
                    if nu is None:
                        rec[f"{m}_rise_s"].append(None)
                        rec[f"{m}_size_ratio"].append(None)
                        continue
                    dv = nu[:, j] - nu[0, j]
                    fin = float(dv[-tail:].mean())
                    rec[f"{m}_rise_s"].append(_rise(dv, fin, rate))
                    rec[f"{m}_size_ratio"].append(fin / final_real)
                    ex[m] = dv.tolist()
                rec["examples"].append(ex)
        out[ax] = rec
    summary = {}
    for ax, rec in out.items():
        s = {"steps": rec["steps"], "real_rise_s": _summary(rec["real_rise_s"])["median"]}
        for m in models:
            s[f"{m}_rise_s"] = _summary(rec[f"{m}_rise_s"])["median"]
            s[f"{m}_size_ratio"] = _summary(rec[f"{m}_size_ratio"])["median"]
        summary[ax] = s
    return {"summary": summary, "detail": out}


def _real_sink(run, a: int, n: int) -> Optional[np.ndarray]:
    """How far the vehicle really sank over a window: from a depth sensor, else from its logged velocity. None if neither reads."""
    for ch in run.channels or []:
        if ch.kind == "depth":
            d = np.asarray(ch.values[a:a + n, 0], dtype=float)
            if np.isfinite(d[[0, -1]]).all():
                return d - d[0]
            return None
    lin = run.real_nu[a:a + n, :3]
    if not np.isfinite(lin).all():
        return None
    return path_from_velocity(lin, run.orientation_quat[a:a + n], run.dt)[:, 2]


def moment_residuals(params, vehicle, runs: list) -> list:
    """For roll, pitch and yaw, the model's one-step prediction of the rate less the logged next rate, over the step (rad/s per s), on
    each sample where that axis's rate holds steady, whatever the others do: the logged state, attitude, forces and fins fed in and
    held over the step, the fit's own teacher-forced Euler substeps. A model right for the vehicle leaves these near zero on any
    manoeuvre; a vehicle re-ballasted since leaves a steady offset, which the fins' biases take up."""
    from fast_dynamics import FastDynamics
    acc = FastDynamics(params, vehicle).acceleration_trig
    out = [[], [], []]
    for run in runs:
        if run.n_samples < 2:
            continue
        rl = _lagged(run, params)
        cfg = SimulationConfig(dt=run.dt)
        sub, h = cfg.n_substeps, cfg.sub_dt
        rates = run.real_nu[:, 3:6]
        ok = np.isfinite(rates[:-1]) & np.isfinite(rates[1:])
        steady = ok & (np.abs(np.nan_to_num(rates[:-1], nan=1.0)) < STEADY_RATE_RAD_S)
        for i in np.flatnonzero(steady.any(axis=1)):
            v = start_velocity(rl, i)
            roll, pitch = (float(x) for x in rl.roll_pitch[i])
            if not (np.all(np.isfinite(v)) and math.isfinite(roll) and math.isfinite(pitch)):
                continue
            sr, cr, sp, cp = math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch)
            cb = None if rl.current_body is None else rl.current_body[i].tolist()
            tau, ctl = rl.forces_torques[i].tolist(), control_at(rl, i)
            v = v.tolist()
            for _ in range(sub):
                vd = acc(v, tau, cp * sr, cp * cr, sp, 1.0, cb, ctl)
                v = [v[k] + vd[k] * h for k in range(6)]
            for j in range(3):
                if steady[i, j] and math.isfinite(v[3 + j]):
                    out[j].append(((v[3 + j] - rates[i + 1, j]) / run.dt, math.atan2(rl.real_nu[i, 2], rl.real_nu[i, 0])))
    return [np.asarray(o, dtype=float).reshape(-1, 2) for o in out]


def matched_gap(here: np.ndarray, ref: np.ndarray) -> tuple:
    """(mean residual here less in training, samples compared): rows (residual, angle of attack), compared bin by bin
    (TRIM_AOA_BIN_RAD) where both have MIN_BIN_SAMPLES, weighted by the smaller count; nan under MIN_TRIM_SAMPLES compared. The
    model's pitch moment is only known at the angles of attack it trained at: OOI's 2024 REMUS flew to about 18 degrees, past
    training's 14, where the model's moment falls about 3x short, and that read as a 4 degree trim (#97)."""
    if not len(here) or not len(ref):
        return float("nan"), 0
    bins = lambda a: np.floor(a[:, 1] / TRIM_AOA_BIN_RAD).astype(int)
    bh, br = bins(here), bins(ref)
    num = den = 0.0
    for k in set(bh.tolist()) & set(br.tolist()):
        h, r = here[bh == k, 0], ref[br == k, 0]
        if len(h) >= MIN_BIN_SAMPLES and len(r) >= MIN_BIN_SAMPLES:
            w = min(len(h), len(r))
            num += w * (h.mean() - r.mean())
            den += w
    return (num / den if den >= MIN_TRIM_SAMPLES else float("nan")), int(den)


def retrim(params, vehicle, train_runs: list, runs: list) -> tuple:
    """(params, (roll, pitch, yaw) bias change in rad, steady samples here, in training): the fin biases moved so that the model's
    steady moment residual on these runs matches its residual in training. A vehicle is re-ballasted between deployments, so the same
    vehicle needs different fin angles to hold it (OOI's REMUS 600s: elevator trims 4 degrees apart, #97). The model explains
    what speed, attitude and depth change, so a deployment flown differently but trimmed the same keeps its biases (comparing raw fin
    angles took a change of manoeuvre for a change of trim: T-fossen-rich, #97). An axis with fewer than MIN_TRIM_SAMPLES steady
    samples on either side, or that the biases can't move, keeps its bias. Two Newton steps on a finite-difference Jacobian."""
    from dataclasses import replace
    names = ("fin_bias_roll", "fin_bias_pitch", "fin_bias_yaw")
    ref = moment_residuals(params, vehicle, train_runs)
    n_train = [len(x) for x in ref]
    shifted = lambda d: replace(params, **{k: getattr(params, k) + float(x) for k, x in zip(names, d)})
    def gap(d):
        got = [matched_gap(h, r) for h, r in zip(moment_residuals(shifted(d), vehicle, runs), ref)]
        return np.array([g for g, _ in got]), [n for _, n in got]
    d = np.zeros(3)
    g, n_here = gap(d)
    axes = [j for j in range(3) if np.isfinite(g[j])]
    eps = math.radians(0.5)
    for _ in range(2):
        if not axes:
            break
        J = np.zeros((3, 3))
        for k in axes:
            e = np.zeros(3)
            e[k] = eps
            J[:, k] = (gap(d + e)[0] - g) / eps
        sub = J[np.ix_(axes, axes)]
        live = [a for a, col in zip(axes, np.abs(sub).max(axis=0)) if col > 1e-9]
        if len(live) < len(axes):
            axes = live
            continue
        d[axes] -= np.linalg.lstsq(sub, g[axes], rcond=None)[0]
        g = gap(d)[0]
    done = [j in axes for j in range(3)]
    d = np.where(done, d, 0.0)
    return shifted(d), d, done, n_here, n_train


def retrim_by_deployment(params, vehicle, train_runs: list, runs: list) -> tuple:
    """([params for each run], {deployment: what was done}): each held-out deployment (a recording date) flown at its own trim."""
    out, notes = [params] * len(runs), {}
    for date in sorted({r.date or "" for r in runs}):
        idx = [i for i, r in enumerate(runs) if (r.date or "") == date]
        p, d, done, n, n_train = retrim(params, vehicle, train_runs, [runs[i] for i in idx])
        for i in idx:
            out[i] = p
        notes[date or "undated"] = {"retrimmed": any(done), "steady_samples": n, "training_steady_samples": n_train,
                                    **{f"{w}_deg": (round(math.degrees(float(x)), 3) if ok else None)
                                       for w, x, ok in zip(("roll", "pitch", "yaw"), d, done)}}
    return out, notes


# Holding course is hard to beat on a straight line: an autopilot logged the fins that kept it there, and any model flying them open loop
# drifts. Where the vehicle really turned, holding course is the wrong answer and the model has to earn its keep: windows whose logged
# heading changed by more than this are scored on their own (#97: OOI's REMUS rows fly mostly straight, and lost to holding course there).
TURNED_DEG = 30.0


def fin_free_run(models: dict, runs: list, vehicle, window_s: float = FIN_WINDOW_S, run_params: Optional[dict] = None) -> dict:
    """A fin vehicle flying itself from each window's real start, on the logged fin angles: heading, pitch, depth and turn rate at
    the end, against assuming it holds its heading and pitch and carries straight on. The fit's own score feeds the logged pitch
    back in, which hides what the elevator does to depth, so this is the check that counts. run_params: {model: [params for each
    run]}, for a model flown at each deployment's own trim (retrim_by_deployment)."""
    keys = ("heading_deg", "pitch_deg", "depth_m", "turn_rate_rad_s")
    errs = {m: {k: [] for k in keys} for m in (*models, "straight_on")}
    turning = {m: [] for m in (*models, "straight_on")}     # heading, on windows where the vehicle really turned
    div = {m: 0 for m in models}
    windows = 0
    for i, run in enumerate(runs):
        n = int(round(window_s * run.sample_rate_hz)) + 1
        logged = Rotation.from_quat(run.orientation_quat).as_euler("xyz")
        for a in range(0, run.n_samples - n + 1, n - 1):
            sink = _real_sink(run, a, n)
            windows += 1
            real = logged[a:a + n]
            r_end = run.real_nu[a + n - 1, 5]
            hold = errs["straight_on"]
            hold["heading_deg"].append(float(np.degrees(abs(_wrap(real[-1, 2] - real[0, 2])))))
            turned = hold["heading_deg"][-1] > TURNED_DEG
            got_all = {}
            hold["pitch_deg"].append(float(np.degrees(abs(_wrap(real[-1, 1] - real[0, 1])))))
            hold["turn_rate_rad_s"].append(float(abs(r_end - run.real_nu[a, 5])))
            if sink is not None:
                speed = run.real_nu[a, 0] if np.isfinite(run.real_nu[a, 0]) else 0.0
                hold["depth_m"].append(float(abs(sink[-1] + speed * math.sin(real[0, 1]) * window_s)))
            for m, p in models.items():
                if run_params and m in run_params:
                    p = run_params[m][i]
                got = free_run(p, vehicle, run, a, n, full=True)
                if got is None:
                    div[m] += 1
                    continue
                rpy, vel, z = got
                e = errs[m]
                e["heading_deg"].append(float(np.degrees(abs(_wrap(rpy[-1, 2] - real[-1, 2])))))
                got_all[m] = e["heading_deg"][-1]
                e["pitch_deg"].append(float(np.degrees(abs(_wrap(rpy[-1, 1] - real[-1, 1])))))
                e["turn_rate_rad_s"].append(float(abs(vel[-1, 5] - r_end)))
                if sink is not None:
                    e["depth_m"].append(float(abs(z[-1] - sink[-1])))
            if turned and len(got_all) == len(models):      # the same windows for every model and for holding course
                for m, h in got_all.items():
                    turning[m].append(h)
                turning["straight_on"].append(hold["heading_deg"][-1])
    out = {"window_s": window_s, "windows": windows, "diverged": div}
    for m, by in errs.items():
        out[m] = {k: _summary(v) for k, v in by.items()}
    out["turning"] = {"turned_deg": TURNED_DEG, "windows": len(turning["straight_on"]),
                      **{m: _summary(v) for m, v in turning.items()}}
    return out


def fin_stability(params, vehicle, speed: float) -> dict:
    """How fast an error grows about straight flight with the fins centred: the largest eigenvalue over sway, heave, the rates, roll and pitch. A
    vehicle that's unstable with its fins held can't be flown far open loop by any model, even a perfect one: Prestero's REMUS doubles an error
    every 0.7 s, so logging its fins at 10 Hz alone sends it 129 degrees from itself in 30 s (#97)."""
    from dynamics_model import compute_acceleration
    ctl = [0.0] * vehicle.actuators.width

    def f(x):
        v = np.r_[speed, x[:5]]
        a = compute_acceleration(params, vehicle, v, np.zeros(6), x[5], x[6], None, ctl)[0]
        return np.r_[a[1:], x[2] + x[4] * math.tan(x[6]) * math.cos(x[5]), x[3] * math.cos(x[5]) - x[4] * math.sin(x[5])]
    J, h = np.zeros((7, 7)), 1e-5
    for i in range(7):
        e = np.zeros(7)
        e[i] = h
        J[:, i] = (f(e) - f(-e)) / (2 * h)
    growth = float(np.linalg.eigvals(J).real.max())
    return {"speed_m_s": float(speed), "growth_per_s": growth, "doubling_s": math.log(2) / growth if growth > 1e-9 else None}


def fin_steps(models: dict, runs: list, vehicle, act) -> dict:
    """Steps of the steering and pitching fins, and the turn or pitch rate each caused, real against the model."""
    from fin_fit import group_angle
    out = {}
    for group, word, j in ((0, "steering", 5), (1, "pitching", 4)):
        idx = [i for i, f in enumerate(act.fins) if f.group == group]
        if not idx:
            continue
        real, sim = [], {m: [] for m in models}
        for run in runs:
            rate = run.sample_rate_hz
            post = int(round(STEP_POST_S * rate))
            ang = group_angle(run, idx)
            span = float(np.percentile(ang, 95) - np.percentile(ang, 5))
            if span < 1e-3:
                continue
            for k in find_steps(ang, rate, .3 * span):
                dv = float(run.real_nu[k + post, j] - run.real_nu[k, j])
                if not np.isfinite(dv) or abs(dv) < 1e-4:
                    continue
                real.append(dv)
                for m, p in models.items():
                    nu = window_replay(p, vehicle, run, k, k + post + 1)
                    sim[m].append(None if nu is None else float(nu[-1, j] - nu[0, j]) / dv)
        out[word] = {"steps": len(real), **{f"{m}_size_ratio": _summary(sim[m])["median"] for m in models}}
    return out


# ------------------------------------------------------------------ plots

def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


COLORS = {"real": "#222222", "calibrated": "#1f77b4", "default": "#d62728", "no_motion": "#999999"}


def pick_runs(models: dict, runs: list, vehicle, zero_rmse: np.ndarray) -> dict:
    """Best, typical and worst held-out recording by the calibrated model's mean skill over the six axes (whole-run
    replay, as the calibration is scored)."""
    scored = []
    for run in runs:
        nu = window_replay(models["calibrated"], vehicle, run, 0, run.n_samples)
        if nu is None:
            scored.append((-np.inf, run))
            continue
        with np.errstate(invalid="ignore"):
            rmse = np.sqrt(np.nanmean((nu - run.real_nu) ** 2, axis=0))   # u, v, w NaN with no DVL (#96)
            sk = 1 - rmse / zero_rmse
        scored.append((float(np.nanmean(sk)) if np.isfinite(sk).any() else -np.inf, run))
    scored.sort(key=lambda t: t[0])
    return {"worst": scored[0][1], "typical": scored[len(scored) // 2][1], "best": scored[-1][1]}


def plot_velocity(plt, models, run, vehicle, path, title):
    t = np.arange(run.n_samples) * run.dt
    fig, axs = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    preds = {m: window_replay(p, vehicle, run, 0, run.n_samples) for m, p in models.items()}
    for j, ax in enumerate(AXES):
        a = axs[j]
        a.plot(t, run.real_nu[:, j], color=COLORS["real"], lw=1.2, label="real vehicle")
        for m, nu in preds.items():
            if nu is not None:
                a.plot(t, nu[:, j], color=COLORS[m], lw=1, alpha=0.85, label=f"{m} simulation")
        a.set_ylabel(f"{AXIS_WORDS[ax]}\n[{UNITS[ax]}]", fontsize=8)
        a.grid(alpha=0.3)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel("time [s]")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_trajectory(plt, models, run, vehicle, path, title):
    t = np.arange(run.n_samples) * run.dt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2))
    label, axes, fixes = drift_reference(run)
    if fixes is None:
        real = path_from_velocity(run.real_nu[:, :3], run.orientation_quat, run.dt)
        lab = "real vehicle (its logged velocity, integrated)"
        a1.plot(real[:, 0], real[:, 1], color=COLORS["real"], lw=1.5, label=lab)
        a2.plot(t, real[:, 2], color=COLORS["real"], lw=1.5, label=lab)
        origin = np.zeros(3)
    else:   # the sensor suite (#96): the vehicle's own fixes, the simulated path started at the first one
        ok = np.all(np.isfinite(fixes), axis=1)
        k0 = int(np.flatnonzero(ok)[0]) if ok.any() else 0
        lab = f"real vehicle ({label})"
        full = np.full((run.n_samples, 3), np.nan)
        full[:, list(axes)] = fixes
        if 0 in axes:
            a1.plot(full[ok, 0] - full[k0, 0], full[ok, 1] - full[k0, 1], ".", color=COLORS["real"], ms=3, label=lab)
        if 2 in axes:
            a2.plot(t[ok], full[ok, 2] - full[k0, 2], ".", color=COLORS["real"], ms=3, label=lab)
        origin = k0
    for m, p in models.items():
        nu = window_replay(p, vehicle, run, 0, run.n_samples)
        if nu is None:
            continue
        pp = path_from_velocity(nu[:, :3], run.orientation_quat, run.dt)
        if fixes is not None:
            pp = pp - pp[origin]
        a1.plot(pp[:, 0], pp[:, 1], color=COLORS[m], lw=1.1, label=f"{m} simulation")
        a2.plot(t, pp[:, 2], color=COLORS[m], lw=1.1, label=f"{m} simulation")
    a1.plot([0], [0], "o", color="k", ms=4)
    a1.set_xlabel("x [m]"); a1.set_ylabel("y [m]"); a1.set_title("path seen from above", fontsize=10)
    a1.axis("equal"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a2.set_xlabel("time [s]"); a2.set_ylabel("z [m] (the logs' vertical axis)"); a2.set_title("vertical", fontsize=10)
    a2.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_drift(plt, drift: dict, path):
    H = drift["horizons_s"]
    fig, a = plt.subplots(figsize=(7.5, 4.8))
    for key, label in (("no_motion", "assuming it doesn't move"), ("default", "default physics"),
                       ("calibrated", "calibrated")):
        med = [s["median"] for s in drift[key]]
        lo = [s["p25"] for s in drift[key]]
        hi = [s["p75"] for s in drift[key]]
        if any(v is None for v in med):
            continue
        a.plot(H, med, "o-", color=COLORS[key], label=label)
        a.fill_between(H, lo, hi, color=COLORS[key], alpha=0.15)
    a.set_xlabel("prediction horizon [s]")
    a.set_ylabel("position error at the end [m]\n(median; band = middle half)")
    a.set_title("Position drift on unseen recordings", fontsize=11)
    a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_orientation(plt, models, run, vehicle, path, title, window_s=ORIENT_WINDOW_S):
    """The calibrated model only: default physics tumbles and would hide everything (its numbers are in the
    report). Yaw is unwrapped so a turn through 180 deg isn't drawn as a jump."""
    t = np.arange(run.n_samples) * run.dt
    logged = np.degrees(Rotation.from_quat(run.orientation_quat).as_euler("xyz"))
    logged[:, 2] = np.degrees(np.unwrap(np.radians(logged[:, 2])))
    models = {"calibrated": models["calibrated"]}
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    n = int(round(window_s * run.sample_rate_hz)) + 1
    for i, name in enumerate(("roll", "pitch", "yaw")):
        axs[i].plot(t, logged[:, i], color=COLORS["real"], lw=1.2, label="real vehicle")
        axs[i].set_ylabel(f"{name} [deg]")
        axs[i].grid(alpha=0.3)
    for m, p in models.items():
        first = True
        for a in range(0, run.n_samples - n + 1, n - 1):
            ang = free_run(p, vehicle, run, a, n)
            if ang is None:
                continue
            ang = np.degrees(ang)
            ang[:, 2] = logged[a, 2] + np.degrees(np.unwrap(_wrap(np.radians(ang[:, 2] - logged[a, 2]))))
            for i in range(3):
                axs[i].plot(t[a:a + n], ang[:, i], color=COLORS[m], lw=1, alpha=0.85,
                            label=f"{m} simulation (restarted every {window_s:g} s)" if first and i == 0 else None)
            first = False
    axs[0].legend(fontsize=8, loc="upper right")
    axs[-1].set_xlabel("time [s]")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_steps(plt, steps: dict, rate: float, path):
    axes = [ax for ax in AXES if steps["detail"][ax]["examples"]]
    if not axes:
        return False
    fig, axs = plt.subplots(1, len(axes), figsize=(4 * len(axes), 3.8), squeeze=False)
    for a, ax in zip(axs[0], axes):
        exs = steps["detail"][ax]["examples"]
        ex = exs[len(exs) // 2]
        t = np.arange(len(ex["real"])) / rate
        a.plot(t, ex["real"], color=COLORS["real"], lw=1.5, label="real vehicle")
        for m in MODELS:
            if m in ex:
                a.plot(t, ex[m], color=COLORS[m], lw=1.1, label=f"{m} simulation")
        a.set_title(f"{AXIS_WORDS[ax]}: a command step\n({len(exs)} found)", fontsize=9)
        a.set_xlabel("time after the step [s]")
        a.set_ylabel(f"change in velocity [{UNITS[ax]}]", fontsize=8)
        a.grid(alpha=0.3)
    axs[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ driver

def dataset_config_from_metrics(metrics: dict, rate_override=None, actuators=None) -> DatasetConfig:
    obj = metrics.get("objective") or {}
    rate = rate_override or obj.get("sampling_rate_hz") or 10.0
    offset = tuple(obj.get("torque_reference_offset_m") or (0.0, 0.0, 0.0))
    cur = obj.get("current_world_mps")
    by_file = obj.get("current_by_file")
    return DatasetConfig(float(rate), offset, None if not cur or not any(cur) else tuple(cur),
                         None if not by_file else tuple((f, tuple(c)) for f, c in by_file),
                         sensors=tuple(obj.get("sensors") or ("auto",)),
                         use_accelerometer=bool(obj.get("use_accelerometer", False)), actuators=actuators)


def sensor_anchors(metrics: dict, runs: list, params, vehicle) -> Optional[dict]:
    """The sensor suite (#96): each held-out window's starting velocity, as the calibration set them (the smoother
    with the calibrated values and the noise it chose)."""
    from run_calibration import sensor_runs_from_metrics
    return sensor_runs_from_metrics(runs, metrics, params, vehicle)


def validate(result_dir: str, data_dir: str, spec_path: str, rate=None, plots: bool = True) -> dict:
    with open(os.path.join(result_dir, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    import yaml
    with open(os.path.join(result_dir, "calibrated_parameters.yaml"), encoding="utf-8") as f:
        params = CalibrationParameters(**yaml.safe_load(f)["parameters"])
    spec = load_vehicle_spec(spec_path)
    vehicle = spec.vehicle
    dc = dataset_config_from_metrics(metrics, rate, spec.vehicle.actuators)
    from dataclasses import replace as _replace
    from run_calibration import _sensor_mounts
    dc = _replace(dc, lever_arms=_sensor_mounts(spec))   # the sensors' mounts (#96)
    held_dates = (metrics.get("dataset") or {}).get("heldout_dates")
    split = split_by_date(discover_csv_files(data_dir)[0], heldout_dates=held_dates)
    runs = [load_run(f, dc) for f in split.heldout_files]
    anchoring = sensor_anchors(metrics, runs, params, vehicle)
    models = {"calibrated": params, "default": initial_parameters(spec)}
    out_dir = os.path.join(result_dir, "validation")
    os.makedirs(out_dir, exist_ok=True)
    res = {"heldout_dates": split.heldout_groups, "heldout_recordings": len(runs),
           "sampling_rate_hz": dc.sampling_rate_hz,
           "trajectory": trajectory_drift(models, runs, vehicle),
           "orientation": orientation_error(models, runs, vehicle),
           "response": response_error(models, runs, vehicle),
           "notes": [("Position drift integrates predicted and logged velocity with the logged orientation, so it is "
                      "the drift velocity error alone causes, measured against the logs' own velocity.")
                     if anchoring is None else
                     ("Position drift is measured against the vehicle's own position fixes (or depth): the simulated "
                      "displacement between the fixes nearest each window's start and end against theirs, so it "
                      "checks the model against an independent measurement, not against a velocity. Each window "
                      "starts from the calibration's estimate of the real velocity (SENSORS.md)."),
                     "Orientation comes from a free-running replay: the model turns and tilts on its own, from the "
                     "real state every 10 s. The calibration itself is scored with the logged orientation fed in."],
           "plots": []}
    steps = step_responses(models, runs, vehicle)
    res["steps"] = steps["summary"]
    if vehicle.actuators is not None and vehicle.actuators.fins:
        train_runs = [load_run(f, dc) for f in split.train_files]
        trimmed, res["fin_retrim"] = retrim_by_deployment(params, vehicle, train_runs, runs)
        per = {"calibrated": trimmed} if any(v["retrimmed"] for v in res["fin_retrim"].values()) else None
        res["fin_free_run"] = fin_free_run(models, runs, vehicle, run_params=per)
        res["fin_free_run_short"] = fin_free_run(models, runs, vehicle, window_s=FIN_SHORT_WINDOW_S, run_params=per)
        if per:
            res["fin_free_run_as_trained"] = fin_free_run(models, runs, vehicle)
        speed = float(np.nanmedian(np.concatenate([r.real_nu[:, 0] for r in runs])))
        if "calibrated" in models and np.isfinite(speed):
            res["fin_stability"] = fin_stability(models["calibrated"], vehicle, speed)
        res["fin_steps"] = fin_steps(models, runs, vehicle, vehicle.actuators)
    if anchoring is not None:
        res["sensor_suite"] = {"anchoring": anchoring, "drift_against": res["trajectory"].get("against")}
    plt = _plt() if plots else None
    if plt is None and plots:
        res["notes"].append("No plots: matplotlib isn't installed (pip install matplotlib).")
    if plt is not None:
        with np.errstate(invalid="ignore"):
            zero = np.sqrt(np.nanmean(np.concatenate([r.real_nu for r in runs]) ** 2, axis=0))
        zero[~(zero >= 1e-9)] = 1e-9
        picks = pick_runs(models, runs, vehicle, zero)
        for which, run in picks.items():
            name = os.path.splitext(os.path.basename(run.filepath))[0]
            p = f"velocity_{which}.png"
            plot_velocity(plt, models, run, vehicle, os.path.join(out_dir, p),
                          f"Velocity, {which} unseen recording ({name}): real vs simulated, whole recording")
            res["plots"].append(p)
        typical = picks["typical"]
        tname = os.path.splitext(os.path.basename(typical.filepath))[0]
        plot_trajectory(plt, models, typical, vehicle, os.path.join(out_dir, "trajectory_typical.png"),
                        f"Position, typical unseen recording ({tname}): velocity integrated over the whole recording")
        plot_drift(plt, res["trajectory"], os.path.join(out_dir, "drift.png"))
        plot_orientation(plt, models, typical, vehicle, os.path.join(out_dir, "orientation_typical.png"),
                         f"Orientation, typical unseen recording ({tname}): the model turning and tilting on its own")
        res["plots"] += ["trajectory_typical.png", "drift.png", "orientation_typical.png"]
        if plot_steps(plt, steps, dc.sampling_rate_hz, os.path.join(out_dir, "steps.png")):
            res["plots"].append("steps.png")
    with open(os.path.join(out_dir, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", required=True, help="calibration output folder (metrics.json, calibrated_parameters.yaml)")
    ap.add_argument("--data-dir", help="the logs (default: the folder the calibration used, from metrics.json)")
    ap.add_argument("--spec", default="vehicle_spec.yaml")
    ap.add_argument("--sampling-rate-hz", type=float, default=None, help="default: what the calibration used")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    data = args.data_dir
    if not data:
        with open(os.path.join(args.result, "metrics.json"), encoding="utf-8") as f:
            data = (json.load(f).get("reproducibility") or {}).get("data_dir")
    if not data or not os.path.isdir(data):
        print("❌ give --data-dir: the logs the calibration used")
        return 2
    r = validate(args.result, data, args.spec, args.sampling_rate_hz, plots=not args.no_plots)
    tr = r["trajectory"]
    for H in REPORT_DRIFT_S:
        i = tr["horizons_s"].index(H)
        c, n = tr["calibrated"][i], tr["no_motion"][i]
        if c["median"] is not None:
            print(f"Position drift after {H} s: calibrated {c['median']:.2f} m (median), "
                  f"distance travelled {n['median']:.2f} m", flush=True)
    o = r["orientation"]["calibrated"]
    for name in ("roll", "pitch", "yaw"):
        m = o[name][-1]["median"]
        if m is not None:
            print(f"{name.capitalize()} error after {ORIENT_REPORT_S[-1]} s on its own: {m:.1f} deg (median)", flush=True)
    print(f"Wrote {os.path.join(args.result, 'validation')} ({len(r['plots'])} plots, validation.json)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
