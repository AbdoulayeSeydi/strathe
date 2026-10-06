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
STEP_PRE_S, STEP_POST_S = 1.0, 3.0
AXES = ["u", "v", "w", "p", "q", "r"]
AXIS_WORDS = {"u": "forward (surge)", "v": "sideways (sway)", "w": "up/down (heave)", "p": "roll rate",
              "q": "pitch rate", "r": "turning rate (yaw)"}
UNITS = {"u": "m/s", "v": "m/s", "w": "m/s", "p": "rad/s", "q": "rad/s", "r": "rad/s"}
MODELS = ("calibrated", "default")


# ------------------------------------------------------------------ simulation pieces

def window_replay(params, vehicle, run, a: int, b: int):
    """Velocity over samples a..b-1, from the real velocity at a, the way the calibration replays it (logged
    orientation fed in). None if the simulation diverges."""
    traj = TrajectoryInputs(nu0=run.real_nu[a].copy(), forces_torques=run.forces_torques[a:b],
                            roll_pitch_series=run.roll_pitch[a:b],
                            current_body_series=None if run.current_body is None else run.current_body[a:b])
    try:
        nu = simulate_trajectory(params, vehicle, SimulationConfig(dt=run.dt), traj).nu
    except (FloatingPointError, ValueError):
        return None
    return nu if np.all(np.isfinite(nu)) else None


def path_from_velocity(nu_lin: np.ndarray, quat: np.ndarray, dt: float) -> np.ndarray:
    """Positions (n,3) from body-frame linear velocity and body->world orientation, starting at 0."""
    vw = Rotation.from_quat(quat).apply(nu_lin)
    return np.vstack([np.zeros(3), np.cumsum(vw[:-1] * dt, axis=0)])


def free_run(params, vehicle, run, a: int, n: int, step_s=None):
    """Roll, pitch, yaw (n,3, rad) with the model integrating its own orientation from the real state at a, driven
    by the logged forces. The calibration's own physics and step: FastDynamics.acceleration_trig (what the fit
    integrates, fed the model's own attitude instead of the logged one) in Forward Euler steps of
    SimulationConfig.physics_dt. The attitude is a quaternion, advanced by the body rates each step. None if it
    diverges."""
    from fast_dynamics import FastDynamics
    acc = FastDynamics(params, vehicle).acceleration_trig
    cfg = SimulationConfig(dt=run.dt) if step_s is None else SimulationConfig(dt=run.dt, physics_dt=step_s)
    sub, h = cfg.n_substeps, cfg.sub_dt
    qx, qy, qz, qw = (float(x) for x in run.orientation_quat[a])
    v = [float(x) for x in run.real_nu[a]]
    out = np.empty((n, 4))
    for k in range(n):
        out[k] = (qx, qy, qz, qw)
        i = a + k
        tau = run.forces_torques[i].tolist()
        cw = None
        if run.current_body is not None and np.any(run.current_body[i]):
            cw = Rotation.from_quat(run.orientation_quat[i]).apply(run.current_body[i]).tolist()
        for _ in range(sub):
            # world-down in body axes = -(third row of R): exactly what the logged roll/pitch give the fit
            r20, r21, r22 = 2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)
            cb = None
            if cw is not None:  # world current into the model's own body axes: R^T cw
                r00, r01, r02 = 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)
                r10, r11, r12 = 2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)
                cb = [r00 * cw[0] + r10 * cw[1] + r20 * cw[2], r01 * cw[0] + r11 * cw[1] + r21 * cw[2],
                      r02 * cw[0] + r12 * cw[1] + r22 * cw[2]]
            # acceleration_trig uses sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll): pass them with cp = 1
            vd = acc(v, tau, r21, r22, -r20, 1.0, cb)
            v = [v[j] + vd[j] * h for j in range(6)]
            if not all(math.isfinite(x) and abs(x) < 1e3 for x in v):
                return None
            wx, wy, wz = v[3] * h, v[4] * h, v[5] * h
            th = math.sqrt(wx * wx + wy * wy + wz * wz)
            if th > 1e-12:
                c, s_ = math.cos(th / 2), math.sin(th / 2) / th
                dx, dy, dz = wx * s_, wy * s_, wz * s_
                qw, qx, qy, qz = (qw * c - qx * dx - qy * dy - qz * dz, qw * dx + qx * c + qy * dz - qz * dy,
                                  qw * dy - qx * dz + qy * c + qz * dx, qw * dz + qx * dy - qy * dx + qz * c)
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return Rotation.from_quat(out).as_euler("xyz")


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _summary(x) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if not len(x):
        return {"n": 0, "median": None, "p25": None, "p75": None, "p90": None}
    return {"n": int(len(x)), "median": float(np.median(x)), "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)), "p90": float(np.percentile(x, 90))}


# ------------------------------------------------------------------ the four checks

def trajectory_drift(models: dict, runs: list, vehicle, horizons_s=DRIFT_HORIZONS_S) -> dict:
    """Position error at the end of non-overlapping windows of each horizon, per model, plus the no-motion
    baseline (distance actually travelled)."""
    out = {"horizons_s": list(horizons_s), "no_motion": [], **{m: [] for m in models}, "diverged": {m: [] for m in models}}
    for H in horizons_s:
        ends = {m: [] for m in models}
        travelled, div = [], {m: 0 for m in models}
        for run in runs:
            n = int(round(H * run.sample_rate_hz))
            for a in range(0, run.n_samples - n, n):
                b = a + n + 1
                real = path_from_velocity(run.real_nu[a:b, :3], run.orientation_quat[a:b], run.dt)
                travelled.append(float(np.linalg.norm(real[-1])))
                for m, p in models.items():
                    nu = window_replay(p, vehicle, run, a, b)
                    if nu is None:
                        div[m] += 1
                        continue
                    pred = path_from_velocity(nu[:, :3], run.orientation_quat[a:b], run.dt)
                    ends[m].append(float(np.linalg.norm(pred[-1] - real[-1])))
        out["no_motion"].append(_summary(travelled))
        for m in models:
            out[m].append(_summary(ends[m]))
            out["diverged"][m].append(div[m])
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
            real_change = run.real_nu[a + n] - run.real_nu[a]
            base.append(real_change ** 2)
            for m, p in models.items():
                nu = window_replay(p, vehicle, run, a, a + n + 1)
                if nu is None:
                    div[m] += 1
                    continue
                sq[m].append((nu[-1] - run.real_nu[a + n]) ** 2)
    persistence = np.sqrt(np.mean(base, axis=0)) if base else np.full(6, np.nan)
    out = {"horizon_s": horizon_s, "windows": len(base), "diverged": div,
           "same_velocity_rmse": persistence.tolist()}
    for m in models:
        rm = np.sqrt(np.mean(sq[m], axis=0)) if sq[m] else np.full(6, np.nan)
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
        if span < 1e-6:
            out[ax] = rec
            continue
        for run in runs:
            rate = run.sample_rate_hz
            post, tail = int(round(STEP_POST_S * rate)), max(1, int(round(0.5 * rate)))
            noise = float(np.std(np.diff(run.real_nu[:, j])))
            for k in find_steps(run.forces_torques[:, j], rate, 0.3 * span):
                dv_real = run.real_nu[k:k + post + 1, j] - run.real_nu[k, j]
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
        rmse = np.sqrt(np.mean((nu - run.real_nu) ** 2, axis=0))
        scored.append((float(np.mean(1 - rmse / zero_rmse)), run))
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
    real = path_from_velocity(run.real_nu[:, :3], run.orientation_quat, run.dt)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2))
    lab = "real vehicle (its logged velocity, integrated)"
    a1.plot(real[:, 0], real[:, 1], color=COLORS["real"], lw=1.5, label=lab)
    a2.plot(t, real[:, 2], color=COLORS["real"], lw=1.5, label=lab)
    for m, p in models.items():
        nu = window_replay(p, vehicle, run, 0, run.n_samples)
        if nu is None:
            continue
        pp = path_from_velocity(nu[:, :3], run.orientation_quat, run.dt)
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

def dataset_config_from_metrics(metrics: dict, rate_override=None) -> DatasetConfig:
    obj = metrics.get("objective") or {}
    rate = rate_override or obj.get("sampling_rate_hz") or 10.0
    offset = tuple(obj.get("torque_reference_offset_m") or (0.0, 0.0, 0.0))
    cur = obj.get("current_world_mps")
    by_file = obj.get("current_by_file")
    return DatasetConfig(float(rate), offset, None if not cur or not any(cur) else tuple(cur),
                         None if not by_file else tuple((f, tuple(c)) for f, c in by_file))


def validate(result_dir: str, data_dir: str, spec_path: str, rate=None, plots: bool = True) -> dict:
    with open(os.path.join(result_dir, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    import yaml
    with open(os.path.join(result_dir, "calibrated_parameters.yaml"), encoding="utf-8") as f:
        params = CalibrationParameters(**yaml.safe_load(f)["parameters"])
    spec = load_vehicle_spec(spec_path)
    vehicle = spec.vehicle
    dc = dataset_config_from_metrics(metrics, rate)
    held_dates = (metrics.get("dataset") or {}).get("heldout_dates")
    split = split_by_date(discover_csv_files(data_dir)[0], heldout_dates=held_dates)
    runs = [load_run(f, dc) for f in split.heldout_files]
    models = {"calibrated": params, "default": initial_parameters(spec)}
    out_dir = os.path.join(result_dir, "validation")
    os.makedirs(out_dir, exist_ok=True)
    res = {"heldout_dates": split.heldout_groups, "heldout_recordings": len(runs),
           "sampling_rate_hz": dc.sampling_rate_hz,
           "trajectory": trajectory_drift(models, runs, vehicle),
           "orientation": orientation_error(models, runs, vehicle),
           "response": response_error(models, runs, vehicle),
           "notes": ["Position drift integrates predicted and logged velocity with the logged orientation, so it is "
                     "the drift velocity error alone causes, measured against the logs' own velocity.",
                     "Orientation comes from a free-running replay: the model turns and tilts on its own, from the "
                     "real state every 10 s. The calibration itself is scored with the logged orientation fed in."],
           "plots": []}
    steps = step_responses(models, runs, vehicle)
    res["steps"] = steps["summary"]
    plt = _plt() if plots else None
    if plt is None and plots:
        res["notes"].append("No plots: matplotlib isn't installed (pip install matplotlib).")
    if plt is not None:
        zero = np.sqrt(np.mean(np.concatenate([r.real_nu for r in runs]) ** 2, axis=0))
        zero[zero < 1e-9] = 1e-9
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
