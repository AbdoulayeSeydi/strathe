"""
simfit/core.py -- what calibrating any simulator's own model from logs shares (PROJECT_STATUS.md #81): the windows a
recording is cut into, scoring a model on them (drift, velocity and attitude error per horizon), the baselines,
Strathe's own model over the same windows, the least-squares fit over a simulator's settings, loading the recordings
and the report's table and plot. An adapter (calibration/simfit/adapters/) supplies the simulator: how its model is
changed and how windows are run in it. Moved from stonefish_fit.py (#72-#80) unchanged.
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import shutil
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

FIT_WINDOW_S, FIT_STRIDE_S = 30.0, 30.0
EVAL_WINDOW_S, EVAL_STRIDE_S, HORIZONS_S = 60.0, 10.0, (10, 30, 60)


class SimFitError(RuntimeError):
    """A problem the lab can fix (shown as one plain line, exit code 2)."""


# ---------------------------------------------------------------------------------------------------------------------
# Windows and scores
# ---------------------------------------------------------------------------------------------------------------------

def windows(run, length_s: float, stride_s: float) -> list:
    n = int(round(length_s / run.dt)) + 1
    return [(a, n) for a in range(0, len(run.real_nu) - n + 1, max(1, int(round(stride_s / run.dt))))]


def _nu(run):
    """The velocity a replay starts from: logged, or with the sensor suite the estimated start (simfit/sensors.py)."""
    if getattr(run, "channels", None) is not None:
        from simfit.sensors import start_nu
        return start_nu(run)
    return run.real_nu


def window_rows(run, wins) -> list:
    nu = _nu(run)
    return [np.hstack([nu[a:a + n], run.orientation_quat[a:a + n], run.forces_torques[a:a + n]])
            for a, n in wins]


def path(V, Q, dt):
    R = Rotation.from_quat(np.nan_to_num(Q) + np.array([0, 0, 0, 1e-12]))
    return np.vstack([np.zeros(3), np.cumsum(R.apply(V[:, :3]) * dt, axis=0)[:-1]])


def score(runs_wins: list, preds: list, horizons=HORIZONS_S) -> dict:
    """Median drift, velocity RMSE per axis and attitude error per horizon, over all windows of all recordings. From
    the sensors (no DVL, #96): drift against the position fixes, simfit/sensors.score."""
    if runs_wins and getattr(runs_wins[0][0], "channels", None) is not None:
        from simfit.sensors import score as sensor_score
        return sensor_score(runs_wins, preds, horizons)
    vel = {h: [] for h in horizons}
    drift = {h: [] for h in horizons}
    att = {h: [] for h in horizons}
    total = diverged = 0
    for (run, wins), pr in zip(runs_wins, preds):
        for (a, n), (V, Q) in zip(wins, pr):
            total += 1
            if np.isnan(V).any():
                diverged += 1
                continue
            rv, rq = run.real_nu[a:a + n], run.orientation_quat[a:a + n]
            Pp, Pr = path(V, Q, run.dt), path(rv, rq, run.dt)
            e = np.degrees(np.abs((Rotation.from_quat(Q).as_euler("xyz") - Rotation.from_quat(rq).as_euler("xyz")
                                   + np.pi) % (2 * np.pi) - np.pi))
            for h in horizons:
                m = int(round(h / run.dt))
                if m >= n:
                    continue
                vel[h].append((V[1:m + 1] - rv[1:m + 1]) ** 2)
                drift[h].append(float(np.linalg.norm(Pp[m] - Pr[m])))
                att[h].append(e[m])
    out = {"windows": total, "diverged": diverged}
    for h in horizons:
        if drift[h]:
            out[f"{h}s"] = {"drift_median_m": round(float(np.median(drift[h])), 3),
                            "drift_p90_m": round(float(np.percentile(drift[h], 90)), 3),
                            "velocity_rmse": np.sqrt(np.concatenate(vel[h]).mean(axis=0)).round(4).tolist(),
                            "attitude_error_median_deg": np.median(np.array(att[h]), axis=0).round(2).tolist()}
    return out


def strathe_free_run(params, vehicle, run, a: int, n: int):
    """Strathe's own model over the same window: its physics step, its own attitude (as validation.free_run)."""
    from dynamics_model import SimulationConfig
    from fast_dynamics import FastDynamics
    acc = FastDynamics(params, vehicle).acceleration_trig
    cfg = SimulationConfig(dt=run.dt)
    sub, h = cfg.n_substeps, cfg.sub_dt
    qx, qy, qz, qw = (float(x) for x in run.orientation_quat[a])
    v = [float(x) for x in _nu(run)[a]]
    V, Q = np.full((n, 6), np.nan), np.full((n, 4), np.nan)
    for k in range(n):
        V[k], Q[k] = v, (qx, qy, qz, qw)
        if k + 1 == n:
            break
        tau = run.forces_torques[a + k].tolist()
        for _ in range(sub):
            r20, r21, r22 = 2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)
            vd = acc(v, tau, r21, r22, -r20, 1.0)
            v = [v[j] + vd[j] * h for j in range(6)]
            if not all(math.isfinite(x) and abs(x) < 1e3 for x in v):
                return V, Q
            wx, wy, wz = v[3] * h, v[4] * h, v[5] * h
            th = math.sqrt(wx * wx + wy * wy + wz * wz)
            if th > 1e-12:
                c, s_ = math.cos(th / 2), math.sin(th / 2) / th
                dx, dy, dz = wx * s_, wy * s_, wz * s_
                qw, qx, qy, qz = (qw * c - qx * dx - qy * dy - qz * dz, qw * dx + qx * c + qy * dz - qz * dy,
                                  qw * dy - qx * dz + qy * c + qz * dx, qw * dz + qx * dy - qy * dx + qz * c)
        nq = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        qx, qy, qz, qw = qx / nq, qy / nq, qz / nq, qw / nq
    return V, Q


def baselines(runs_wins: list) -> dict:
    zero, keep = [], []
    for run, wins in runs_wins:
        zero.append([(np.zeros((n, 6)), np.repeat(run.orientation_quat[a:a + 1], n, axis=0)) for a, n in wins])
        keep.append([(np.repeat(_nu(run)[a:a + 1], n, axis=0), np.repeat(run.orientation_quat[a:a + 1], n, axis=0))
                     for a, n in wins])
    return {"no motion": score(runs_wins, zero), "keeps its starting velocity": score(runs_wins, keep)}


# ---------------------------------------------------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------------------------------------------------

LINEAR_FLOOR, ANGULAR_FLOOR = 0.05, 0.02  # m/s, rad/s: an axis that barely varies isn't weighted without limit


def axis_scales(velocities: np.ndarray, weighting: str = "variation") -> np.ndarray:
    """What each axis's velocity error is divided by in the fit. 'variation' (default): the axis's standard deviation
    over the training recordings, floored, so a vehicle cruising steadily at 0.7 m/s weighs surge errors by how much
    surge actually changes. 'magnitude': its RMS (Strathe's own objective's scale; #72's first runs), which makes
    surge errors cheap during a cruise and let the fit trade surge for pitch and yaw (#73)."""
    if weighting == "magnitude":
        sc = np.sqrt(np.mean(velocities ** 2, axis=0))
    elif weighting == "variation":
        sc = np.std(velocities, axis=0)
    else:
        raise SimFitError(f"unknown weighting {weighting!r}: use variation or magnitude")
    return np.maximum(sc, np.array([LINEAR_FLOOR] * 3 + [ANGULAR_FLOOR] * 3))


def fit(simulate, train: list, knobs, work: str, workers: int = 4, max_iterations: int = 15, drift_weight: float = 1.0,
        log=print, weighting: str = "variation", checkpoint: str = None, fingerprint: dict = None) -> dict:
    """Least squares over a simulator's own settings on the training recordings (30 s free-running windows every 30 s;
    velocity error per axis scaled by axis_scales, plus end-of-window drift scaled by the no-motion drift). The
    Jacobian's simulator runs go in parallel.

    simulate(x, i, run, rows, folder) -> [(V, Q) per window]: recording i's windows run with the settings for
    parameter vector x (an adapter's model written into `folder`). knobs: names(), values(x), x0(), bounds().

    checkpoint: a JSON file the best point so far is written to after every evaluation that improves it (and the
    result, marked complete, at the end), so an interrupted fit can be finished from it (finish_from_checkpoint, #90:
    a Gazebo fit of bluerov2_gz ran 9.5 h). fingerprint: what the checkpoint must match to be finished."""
    from scipy.optimize import least_squares
    runs_wins = [(run, windows(run, FIT_WINDOW_S, FIT_STRIDE_S)) for run in train]
    if not any(w for _, w in runs_wins):
        raise SimFitError(f"the training recordings are shorter than one {FIT_WINDOW_S:g} s window")
    rows = [window_rows(run, w) for run, w in runs_wins]
    sensor = getattr(train[0], "channels", None) is not None   # no DVL (#96): each sensor against what the simulator would have read
    if sensor:
        from simfit import sensors as SN
        sens_scale, lay = SN.scales(runs_wins), SN.layout(runs_wins)
        scale = np.array([LINEAR_FLOOR] * 3 + [ANGULAR_FLOOR] * 3)   # (unused; recorded for the checkpoint)
        real_paths = [[None] * len(w) for _, w in runs_wins]
        d0 = 1.0
    else:
        scale = axis_scales(np.vstack([run.real_nu for run in train]), weighting)
        real_paths = [[path(run.real_nu[a:a + n], run.orientation_quat[a:a + n], run.dt) for a, n in w]
                      for run, w in runs_wins]
        d0 = max(float(np.median([np.linalg.norm(P[-1]) for ps in real_paths for P in ps])), 0.1)
    counter = {"n": 0}

    def residuals_at(x, tag):
        d = os.path.join(work, f"eval_{tag}")
        out = []
        for i, ((run, wins), rw, rp) in enumerate(zip(runs_wins, rows, real_paths)):
            preds = simulate(x, i, run, rw, d)
            if sensor:
                out += SN.residuals(run, wins, preds, sens_scale, lay[i])
                continue
            for (a, n), (V, Q), Pr in zip(wins, preds, rp):
                if np.isnan(V).any():
                    out.append(np.full(6 * (n - 1) + 1, 10.0))
                    continue
                e = ((V[1:] - run.real_nu[a + 1:a + n]) / scale).ravel() / math.sqrt(n - 1)
                dr = np.linalg.norm(path(V, Q, run.dt)[-1] - Pr[-1]) / d0
                out.append(np.append(e, math.sqrt(drift_weight) * dr))
        shutil.rmtree(d, ignore_errors=True)
        return np.concatenate(out)

    cache = {}
    best = {"cost": math.inf, "x": None, "start_cost": None}

    def save(complete=False, status=None):
        if not checkpoint or best["x"] is None:
            return
        doc = {"fingerprint": fingerprint or {}, "names": knobs.names(), "x": [float(v) for v in best["x"]],
               "values": knobs.values(np.array(best["x"])), "cost": best["cost"], "start_cost": best["start_cost"],
               "evaluations": counter["n"], "axis_scales": scale.tolist(), "complete": complete,
               "status": status or "interrupted: finished from its best point so far, not converged"}
        tmp = checkpoint + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
        os.replace(tmp, checkpoint)  # never left half-written by an interruption

    def fun(x):
        key = tuple(np.round(x, 12))
        if key in cache:
            return cache[key]
        counter["n"] += 1
        r = residuals_at(x, f"f{counter['n']}")
        c = 0.5 * float(r @ r)
        log(f"  evaluation {counter['n']}: cost {c:.4f}")
        if best["start_cost"] is None:
            best["start_cost"] = c
        if c < best["cost"]:
            best["cost"], best["x"] = c, np.array(x, dtype=float)
            save()
        cache.clear()
        cache[key] = r
        return r

    lo, hi = knobs.bounds()
    h = 0.05

    def jac(x):
        f0 = fun(x)
        cols = [None] * len(x)

        def col(j):
            xj = x.copy()
            step = h if x[j] + h <= hi[j] else -h
            xj[j] += step
            return j, (residuals_at(xj, f"j{counter['n']}_{j}") - f0) / step
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for j, c in ex.map(col, range(len(x))):
                cols[j] = c
        return np.column_stack(cols)

    x0 = knobs.x0()
    r0 = fun(x0)  # cached: least_squares' first evaluation reuses it
    res = least_squares(fun, x0, jac=jac, bounds=(lo, hi), method="trf", max_nfev=max_iterations, x_scale=1.0)
    out = {"x": res.x.tolist(), "names": knobs.names(), "values": knobs.values(res.x), "axis_scales": scale.tolist(),
           "start_cost": 0.5 * float(r0 @ r0),
           "cost": float(res.cost), "evaluations": counter["n"], "status": res.message}
    best["x"], best["cost"] = res.x, float(res.cost)
    save(complete=True, status=res.message)
    return out


def finish_from_checkpoint(checkpoint: str, knobs, fingerprint: dict = None) -> dict:
    """The fit result an interrupted fit's checkpoint holds (its best point so far), in fit()'s shape. Refuses a
    checkpoint written for other settings."""
    if not os.path.exists(checkpoint):
        raise SimFitError(f"--finish: no checkpoint ({os.path.basename(checkpoint)}) in {os.path.dirname(checkpoint)}; "
                          f"nothing to finish (did the fit start, with the same --out?)")
    try:
        with open(checkpoint, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise SimFitError(f"--finish: the checkpoint {checkpoint} can't be read ({e})")
    if doc.get("names") != knobs.names():
        raise SimFitError(f"--finish: the checkpoint was written for other settings (it tuned "
                          f"{', '.join(doc.get('names') or [])}; this command tunes {', '.join(knobs.names())}). Give "
                          f"the same --fit, --hold and --axisymmetric as the interrupted run")
    for k, v in (fingerprint or {}).items():
        if doc.get("fingerprint", {}).get(k) not in (None, v):
            raise SimFitError(f"--finish: the checkpoint was written for another {k} ({doc['fingerprint'][k]}, not {v}); "
                              f"give the same settings as the interrupted run")
    x = np.array(doc["x"], dtype=float)
    return {"x": x.tolist(), "names": knobs.names(), "values": knobs.values(x), "axis_scales": doc["axis_scales"],
            "start_cost": doc["start_cost"], "cost": doc["cost"], "evaluations": doc["evaluations"],
            "status": doc["status"], "finished_from_checkpoint": not doc.get("complete")}


# ---------------------------------------------------------------------------------------------------------------------
# Recordings, table and plot
# ---------------------------------------------------------------------------------------------------------------------

def load_runs(data_dir: str, heldout_dates: Optional[list]):
    from data_split import discover_csv_files, split_by_date
    from objective import DatasetConfig, load_run
    files = discover_csv_files(data_dir)[0]
    try:
        split = split_by_date(files, heldout_dates=heldout_dates)
    except ValueError as e:  # e.g. a held-out date the logs don't have: it lists the ones they do (#88, F5)
        raise SimFitError(f"{e} (--heldout-dates takes dates of your recordings)")
    rate = None
    rp = os.path.join(data_dir, "ingest_report.json")
    if os.path.exists(rp):
        rate = json.load(open(rp, encoding="utf-8")).get("sampling_rate_hz")
    cfg = DatasetConfig(float(rate or 10.0), (0.0, 0.0, 0.0), None, None)  # torques as logged, about the reference
    train, held = [load_run(f, cfg) for f in split.train_files], [load_run(f, cfg) for f in split.heldout_files]
    if len({getattr(r, "channels", None) is not None for r in train + held}) > 1:   # one objective for all, as calibrate does
        from dataclasses import replace as _replace
        cfg = _replace(cfg, sensors=("all",))
        train, held = [load_run(f, cfg) for f in split.train_files], [load_run(f, cfg) for f in split.heldout_files]
    if any(getattr(r, "channels", None) is not None for r in train + held):
        # no DVL (#96): each window starts from the estimated velocity (a physics-free smoother: the simulator's
        # model may be of any vehicle), and the simulator is scored against the sensors (simfit/sensors.py)
        from measurements import measured_noise
        from simfit.sensors import set_starts
        info = set_starts(train + held, noise=measured_noise(data_dir))
        if info["fallbacks"]:
            print("  note: the starting-velocity estimate fell back to the DVL (or rest) on "
                  + "; ".join(info["fallbacks"]), flush=True)
    return train, held, split


def table(results: dict, horizons=HORIZONS_S) -> list:
    L = ["| Model | " + " | ".join(f"drift {h} s" for h in horizons) + " | velocity error over 10 s, u v w (m/s) "
         "| roll, pitch, yaw after 10 s | diverged |", "|---|" + "---|" * (len(horizons) + 3)]
    for name, r in results.items():
        cells = [f"{r[f'{h}s']['drift_median_m']:.2f} m" if f"{h}s" in r else "n/a" for h in horizons]
        v = r.get("10s", {}).get("velocity_rmse")
        a = r.get("10s", {}).get("attitude_error_median_deg")
        L.append(f"| {name} | " + " | ".join(cells) + f" | {', '.join(f'{x:.3f}' for x in v[:3]) if v else 'n/a'} "
                 f"| {', '.join(f'{x:.1f}°' for x in a) if a else 'n/a'} | {r['diverged']}/{r['windows']} |")
    return L


def plot(results: dict, path_png: str, title: str = "Unseen recordings: original vs calibrated model") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, r in results.items():
        hs = [h for h in HORIZONS_S if f"{h}s" in r]
        ax.plot(hs, [r[f"{h}s"]["drift_median_m"] for h in hs], marker="o", label=name)
    ax.set_xlabel("prediction horizon [s]"); ax.set_ylabel("median position error [m]")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path_png, dpi=120); plt.close(fig)


# --- what the drift alone hides (#90, bluerov2_gz) -----------------------------------------------------------------
def attitude_worse(original: dict, new: dict) -> list:
    """["pitch 7.8° -> 17.3°", ...]: axes whose median attitude error after 10 s is 25% and 2° worse than the original's."""
    ao = original.get("10s", {}).get("attitude_error_median_deg")
    ax = new.get("10s", {}).get("attitude_error_median_deg")
    if not (ao and ax):
        return []
    return [f"{n} {a:.1f}° -> {b:.1f}°" for n, a, b in zip(("roll", "pitch", "yaw"), ao, ax) if b > 1.25 * a and b - a > 2.0]


def longest_drift(r: dict):
    h = next((f"{h}s" for h in reversed(HORIZONS_S) if f"{h}s" in r), None)
    return r[h]["drift_median_m"] if h else None


def prefer_cg_held(original: dict, tuned: dict, held: dict) -> tuple:
    """(True, why) when the refit with the centre of gravity held is the one to use: attitude no longer worse than the
    original's (or fewer axes worse), drift within 10% of the tuned fit's, and no more blow-ups."""
    wt, wh = attitude_worse(original, tuned), attitude_worse(original, held)
    dt, dh = longest_drift(tuned), longest_drift(held)
    bt, bh = tuned.get("diverged") or 0, held.get("diverged") or 0
    if len(wh) >= len(wt):
        return False, "holding the centre of gravity didn't make the attitude better"
    if dt is not None and dh is not None and dh > 1.1 * dt:
        return False, f"holding it costs too much position accuracy ({dh:.2f} m against {dt:.2f} m)"
    if bh > bt:
        return False, f"it blew up in more unseen windows ({bh} against {bt})"
    return True, ("its attitude is no worse than your original's" if not wh else "fewer axes turn or tilt worse") + \
        (f", and its position error is {dh:.2f} m against {dt:.2f} m" if dt is not None and dh is not None else "")


def report_warnings(results, original: str, new: str, what: str) -> list:
    """What the drift alone hides (#90, bluerov2_gz): attitude that got worse, and standing still doing better."""
    o, x, L = results[original], results[new], []
    worse = attitude_worse(o, x)
    if worse:
        L.append(f"**Be careful:** {what} turns or tilts worse than your original after 10 s ({', '.join(worse)}). "
                 f"The fit traded attitude for position: check the centre of gravity and buoyancy it changed, or keep "
                 f"them (`--fit` without `cg`, or `--hold` your measured values) and compare. (`fit-sim` and "
                 f"`fit-stonefish` do that comparison by themselves unless `--no-cg-rerun` is given.)\n")
    if x.get("diverged") and x.get("windows"):
        n, m = x["diverged"], x["windows"]
        before = o.get("diverged") or 0
        L.append(f"**Don't simulate with it as is: it blew up in {n} of {m} unseen windows** (your original: {before}). "
                 f"The medians above leave those windows out. Some values it found make the simulation unstable in "
                 f"places: check the report's changes for very large damping on a light axis, and try a smaller "
                 f"physics step or `--hold` on those values.\n")
    still = results.get("no motion")
    if still:
        beaten = [h for h in HORIZONS_S if f"{h}s" in still and f"{h}s" in x
                  and still[f"{h}s"]["drift_median_m"] < x[f"{h}s"]["drift_median_m"]]
        if beaten:
            L.append(f"**Standing still beats {what} after {beaten[0]} s** ({still[f'{beaten[0]}s']['drift_median_m']:.2f} "
                     f"m against {x[f'{beaten[0]}s']['drift_median_m']:.2f} m). Usual for logs from a tank or a "
                     f"station-keeping mission, where the vehicle stays within a few metres; trust the shorter "
                     f"horizons and the velocity errors there.\n")
    return L
