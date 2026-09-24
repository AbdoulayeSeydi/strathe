#!/usr/bin/env python3
"""
fit_rollpitch_disturbance.py -- fit the roll/pitch disturbance (disturbance.py)
to the real logs, for one calibrated parameter set.

    python fit_rollpitch_disturbance.py [--params calibrated_parameters_fleet_truth.yaml]
                                        [--data-dir ../data/raw/Bluerov2_trajectory_dataset]
                                        [--out disturbances/bluerov2_rollpitch.yaml] [--workers 6]

Method (PROJECT_STATUS.md #51):
1. Cut every run into 60 s windows. Replay each window through the calibrated
   physics with orientation evolving freely (as in a fleet simulation),
   starting from the real state, driven by the LOGGED thrust plus a candidate
   disturbance.
2. Compare the simulated roll and pitch against the real ones on three
   statistics per axis: the spread of the rate, the rate's autocorrelation
   at 0.1-1.5 s lags (its "shape" in time), and the spread of the angle.
3. Pick sigma (torque size) and tau (correlation time) per axis on the
   TRAINING dates by a coarse-to-fine grid search; report the same
   statistics on the HELD-OUT dates, with and without the disturbance, and
   in MuJoCo, as the check.

The fit is specific to the damping/restoring parameters it was made with:
the output YAML records a hash of that parameter file, and loading it with a
different truth file warns.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

WINDOW = 600                                   # 60 s at 10 Hz
ACF_LAGS = (1, 2, 3, 5, 7, 10, 15)            # 0.1 .. 1.5 s
HELDOUT = ["16-02-2024", "19-02-2024"]
AXES = {"roll": (0, 3), "pitch": (1, 4)}      # (angle column, rate column)


TORQUE_OFFSET = [0.0, 0.0, -0.085]   # set from --torque-reference-offset-m in main()


def _load(params_path, data_dir, dates_filter, torque_offset=None):
    from data_split import discover_csv_files, split_by_date
    from dynamics_model import CalibrationParameters
    from objective import DatasetConfig, load_run
    from optimizer import load_vehicle_spec
    spec = load_vehicle_spec("vehicle_spec.yaml")
    with open(params_path, encoding="utf-8") as f:
        params = CalibrationParameters(**yaml.safe_load(f)["parameters"])
    split = split_by_date(discover_csv_files(data_dir)[0], heldout_dates=HELDOUT)
    files = split.train_files if dates_filter == "train" else split.heldout_files
    dc = DatasetConfig(10.0, tuple(TORQUE_OFFSET if torque_offset is None else torque_offset))
    return params, spec.vehicle, [load_run(f, dc) for f in files]


def windows(runs, max_windows=None):
    out = [(ri, a) for ri, r in enumerate(runs) for a in range(0, r.n_samples - WINDOW + 1, WINDOW)]
    if max_windows and len(out) > max_windows:
        idx = np.linspace(0, len(out) - 1, max_windows).round().astype(int)
        out = [out[i] for i in idx]
    return out


def _acf(x, lags):
    x = x - x.mean()
    d = float((x * x).sum()) or 1e-12
    return np.array([float((x[:-l] * x[l:]).sum()) / d for l in lags])


def stats(rates_and_angles):
    """rates_and_angles: list of (N,2) rate [p,q] and (N,2) angle [roll,pitch] arrays per window."""
    out = {}
    for ax, k in (("roll", 0), ("pitch", 1)):
        rate = [w[0][:, k] for w in rates_and_angles]
        ang = [w[1][:, k] for w in rates_and_angles]
        out[ax] = {"rate_std": float(np.sqrt(np.mean([np.var(x) for x in rate]))),
                   "angle_std_deg": float(np.degrees(np.sqrt(np.mean([np.var(x) for x in ang])))),
                   "rate_acf": np.mean([_acf(x, ACF_LAGS) for x in rate], axis=0).tolist()}
    return out


def real_windows(runs, wins):
    return [(runs[ri].real_nu[a:a + WINDOW, 3:5], runs[ri].roll_pitch[a:a + WINDOW]) for ri, a in wins]


def simulate(job):
    """job = (params_path, data_dir, split, wins, roll(sigma,tau), pitch(sigma,tau), engine, seed)."""
    params_path, data_dir, split, wins, roll, pitch, engine, seed, torque_offset = job
    from disturbance import AxisDisturbance, DisturbanceConfig, RollPitchDisturbance
    from dynamics_model import SimulationConfig
    params, vehicle, runs = _load(params_path, data_dir, split, torque_offset)
    cfg = DisturbanceConfig(AxisDisturbance(*roll), AxisDisturbance(*pitch))
    sc = SimulationConfig(dt=0.1)
    if engine == "mujoco":
        from mujoco_backend import MuJoCoArenaCore
    else:
        from fast_dynamics import FastCore
    out, diverged = [], 0
    for wi, (ri, a) in enumerate(wins):
        run = runs[ri]
        r0, p0 = run.roll_pitch[a]
        rot = Rotation.from_euler("xyz", [r0, p0, 0.0])
        dist = RollPitchDisturbance(cfg, 0.1, np.random.SeedSequence([seed, ri, a]))
        rates, angs = np.zeros((WINDOW, 2)), np.zeros((WINDOW, 2))
        try:
            if engine == "mujoco":
                core = MuJoCoArenaCore(params, vehicle, sc)
                core.reset(nu=run.real_nu[a])
                from mujoco_backend import set_body_pose
                set_body_pose(core._model, core._data, 1, np.zeros(3), rot)
                core._data.qvel[0:3] = rot.apply(run.real_nu[a][:3])
            else:
                core = FastCore(params, vehicle, sc)
                core.reset(nu=run.real_nu[a], orientation=rot)
            rates[0], angs[0] = run.real_nu[a][3:5], (r0, p0)
            for k in range(1, WINDOW):
                core.step(run.forces_torques[a + k - 1] + dist.next())
                if engine == "mujoco":
                    nu = core.state.nu
                    rr, pp, _ = core.state.orientation.as_euler("xyz")
                else:
                    nu = core.nu_array()
                    rr, pp = core.roll_pitch()
                rates[k], angs[k] = nu[3:5], (rr, pp)
        except FloatingPointError:
            diverged += 1
            continue
        out.append((rates, angs))
    return stats(out), diverged


def score(sim, real, ax):
    s, r = sim[ax], real[ax]
    return (np.log(s["rate_std"] / r["rate_std"]) ** 2
            + np.log(s["angle_std_deg"] / r["angle_std_deg"]) ** 2
            + float(np.mean((np.array(s["rate_acf"]) - np.array(r["rate_acf"])) ** 2)) * 4.0)


def grid_search(params_path, data_dir, wins, real, sigmas, taus, workers, seed=1):
    combos = list(itertools.product(sigmas, taus))
    jobs = [(params_path, data_dir, "train", wins, (s, t), (s, t), "stand_in", seed, TORQUE_OFFSET)
            for s, t in combos]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(simulate, jobs))
    best = {}
    for ax in AXES:
        scored = sorted(((score(res[0], real, ax), c) for c, res in zip(combos, results)), key=lambda x: x[0])
        best[ax] = scored[0]
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default="calibrated_parameters_fleet_truth.yaml")
    ap.add_argument("--data-dir", default="../data/raw/Bluerov2_trajectory_dataset")
    ap.add_argument("--out", default="disturbances/bluerov2_rollpitch.yaml")
    ap.add_argument("--report-dir", default="results/rollpitch_disturbance")
    ap.add_argument("--max-windows", type=int, default=60, help="training windows used per candidate")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--torque-reference-offset-m", type=float, nargs=3, default=[0.0, 0.0, -0.085],
                    help="re-express the logged torques about this point (see objective.shift_torque_reference); "
                         "the default fixes the reference BlueROV2 dataset's 8.5 cm moment arm (PROJECT_STATUS.md #50)")
    args = ap.parse_args(argv)
    TORQUE_OFFSET[:] = args.torque_reference_offset_m
    t0 = time.time()
    from disturbance import params_sha

    _, _, train_runs = _load(args.params, args.data_dir, "train")
    _, _, held_runs = _load(args.params, args.data_dir, "heldout")
    tw, hw = windows(train_runs, args.max_windows), windows(held_runs)
    real_train, real_held = stats(real_windows(train_runs, tw)), stats(real_windows(held_runs, hw))
    print(f"{len(tw)} training / {len(hw)} held-out 60 s windows", flush=True)

    coarse = grid_search(args.params, args.data_dir, tw, real_train,
                         sigmas=[0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4], taus=[0.3, 1.0, 3.0, 10.0, 30.0],
                         workers=args.workers)
    print("coarse:", {ax: (round(v[0], 4), v[1]) for ax, v in coarse.items()}, flush=True)
    fitted = {}
    for ax in AXES:
        s0, t0_ = coarse[ax][1]
        fine = grid_search(args.params, args.data_dir, tw, real_train,
                           sigmas=[s0 * f for f in (0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6)],
                           taus=[t0_ * f for f in (0.5, 0.7, 1.0, 1.4, 2.0)], workers=args.workers)
        fitted[ax] = fine[ax]
        print(f"fine {ax}: score {fine[ax][0]:.4f} sigma {fine[ax][1][0]:.3f} N*m tau {fine[ax][1][1]:.2f} s",
              flush=True)

    roll, pitch = fitted["roll"][1], fitted["pitch"][1]
    checks = {}
    for name, (split, wins, r_, p_, eng) in {
        "held-out, no disturbance": ("heldout", hw, (0.0, 1.0), (0.0, 1.0), "stand_in"),
        "held-out, fitted disturbance": ("heldout", hw, roll, pitch, "stand_in"),
        "held-out, fitted disturbance, MuJoCo": ("heldout", hw[::3], roll, pitch, "mujoco"),
    }.items():
        checks[name] = simulate((args.params, args.data_dir, split, wins, r_, p_, eng, 2, TORQUE_OFFSET))
    real_mj = stats(real_windows(held_runs, hw[::3]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    doc = {
        "roll": {"sigma_nm": round(float(roll[0]), 4), "tau_s": round(float(roll[1]), 3)},
        "pitch": {"sigma_nm": round(float(pitch[0]), 4), "tau_s": round(float(pitch[1]), 3)},
        "fitted_for": {"params_file": os.path.basename(args.params), "params_sha": params_sha(args.params),
                       "torque_reference_offset_m": list(TORQUE_OFFSET)},
        "fit": {"date": str(date.today()), "method": "fit_rollpitch_disturbance.py (PROJECT_STATUS.md #51)",
                "training_windows": len(tw), "heldout_windows": len(hw), "window_s": WINDOW / 10},
        "real_heldout": real_held,
        "sim_heldout": checks["held-out, fitted disturbance"][0],
        # without the disturbance, for comparison: a disturbance only ADDS motion, so where the
        # model already over-tilts (angle spread above the real one) it can't fix that
        "sim_heldout_no_disturbance": checks["held-out, no disturbance"][0],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Roll/pitch disturbance fitted to the real BlueROV2 logs. See disturbance.py.\n")
        yaml.safe_dump(doc, f, sort_keys=False)

    os.makedirs(args.report_dir, exist_ok=True)
    lines = ["# Roll/pitch disturbance fit\n",
             f"Parameters: `{args.params}`. Fitted on {len(tw)} training windows (60 s), checked on {len(hw)} "
             f"held-out windows. Fitted: roll sigma {roll[0]:.3f} N*m, tau {roll[1]:.2f} s; pitch sigma "
             f"{pitch[0]:.3f} N*m, tau {pitch[1]:.2f} s.\n",
             "| Held-out statistic | Real | No disturbance | Fitted disturbance | Fitted, MuJoCo (real, same windows) |",
             "|---|---|---|---|---|"]
    for ax in AXES:
        for key, label, fmt in (("rate_std", f"{ax} rate std (rad/s)", "{:.3f}"),
                                ("angle_std_deg", f"{ax} angle std (deg)", "{:.1f}")):
            row = [fmt.format(real_held[ax][key])] + [fmt.format(checks[n][0][ax][key]) for n in checks]
            lines.append(f"| {label} | {row[0]} | {row[1]} | {row[2]} | {row[3]} ({fmt.format(real_mj[ax][key])}) |")
        for li, lag in enumerate(ACF_LAGS):
            if lag in (2, 5, 10):
                vals = [real_held[ax]["rate_acf"][li]] + [checks[n][0][ax]["rate_acf"][li] for n in checks]
                lines.append(f"| {ax} rate autocorrelation at {lag / 10:.1f} s | {vals[0]:+.2f} | {vals[1]:+.2f} | "
                             f"{vals[2]:+.2f} | {vals[3]:+.2f} ({real_mj[ax]['rate_acf'][li]:+.2f}) |")
    lines.append(f"\nDiverged windows: " + ", ".join(f"{n}: {checks[n][1]}" for n in checks))
    lines.append(f"\nElapsed {time.time() - t0:.0f} s.")
    with open(os.path.join(args.report_dir, "fit_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.report_dir, "fit_details.json"), "w", encoding="utf-8") as f:
        json.dump({"coarse": {k: [v[0], list(v[1])] for k, v in coarse.items()},
                   "fitted": {k: [v[0], list(v[1])] for k, v in fitted.items()},
                   "checks": {k: v[0] for k, v in checks.items()}, "real_train": real_train,
                   "real_heldout": real_held}, f, indent=1)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    from keep_awake import keep_awake
    with keep_awake("roll/pitch disturbance fit"):
        rc = main()
    sys.exit(rc)
