#!/usr/bin/env python3
"""
diagnose_rotational_weakness.py -- why do calibrated roll/yaw rates look
WORSE than default physics in HoloOcean (README, Claim A) when the same
calibrations improve them by 50-80% on held-out data with stand-in physics?

Hypothesis tested here: holoocean_backend.py hands HoloOcean a world-frame
linear acceleration computed as R @ v_dot, missing the rotating-frame
transport term R @ (omega x v). The correct conversion of a body-frame
velocity derivative to a world-frame acceleration is
    d/dt (R v) = R (v_dot + omega x v).
The same omission in mujoco_backend.py was found and fixed against real
MuJoCo (tests/test_mujoco_transport.py). With
it missing, every tick the world-frame velocity fails to rotate with the
vehicle, so surge and sway leak into each other while turning, and through
the Coriolis terms that error feeds every rotational axis.

HoloOcean itself doesn't run on macOS, so this script reproduces
HoloOcean's Custom Dynamics SCHEME (see holoocean_backend.simulate_trajectory_holoocean):
once per 0.1 s data tick, compute body-frame acceleration from the current
state, convert it to world frame, hold it constant while integrating a free
rigid body (orientation evolves, velocity read back in body frame). It does
not reproduce HoloOcean's own integrator; it isolates the frame conversion.

Replays real held-out logs (dates 16-02-2024, 19-02-2024) open-loop, three ways:
    stand_in       dynamics_model teacher-forced replay (the calibration objective)
    ho_original    HoloOcean scheme, R @ v_dot           (what holoocean_backend does)
    ho_fixed       HoloOcean scheme, R @ (v_dot + w x v) (kinematically correct)
and reports per-axis RMSE improvement of each calibration over generic physics.

    python diagnose_rotational_weakness.py [--window 150] [--workers 7] [--engine-tick 0.0333]

## Result (2026-09-23)
- The calibration itself is not weak on roll/yaw. With the stand-in physics
  it was fitted with, every converged calibration improves held-out roll rate
  by ~82% and yaw rate by 23-51% over generic physics.
- The emulated HoloOcean scheme DIVERGES for every physically plausible
  calibration (roll/pitch inertia ~0.25 kg m^2), with or without the transport
  term, at both a 0.1 s and a 1/30 s command update: acceleration recomputed
  once per tick is explicitly unstable against those damping/inertia ratios.
  Real HoloOcean did produce finite numbers for the same parameters, so its
  integration differs from this emulation in some way that matters -- this
  script can't reproduce the Claim A numbers, and its ho_* rows mostly print
  NaN. Only the 10hz_coriolis fit (non-physical inertia 5.0) survives.
- What IS established: holoocean_backend omitted the transport term (fixed,
  default on, unconfirmed in HoloOcean), and its tick rate was never checked
  against the 10 Hz data (guard added). verify_holoocean_frame_conventions.py
  measures both on real HoloOcean; the roll/yaw question should be re-asked
  only after that passes.
"""
from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

from data_split import discover_csv_files, split_by_date
from dynamics_model import CalibrationParameters
from fast_dynamics import FastDynamics, _matmul3, _rotvec_matrix, roll_pitch_from_matrix
from objective import DatasetConfig, load_run, rmse_by_dof
from optimizer import initial_parameters, load_vehicle_spec

DATA_DIR = "../data/raw/Bluerov2_trajectory_dataset"
HELDOUT = ["16-02-2024", "19-02-2024"]
CANDIDATES = {
    "generic": None,
    "holoocean_coriolis (Claim A)": "../calibration_output_holoocean_coriolis/calibrated_parameters.yaml",
    "expA (best converged)": "../calibration_output_expA/calibrated_parameters.yaml",
    "mujoco_full": "../calibration_output_mujoco_full/calibrated_parameters.yaml",
    "10hz_coriolis": "../calibration_output_10hz_coriolis/calibrated_parameters.yaml",
}
DOFS = "u v w p q r".split()


def replay_holoocean_scheme(params, vehicle, forces, roll_pitch0, nu0, dt, transport: bool, n_sub: int = 20,
                            engine_tick_s: float = None):
    """Open-loop replay with HoloOcean's scheme. Returns (N,6) body velocities.
    engine_tick_s: simulated time HoloOcean advances per env.step() (one step
    per logged sample). None = the data's own dt. HoloOcean's default
    ticks_per_sec is 30 (1/30 s per step) unless the scenario overrides it."""
    dyn = FastDynamics(params, vehicle)
    r0, p0 = roll_pitch0
    cr, sr, cp, sp = math.cos(r0), math.sin(r0), math.cos(p0), math.sin(p0)
    # R = Ry(pitch) Rx(roll), yaw 0 -- same as holoocean_backend's initial state
    R = [[cp, sp * sr, sp * cr], [0.0, cr, -sr], [-sp, cp * sr, cp * cr]]
    V = [sum(R[i][j] * nu0[j] for j in range(3)) for i in range(3)]          # world linear velocity
    W = [sum(R[i][j] * nu0[3 + j] for j in range(3)) for i in range(3)]      # world angular velocity
    n = forces.shape[0]
    out = np.zeros((n, 6))
    out[0] = nu0
    h = (engine_tick_s or dt) / n_sub
    for k in range(n - 1):
        vb = [sum(R[j][i] * V[j] for j in range(3)) for i in range(3)]
        wb = [sum(R[j][i] * W[j] for j in range(3)) for i in range(3)]
        roll, pitch = roll_pitch_from_matrix(R)
        vd = dyn.acceleration(vb + wb, forces[k], roll, pitch)
        lin = vd[:3]
        if transport:
            lin = [lin[0] + wb[1] * vb[2] - wb[2] * vb[1],
                   lin[1] + wb[2] * vb[0] - wb[0] * vb[2],
                   lin[2] + wb[0] * vb[1] - wb[1] * vb[0]]
        a_w = [sum(R[i][j] * lin[j] for j in range(3)) for i in range(3)]
        al_w = [sum(R[i][j] * vd[3 + j] for j in range(3)) for i in range(3)]
        for _ in range(n_sub):  # engine integrates with the command held constant in WORLD frame
            V = [V[i] + a_w[i] * h for i in range(3)]
            W = [W[i] + al_w[i] * h for i in range(3)]
            R = _matmul3(_rotvec_matrix(W[0] * h, W[1] * h, W[2] * h), R)
        vb = [sum(R[j][i] * V[j] for j in range(3)) for i in range(3)]
        wb = [sum(R[j][i] * W[j] for j in range(3)) for i in range(3)]
        out[k + 1] = vb + wb
        if not np.all(np.isfinite(out[k + 1])) or np.abs(out[k + 1]).max() > 1e3:
            return None  # diverged
    return out


def _windows(run, window):
    if window is None or window >= run.n_samples:
        return [(0, run.n_samples)]
    return [(s, s + window) for s in range(0, run.n_samples - window + 1, window)]


def evaluate(job):
    name, path, window, engine_tick_s = job
    spec = load_vehicle_spec("vehicle_spec.yaml")
    params = initial_parameters(spec) if path is None else CalibrationParameters(**yaml.safe_load(open(path))["parameters"])
    files = discover_csv_files(DATA_DIR)[0]
    runs = [load_run(f, DatasetConfig(10.0)) for f in split_by_date(files, heldout_dates=HELDOUT).heldout_files]
    res = {"stand_in": [], "ho_original": [], "ho_fixed": []}
    diverged = {"ho_original": 0, "ho_fixed": 0}
    for run in runs:
        res["stand_in"].append(rmse_by_dof(params, run, spec.vehicle))
        for a, b in _windows(run, window):
            for eng, transport in (("ho_original", False), ("ho_fixed", True)):
                sim = replay_holoocean_scheme(params, spec.vehicle, run.forces_torques[a:b], run.roll_pitch[a],
                                              run.real_nu[a], run.dt, transport, engine_tick_s=engine_tick_s)
                if sim is None:
                    diverged[eng] += 1
                    continue
                res[eng].append(np.sqrt(np.mean((sim - run.real_nu[a:b]) ** 2, axis=0)))
    return name, {k: np.mean(v, axis=0) if v else np.full(6, np.nan) for k, v in res.items()}, diverged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=150,
                    help="replay window in samples (150 = HoloOcean calibration's 15 s); 0 = whole runs")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--engine-tick", type=float, default=None,
                    help="seconds HoloOcean advances per logged sample (default: the data's 0.1 s; "
                         "HoloOcean's own default ticks_per_sec=30 would be 0.0333)")
    args = ap.parse_args()
    window = args.window or None
    with ProcessPoolExecutor(args.workers) as pool:
        results = {name: (r, d) for name, r, d in pool.map(evaluate, [(n, p, window, args.engine_tick) for n, p in CANDIDATES.items()])}
    base = results["generic"][0]
    print(f"\nHeld-out per-axis improvement over generic physics "
          f"({'whole runs' if window is None else f'{window}-sample windows'}, engine tick "
          f"{args.engine_tick or 0.1:.4f} s); negative = calibration is WORSE\n")
    for eng, title in (("stand_in", "stand-in, teacher-forced (calibration objective)"),
                       ("ho_original", "HoloOcean scheme, as implemented (R @ v_dot)"),
                       ("ho_fixed", "HoloOcean scheme, with the w x v transport term")):
        print(f"--- {title}")
        print(f"    {'':30s}" + "".join(f"{d:>8s}" for d in DOFS) + "   diverged windows")
        for name, (r, d) in results.items():
            imp = 100 * (base[eng] - r[eng]) / base[eng]
            div = "" if eng == "stand_in" else f"   {d[eng]}"
            print(f"    {name:30s}" + "".join(f"{x:+8.1f}" for x in imp) + div)
        print()


if __name__ == "__main__":
    main()
