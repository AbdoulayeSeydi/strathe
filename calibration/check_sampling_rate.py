#!/usr/bin/env python3
"""
Empirically estimates the dataset's true sampling rate - never confirmed
externally (no time column, no associated paper; ASSUMED_DT=50Hz has been
used as a flagged placeholder throughout every result so far).

Method: the dataset logs BOTH body-frame velocity (Bu,Bv,Bw) AND world-frame
position (x,y,z) AND orientation (qx,qy,qz,qw). Velocity integrates to
position - that's pure kinematics, true regardless of the vehicle's actual
dynamics. So: rotate logged body-frame velocity into world frame using the
logged orientation, forward-integrate it at each CANDIDATE dt, and compare
the reconstructed position trajectory to the ACTUAL logged position. The
candidate dt that reconstructs position most accurately is the best
empirical estimate of the true sampling rate.

This is completely independent of dynamics_model.py, the calibration, or
any damping/inertia assumption - it only assumes the logged position is
itself a reasonably accurate reference, which is the one caveat worth
keeping in mind.

Usage:
    python3 check_sampling_rate.py --data-dir ../data/raw/Bluerov2_trajectory_dataset --candidates 10 20 25 30 40 50 60 100 --max-runs 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from data_split import discover_csv_files


def reconstruct_position_error(df, dt):
    Bu, Bv, Bw = df["Bu"].to_numpy(), df["Bv"].to_numpy(), df["Bw"].to_numpy()
    qx, qy, qz, qw = df["qx"].to_numpy(), df["qy"].to_numpy(), df["qz"].to_numpy(), df["qw"].to_numpy()
    x, y, z = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()

    n = len(df)
    rot = Rotation.from_quat(np.stack([qx, qy, qz, qw], axis=-1))
    body_vel = np.stack([Bu, Bv, Bw], axis=-1)
    world_vel = rot.apply(body_vel)

    pos_recon = np.zeros((n, 3))
    pos_recon[0] = [x[0], y[0], z[0]]
    for k in range(n - 1):
        pos_recon[k + 1] = pos_recon[k] + world_vel[k] * dt

    real_pos = np.stack([x, y, z], axis=-1)
    return float(np.sqrt(np.mean(np.sum((pos_recon - real_pos) ** 2, axis=1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--candidates", type=float, nargs="+",
                         default=[10, 20, 25, 30, 40, 50, 60, 100])
    parser.add_argument("--max-runs", type=int, default=5)
    args = parser.parse_args()

    csv_paths, duplicates = discover_csv_files(args.data_dir)
    for dup_path, kept_path in duplicates:
        print(f"Duplicate filename skipped: {dup_path} (kept {kept_path})")

    test_paths = csv_paths[:args.max_runs]
    print(f"Testing against {len(test_paths)} run(s):")
    for p in test_paths:
        print(f"  {Path(p).name}")
    print()

    print(f"{'Rate (Hz)':>10}{'dt (s)':>10}{'Mean position RMSE (m)':>26}")
    print("-" * 46)
    results = []
    for hz in args.candidates:
        dt = 1.0 / hz
        errors = [reconstruct_position_error(pd.read_csv(fp), dt) for fp in test_paths]
        mean_err = float(np.mean(errors))
        results.append((hz, mean_err))
        print(f"{hz:>10.1f}{dt:>10.4f}{mean_err:>26.4f}")

    best_hz, best_err = min(results, key=lambda r: r[1])
    print(f"\nBest candidate: {best_hz:.1f} Hz (dt={1/best_hz:.4f}s), mean position RMSE = {best_err:.4f} m")
    print("Lower reconstruction error = more consistent with being the true sampling rate.")
    print("\nCaveat: this assumes the logged position (x,y,z) is itself a reasonably accurate")
    print("reference. If it comes from something that drifts on its own (e.g. dead-reckoning")
    print("rather than an external tracking system), this comparison inherits that uncertainty.")


if __name__ == "__main__":
    main()
