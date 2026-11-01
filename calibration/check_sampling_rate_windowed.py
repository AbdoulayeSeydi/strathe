#!/usr/bin/env python3
"""
Windowed version of the sampling-rate check: instead of integrating
position error over an ENTIRE run (which lets any small velocity-sensor
bias compound over 40-80+ seconds, potentially masking the true rate
behind whatever dt best cancels that session's particular drift), this
resets the reconstructed position to the REAL logged position every
`window_seconds`, so drift can never accumulate beyond one short window,
anywhere in the run. Uses ALL the motion in each file (many short windows
per run), not just the start - some runs sit stationary for several
seconds initially, which would make a "first N seconds only" test
meaningless.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def reconstruct_position_error_windowed(df, dt, window_seconds):
    Bu, Bv, Bw = df["Bu"].to_numpy(), df["Bv"].to_numpy(), df["Bw"].to_numpy()
    qx, qy, qz, qw = df["qx"].to_numpy(), df["qy"].to_numpy(), df["qz"].to_numpy(), df["qw"].to_numpy()
    x, y, z = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()

    n = len(df)
    rot = Rotation.from_quat(np.stack([qx, qy, qz, qw], axis=-1))
    body_vel = np.stack([Bu, Bv, Bw], axis=-1)
    world_vel = rot.apply(body_vel)
    real_pos = np.stack([x, y, z], axis=-1)

    window_samples = max(1, int(round(window_seconds / dt)))
    all_sq_errors = []

    for start in range(0, n - 1, window_samples):
        end = min(start + window_samples, n)
        pos_recon = np.zeros((end - start, 3))
        pos_recon[0] = real_pos[start]
        for k in range(end - start - 1):
            pos_recon[k + 1] = pos_recon[k] + world_vel[start + k] * dt
        errors = pos_recon - real_pos[start:end]
        all_sq_errors.append(np.sum(errors ** 2, axis=1))

    all_sq_errors = np.concatenate(all_sq_errors)
    return float(np.sqrt(np.mean(all_sq_errors)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--candidates", type=float, nargs="+", required=True)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-runs", type=int, default=39)
    args = parser.parse_args()

    csv_paths = sorted(str(p) for p in Path(args.data_dir).rglob("*.csv"))
    test_paths = csv_paths[:args.max_runs]
    need = {"Bu", "Bv", "Bw", "x", "y", "z"}
    lacking = [p for p in test_paths if not need <= set(pd.read_csv(p, nrows=0).columns)]
    if lacking:
        # the rate is inferred by integrating logged velocity against logged position: logs without both (no DVL,
        # #96, or no positions) can't be checked this way
        print(f"Can't infer the sampling rate from {len(lacking)} of {len(test_paths)} file(s): this check needs body "
              f"velocity (Bu, Bv, Bw) and positions (x, y, z) in each. Give the rate with --sampling-rate-hz (or "
              f"convert the logs with `strathe ingest`, which measures it from their timestamps).")
        raise SystemExit(1)
    print(f"Testing against {len(test_paths)} run(s), window={args.window_seconds}s\n")

    print(f"{'Rate (Hz)':>10}{'dt (s)':>10}{'Windowed pos RMSE (m)':>24}")
    print("-" * 44)
    results = []
    for hz in args.candidates:
        dt = 1.0 / hz
        errors = [reconstruct_position_error_windowed(pd.read_csv(fp), dt, args.window_seconds)
                  for fp in test_paths]
        mean_err = float(np.mean(errors))
        results.append((hz, mean_err))
        print(f"{hz:>10.1f}{dt:>10.4f}{mean_err:>24.4f}")

    best_hz, best_err = min(results, key=lambda r: r[1])
    print(f"\nBest candidate: {best_hz:.1f} Hz, windowed RMSE = {best_err:.4f} m")


if __name__ == "__main__":
    main()