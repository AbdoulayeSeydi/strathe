#!/usr/bin/env python3
"""
Per-run breakdown of windowed position-reconstruction error at a single
candidate rate, so we can check whether 10Hz is consistently good across
EVERY individual run, not just on average per date - a date-level average
could hide one bad outlier run.

Usage:
    python3 check_sampling_rate_per_run.py \\
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \\
        --rate-hz 10.0 --window-seconds 2.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from data_split import discover_csv_files


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
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    args = parser.parse_args()

    csv_paths, duplicates = discover_csv_files(args.data_dir)
    for dup_path, kept_path in duplicates:
        print(f"Duplicate filename skipped: {dup_path} (kept {kept_path})")

    dt = 1.0 / args.rate_hz
    results = []
    for fp in csv_paths:
        df = pd.read_csv(fp)
        err = reconstruct_position_error_windowed(df, dt, args.window_seconds)
        results.append((Path(fp).name, err))

    results.sort(key=lambda r: r[1])
    errs = np.array([r[1] for r in results])
    mean, std = errs.mean(), errs.std()

    print(f"\nPer-run windowed position RMSE at {args.rate_hz} Hz (window={args.window_seconds}s):\n")
    print(f"{'run':<40}{'RMSE (m)':>12}")
    print("-" * 55)
    outlier_count = 0
    for name, err in results:
        is_outlier = err > mean + 2 * std
        flag = "  <-- outlier (>2 std above mean)" if is_outlier else ""
        if is_outlier:
            outlier_count += 1
        print(f"{name:<40}{err:>12.4f}{flag}")

    print(f"\nMean: {mean:.4f}  Std: {std:.4f}  Min: {errs.min():.4f}  Max: {errs.max():.4f}")
    print(f"Runs flagged as outliers (>2 std above mean): {outlier_count}/{len(results)}")


if __name__ == "__main__":
    main()