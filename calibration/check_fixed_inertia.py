#!/usr/bin/env python3
"""
Tests fixing Ieff_roll/Ieff_pitch at their REAL, independently-sourced
physical values (0.26, 0.23 kg*m^2 - a public BlueROV2 Gazebo model,
same "heavy" configuration) instead of letting them float to the
non-physical 9-10 kg*m^2 the recent fits converged to. Directly checks:

1. Does fixing them cost anything in aggregate RMSE? (Sensitivity
   analysis suggests: probably not.)
2. Does it specifically change r/yaw performance? (Motivated by the
   Coriolis yaw term's dependence on Ieff_pitch - Ieff_roll - see
   PROJECT_STATUS.md finding #16.)

Fast - fixed_params support means the optimizer only re-fits the OTHER
20 parameters, on a small run subset for speed.

Usage:
    python check_fixed_inertia.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0
"""
import argparse

import numpy as np

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, rmse_by_dof
from optimizer import load_vehicle_spec, CalibrationConfig, calibrate

DOF_LABELS = ["u", "v", "w", "p", "q", "r"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--max-nfev", type=int, default=10)
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files[:args.n_runs]]
    print(f"Using {len(train_runs)} training run(s) — fixing Ieff_roll=0.26, Ieff_pitch=0.23 "
          f"(real, independently-sourced BlueROV2 values), re-fitting the other 20 parameters\n")

    loaded = load_vehicle_spec(args.spec)
    vehicle = loaded.vehicle

    def progress(call_count, param_vec, cost):
        print(f"  eval {call_count}: cost={cost:.4f}")

    config = CalibrationConfig(
        n_restarts=1, seed=42, max_nfev=args.max_nfev,
        fixed_params={"Ieff_roll": 0.26, "Ieff_pitch": 0.23},
        on_iteration=progress,
    )
    try:
        result = calibrate(loaded, train_runs, config)
        converged = result.converged
    except RuntimeError as e:
        print(f"Did not converge in budget: {str(e)[:100]}")
        return

    print(f"\nConverged: {converged}, cost: {result.cost:.6f}, train RMSE: {result.train_rmse:.4f}")

    per_dof = rmse_by_dof(result.parameters, train_runs[0], vehicle) if train_runs else None
    print(f"\nPer-DOF RMSE (fixed-inertia fit, run 0 only, quick check):")
    for i, label in enumerate(DOF_LABELS):
        print(f"  {label}: {per_dof[i]:.4f}")

    print(f"\nCompare this train RMSE against an equivalent UNFIXED fit on the same subset")
    print(f"to see the real cost (if any) of forcing physically-real inertia.")


if __name__ == "__main__":
    main()