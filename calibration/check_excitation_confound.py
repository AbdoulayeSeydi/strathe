#!/usr/bin/env python3
"""
Checks whether the "worst run" used for the force-breakdown rotational-
dynamics diagnosis was actually well-excited on the vertical thrusters
(pwm5-8, which drive heave/roll/pitch), or whether its high RMSE is partly
an artifact of low real excitation on those specific axes — a run with a
small, quiet real signal can produce a noisy-looking RMSE without the
model actually being worse at explaining it.

Ranks all runs by vertical-thruster excitation (mean std of pwm5-8) against
roll/pitch RMSE, reports the correlation, and shows where the previously-
flagged worst run falls on the excitation ranking.

Usage:
    python3 check_excitation_confound.py \\
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \\
        --spec vehicle_spec.yaml \\
        --calibrated-params ../calibration_output/calibrated_parameters.yaml \\
        --sampling-rate-hz 50 \\
        --worst-run odom-12-01-2024-run5.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_split import discover_csv_files
from dynamics_model import CalibrationParameters
from objective import DatasetConfig, load_run, rmse_by_dof
from optimizer import load_vehicle_spec


def load_calibrated_parameters(path: str) -> CalibrationParameters:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CalibrationParameters(**data["parameters"])


def vertical_thruster_excitation(filepath: str) -> float:
    """Mean std across pwm5-8 (the vertical thrusters driving
    heave/roll/pitch) — raw PWM columns, not part of RunData's six
    force/torque representation, so read directly from the CSV here."""
    df = pd.read_csv(filepath)
    pwm_cols = [f"pwm{i}" for i in range(5, 9)]
    missing = [c for c in pwm_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{filepath} missing expected PWM columns: {missing}")
    return float(df[pwm_cols].std().mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--calibrated-params", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--worst-run", default=None,
                         help="Filename or substring of the previously-flagged worst run")
    args = parser.parse_args()

    loaded_spec = load_vehicle_spec(args.spec)
    calibrated = load_calibrated_parameters(args.calibrated_params)
    vehicle = loaded_spec.vehicle
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)

    csv_paths, duplicates = discover_csv_files(args.data_dir)
    for dup_path, kept_path in duplicates:
        print(f"Duplicate filename skipped: {dup_path} (kept {kept_path})")

    rows = []
    for fp in csv_paths:
        run = load_run(fp, dataset_config)
        dof_rmse = rmse_by_dof(calibrated, run, vehicle)
        excitation = vertical_thruster_excitation(fp)
        rows.append({
            "filepath": fp,
            "excitation_pwm5_8_std": excitation,
            "rmse_p": dof_rmse[3],
            "rmse_q": dof_rmse[4],
            "rmse_pq_mean": (dof_rmse[3] + dof_rmse[4]) / 2,
        })

    df = pd.DataFrame(rows).sort_values("excitation_pwm5_8_std").reset_index(drop=True)

    print("=" * 100)
    print("VERTICAL-THRUSTER EXCITATION vs ROLL/PITCH RMSE — all runs, sorted by excitation (low to high)")
    print("=" * 100)
    print(f"{'run':<55}{'pwm5-8 std':>14}{'RMSE p':>10}{'RMSE q':>10}{'RMSE(p,q) mean':>16}")
    for _, row in df.iterrows():
        name = Path(row["filepath"]).name
        print(f"{name:<55}{row['excitation_pwm5_8_std']:>14.2f}"
              f"{row['rmse_p']:>10.4f}{row['rmse_q']:>10.4f}{row['rmse_pq_mean']:>16.4f}")

    corr = df["excitation_pwm5_8_std"].corr(df["rmse_pq_mean"])
    print(f"\nCorrelation (excitation vs. roll/pitch RMSE): {corr:+.3f}")
    if corr < -0.3:
        print("NEGATIVE correlation — lower excitation tends to go with HIGHER RMSE. This IS")
        print("consistent with the confound: low-excitation runs may look artificially worse.")
    elif corr > 0.3:
        print("POSITIVE correlation — higher excitation tends to go with HIGHER RMSE. This is")
        print("the OPPOSITE of the confound — worse fit on runs that stress rotation MORE,")
        print("which actually strengthens (not undermines) the missing-physics conclusion.")
    else:
        print("Weak/no correlation — excitation level doesn't clearly explain the RMSE pattern.")

    if args.worst_run:
        matches = df[df["filepath"].str.contains(args.worst_run)]
        if matches.empty:
            print(f"\nNo run matched '{args.worst_run}'")
        else:
            for _, row in matches.iterrows():
                rank = df.index[df["filepath"] == row["filepath"]][0]
                print(f"\nPreviously-flagged worst run ({Path(row['filepath']).name}):")
                print(f"  Excitation rank: {rank + 1}/{len(df)} (1 = lowest excitation)")
                print(f"  Excitation (pwm5-8 std): {row['excitation_pwm5_8_std']:.2f}")
                print(f"  RMSE(p,q) mean: {row['rmse_pq_mean']:.4f}")

    print("\nBest-excited runs (candidates for a cleaner re-run of the force-breakdown analysis):")
    for _, row in df.sort_values("excitation_pwm5_8_std", ascending=False).head(5).iterrows():
        print(f"  {Path(row['filepath']).name}: excitation={row['excitation_pwm5_8_std']:.2f}, "
              f"RMSE(p,q) mean={row['rmse_pq_mean']:.4f}")


if __name__ == "__main__":
    main()