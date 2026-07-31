#!/usr/bin/env python3
"""
Diagnoses the NaN crash found when re-running the 10Hz stand-in
calibration with Coriolis active: default (uncalibrated) parameters,
against REAL logged forces (not the synthetic forces every prior Coriolis
stability test used — a real gap, since that's exactly what crashed).

Reproduces the EXACT failing scenario against ONE real training run,
step by step, printing velocity/force magnitude right up to the point of
failure — so we know WHERE it breaks and WHY, not just that it does.

Usage:
    python diagnose_coriolis_crash.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0
"""
import argparse
import numpy as np

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run
from optimizer import load_vehicle_spec, initial_parameters
from dynamics_model import SimulationConfig, compute_acceleration, _mass_and_damping_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--all-runs", action="store_true",
                         help="Check ALL 25 training runs at FULL length (matches what the real "
                              "optimizer actually evaluates), not just 500 steps of one run.")
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=["16-02-2024", "19-02-2024"])
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)

    loaded_spec = load_vehicle_spec(args.spec)
    vehicle = loaded_spec.vehicle
    default_params = initial_parameters(loaded_spec)
    sim_config = SimulationConfig(dt=1.0 / args.sampling_rate_hz)
    n_substeps = sim_config.n_substeps
    sub_dt = sim_config.sub_dt

    files_to_check = split.train_files if args.all_runs else split.train_files[:1]
    max_steps = None if args.all_runs else 500

    for filepath in files_to_check:
        run = load_run(filepath, dataset_config)
        n_check = run.n_samples - 1 if max_steps is None else min(run.n_samples - 1, max_steps)
        v = run.nu0.copy()
        crashed = False
        for k in range(n_check):
            tau = run.forces_torques[k]
            roll, pitch = run.roll_pitch[k]
            for _ in range(n_substeps):
                v_dot, damping_force, g_eta = compute_acceleration(default_params, vehicle, v, tau, roll, pitch)
                v_new = v + v_dot * sub_dt
                if not np.all(np.isfinite(v_new)):
                    print(f"*** {filepath}: NaN/inf at step {k}/{n_check} ***")
                    print(f"    v BEFORE: {v}")
                    print(f"    tau (real logged force): {tau}")
                    print(f"    v_dot computed: {v_dot}")
                    crashed = True
                    break
                v = v_new
            if crashed:
                break
        if not crashed:
            print(f"OK: {filepath} ({n_check} steps checked, max|v|={np.max(np.abs(v)):.3f})")

    if not args.all_runs:
        print("\n(Only checked 1 run at 500 steps — pass --all-runs to check all 25 training "
              "runs at their FULL length, matching what the real optimizer actually evaluates.)")


if __name__ == "__main__":
    main()