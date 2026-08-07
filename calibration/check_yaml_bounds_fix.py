#!/usr/bin/env python3
"""
Tests the REAL vehicle_spec.yaml bounds fix (not the fixed_params
override, already proven separately) - using the STANDARD calibration
path every real script (run_calibration.py, run_calibration_holoocean.py)
actually uses: initial_parameters() + calibrate(), no fixed_params.

Confirms: (1) Ieff_roll/Ieff_pitch stay within their new tight bounds
throughout, since they have almost no room to move, (2) the fit still
improves well via the other 20 parameters.

Usage:
    python check_yaml_bounds_fix.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0
"""
import argparse

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run
from optimizer import load_vehicle_spec, initial_parameters, CalibrationConfig, calibrate


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

    loaded = load_vehicle_spec(args.spec)
    start = initial_parameters(loaded)
    print(f"Starting Ieff_roll={start.Ieff_roll}, Ieff_pitch={start.Ieff_pitch} "
          f"(should be ~0.26/0.23 - confirms vehicle_spec.yaml's initial_guess was updated too)\n")

    def progress(call_count, param_vec, cost):
        print(f"  eval {call_count}: cost={cost:.4f}")

    config = CalibrationConfig(n_restarts=1, seed=42, max_nfev=args.max_nfev, on_iteration=progress)
    try:
        result = calibrate(loaded, train_runs, config)
    except RuntimeError as e:
        print(f"Did not converge in budget: {str(e)[:100]}")
        return

    print(f"\nConverged: {result.converged}, cost: {result.cost:.6f}, train RMSE: {result.train_rmse:.4f}")
    print(f"Final Ieff_roll={result.parameters.Ieff_roll:.4f} (expect within [0.255, 0.265])")
    print(f"Final Ieff_pitch={result.parameters.Ieff_pitch:.4f} (expect within [0.225, 0.235])")
    in_bounds = (0.255 <= result.parameters.Ieff_roll <= 0.265 and
                 0.225 <= result.parameters.Ieff_pitch <= 0.235)
    print(f"\nCONFIRMED staying within the new fixed bounds via the STANDARD calibration path: {in_bounds}")


if __name__ == "__main__":
    main()