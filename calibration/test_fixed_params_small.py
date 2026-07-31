#!/usr/bin/env python3
"""
A small, fast, fully-visible test of the fixed_params mechanism before
trusting it with a long real run — prints progress after EVERY single
evaluation (not just at the end), and uses a tiny max_nfev so we get a
real answer in minutes, not hours: is this slow, or genuinely stuck?

Usage:
    python test_fixed_params_small.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0
"""
import argparse
import time

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run
from optimizer import load_vehicle_spec, CalibrationConfig, calibrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=["16-02-2024", "19-02-2024"])
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files]
    loaded = load_vehicle_spec(args.spec)

    def progress(call_count, param_vec, cost):
        print(f"  eval {call_count}: cost={cost:.4f} ({time.monotonic() - t0:.1f}s elapsed)")

    print("Testing fixed_params mechanism with a SMALL budget (max_nfev=3), full visibility...")
    t0 = time.monotonic()
    config = CalibrationConfig(n_restarts=1, seed=42, max_nfev=3,
                                fixed_params={"Zw_dot": 14.6}, on_iteration=progress)
    try:
        result = calibrate(loaded, train_runs, config)
        print(f"Converged: {result.converged}, cost: {result.cost:.6f}")
    except RuntimeError as e:
        print(f"Did not converge in 3 evals (expected/fine): {str(e)[:100]}")
    print(f"\nTotal time for up to 3 real evaluations: {time.monotonic() - t0:.1f}s")
    print("If each 'eval' line above took roughly 40-50s, the mechanism works and is just "
          "slow, as expected — the earlier run's problem was likely just missing visibility. "
          "If it took MUCH longer per eval, or never printed anything, something is genuinely wrong.")


if __name__ == "__main__":
    main()