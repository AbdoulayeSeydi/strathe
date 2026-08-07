#!/usr/bin/env python3
"""
Tests whether scipy's OWN optimizer machinery (not our residual function —
already confirmed fast and consistent per-parameter, diagnose_jacobian_stall.py)
stalls on this specific 22-parameter problem. Uses a small, fast 3-run
subset (each evaluation ~9.5s, confirmed) so if scipy itself needs many
minutes/hours to get past nfev=1 even here, that directly confirms the
issue is internal to the optimizer, not the per-evaluation cost.

Usage:
    python diagnose_scipy_stall.py \
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
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files[:args.n_runs]]
    print(f"Using {len(train_runs)} training run(s) — each real evaluation confirmed ~9.5s\n")

    loaded = load_vehicle_spec(args.spec)

    def progress(call_count, param_vec, cost):
        print(f"  real eval {call_count}: cost={cost:.4f} ({time.monotonic() - t0:.1f}s elapsed)")

    print("Running the REAL scipy optimizer (verbose=2, full visibility) on this small, "
          "fast subset — if scipy's own internals are the problem, we'll see it stall "
          "here too, at a scale where each evaluation is fast and confirmed consistent.")
    t0 = time.monotonic()
    config = CalibrationConfig(n_restarts=1, seed=42, max_nfev=30, verbose=2, on_iteration=progress)
    try:
        result = calibrate(loaded, train_runs, config)
        print(f"\nConverged: {result.converged}, cost: {result.cost:.6f}")
    except RuntimeError as e:
        print(f"\nDid not converge in budget (fine): {str(e)[:150]}")
    print(f"\nTotal wall-clock time: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()