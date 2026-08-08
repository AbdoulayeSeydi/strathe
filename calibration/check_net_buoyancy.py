#!/usr/bin/env python3
"""
Tests whether net_buoyancy_z is genuinely load-bearing, matching the
exact investigation pattern already used and confirmed for
Ieff_roll/Ieff_pitch: (1) already-measured sensitivity (0.001, tied for
lowest of all 22 parameters — see PROJECT_STATUS.md), (2) this script —
fix it near a physically-sensible trim value and re-fit the other 21
parameters, checking whether fit quality actually suffers.

Currently pinned at its extreme bound (+10N ~= +1kg upward trim force in
the fitted result) — a real question worth answering before deciding
whether that's genuine physics or another non-load-bearing drift, same
as Ieff_roll/Ieff_pitch turned out to be.

"Physically-sensible trim value" here: 0.0N (perfectly neutral trim) —
the vehicle_spec.yaml's OWN stated assumption/prior ("assumed near-
neutral trim", confidence: assumed) before any fitting ever happened.
Testing the file's own original prior, not a new external source.

Usage:
    python check_net_buoyancy.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0
"""
import argparse

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run
from optimizer import load_vehicle_spec, CalibrationConfig, calibrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--max-nfev", type=int, default=10)
    parser.add_argument("--fixed-value", type=float, default=0.0,
                         help="Value to fix net_buoyancy_z at. Default 0.0 = "
                              "vehicle_spec.yaml's own original 'neutral trim' prior.")
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files[:args.n_runs]]
    print(f"Using {len(train_runs)} training run(s) — fixing net_buoyancy_z={args.fixed_value} "
          f"(vehicle_spec.yaml's own original neutral-trim prior), re-fitting the other 21 parameters\n")

    loaded = load_vehicle_spec(args.spec)

    def progress(call_count, param_vec, cost):
        print(f"  eval {call_count}: cost={cost:.4f}")

    config = CalibrationConfig(
        n_restarts=1, seed=42, max_nfev=args.max_nfev,
        fixed_params={"net_buoyancy_z": args.fixed_value},
        on_iteration=progress,
    )
    try:
        result = calibrate(loaded, train_runs, config)
        print(f"\nConverged: {result.converged}, cost: {result.cost:.6f}, "
              f"train RMSE: {result.train_rmse:.4f}")
    except RuntimeError as e:
        print(f"\nDid not converge in budget (fine — trajectory still informative): {str(e)[:150]}")


if __name__ == "__main__":
    main()