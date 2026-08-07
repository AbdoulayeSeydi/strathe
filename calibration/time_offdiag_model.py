#!/usr/bin/env python3
"""
Directly measures the real per-evaluation cost of the NEW 22-parameter
off-diagonal added-mass model, at FULL scale (all 25 training runs, full
length) — comparing against the confirmed-good baseline from before this
change (~41-47s per full pass, measured multiple times earlier this
session). Prints progress per-run, not just at the end, so this can't
itself become an unexplained silent wait.

Usage:
    python time_offdiag_model.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0
"""
import argparse
import time

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, rmse_for_run
from optimizer import load_vehicle_spec, initial_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files]
    print(f"Loaded {len(train_runs)} training runs, "
          f"{sum(r.n_samples for r in train_runs)} total steps\n")

    loaded = load_vehicle_spec(args.spec)
    params = initial_parameters(loaded)
    vehicle = loaded.vehicle
    print(f"Xq_dot={params.Xq_dot}, Zq_dot={params.Zq_dot} (should be 0.0, matching the "
          f"exact starting point of the run that hung)\n")

    print("Timing ONE evaluation (rmse_for_run per training run), printing progress "
          "per run so this can't itself silently hang...")
    t_total_start = time.monotonic()
    for i, run in enumerate(train_runs):
        t0 = time.monotonic()
        rmse = rmse_for_run(params, run, vehicle)
        elapsed = time.monotonic() - t0
        print(f"  run {i+1}/{len(train_runs)} ({run.n_samples} steps): "
              f"RMSE={rmse:.4f} ({elapsed:.2f}s)")
    total = time.monotonic() - t_total_start

    print(f"\nTotal time for ONE full evaluation (all 25 training runs): {total:.1f}s")
    print(f"Compare against the confirmed-good baseline from BEFORE this change: ~41-47s")
    print(f"If this is similar (~41-47s), the new code is NOT meaningfully slower — "
          f"the earlier 12-hour hang was likely a genuine bug/stall, not raw compute cost.")
    print(f"If this is dramatically slower (many minutes), the Cholesky/linalg.solve "
          f"overhead per-timestep is the real, confirmed cause.")


if __name__ == "__main__":
    main()