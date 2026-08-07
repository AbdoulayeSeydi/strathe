#!/usr/bin/env python3
"""
Times a full evaluation using the EXACT parameters currently sitting in
a checkpoint file (a point the real optimizer has already moved to,
genuinely different from the nominal starting point) — something never
directly tested before tonight. Every previous test used the nominal
point or a tiny perturbation from it; this tests the actual region of
parameter space the optimizer is really exploring.

Usage:
    python diagnose_checkpoint_params.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --checkpoint-path ../calibration_output_10hz_offdiag_test4/checkpoint.yaml
"""
import argparse
import time

import yaml

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, rmse_for_run
from optimizer import load_vehicle_spec
from dynamics_model import CalibrationParameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--checkpoint-path", required=True)
    args = parser.parse_args()

    with open(args.checkpoint_path, encoding="utf-8") as f:
        checkpoint = yaml.safe_load(f)
    params = CalibrationParameters(**checkpoint["current_parameters"])
    print(f"Loaded the EXACT current_parameters from the checkpoint "
          f"(call_count={checkpoint['call_count']}, cost={checkpoint['current_cost']})\n")

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files]

    loaded = load_vehicle_spec(args.spec)
    vehicle = loaded.vehicle

    print("Evaluating against every training run individually, printing progress "
          "per run so we see exactly where (if anywhere) this goes wrong...")
    t_total = time.monotonic()
    for i, run in enumerate(train_runs):
        t0 = time.monotonic()
        try:
            rmse = rmse_for_run(params, run, vehicle)
            elapsed = time.monotonic() - t0
            print(f"  run {i+1}/{len(train_runs)} ({run.n_samples} steps): "
                  f"RMSE={rmse:.4f} ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"  run {i+1}/{len(train_runs)} ({run.n_samples} steps): "
                  f"RAISED after {elapsed:.2f}s — {e}")

    print(f"\nTotal: {time.monotonic() - t_total:.1f}s for this ACTUAL checkpoint point, "
          f"all {len(train_runs)} runs. Compare against the nominal point's confirmed ~96-100s.")


if __name__ == "__main__":
    main()