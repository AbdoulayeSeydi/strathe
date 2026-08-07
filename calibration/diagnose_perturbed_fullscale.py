#!/usr/bin/env python3
"""
Tests the ONE combination never actually tested before: a PERTURBED
parameter value (not the nominal point) at FULL scale (all 25 training
runs, full length) — exactly what the real optimizer's first Jacobian
step does. Prints per-RUN progress within each perturbed evaluation, so
if one specific run is where things go wrong, we'll see exactly which one
and how long it took, instead of a silent multi-hour wait.

Usage:
    python diagnose_perturbed_fullscale.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --param-name Xq_dot
"""
import argparse
import time

import numpy as np

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, rmse_for_run
from optimizer import load_vehicle_spec, initial_parameters, parameter_bounds
from dynamics_model import CalibrationParameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--param-name", default="Xq_dot",
                         help="Which single parameter to perturb for this test")
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files]
    print(f"Loaded {len(train_runs)} FULL training runs (full length, not a subset)\n")

    loaded = load_vehicle_spec(args.spec)
    params = initial_parameters(loaded)
    vehicle = loaded.vehicle
    lb, ub = parameter_bounds(loaded)
    x0 = params.to_vector()
    param_names = params.param_names
    i = param_names.index(args.param_name)

    step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(x0[i]))
    x_perturbed = x0.copy()
    x_perturbed[i] = np.clip(x0[i] + step, lb[i], ub[i])
    perturbed_params = CalibrationParameters.from_vector(x_perturbed)

    print(f"Perturbing {args.param_name}: {x0[i]} -> {x_perturbed[i]:.6g}\n")
    print("Evaluating against EVERY training run individually, printing progress "
          "per run so we can see exactly where (if anywhere) this goes wrong...")
    t_total = time.monotonic()
    for j, run in enumerate(train_runs):
        t0 = time.monotonic()
        try:
            rmse = rmse_for_run(perturbed_params, run, vehicle)
            elapsed = time.monotonic() - t0
            print(f"  run {j+1}/{len(train_runs)} ({run.n_samples} steps, {run.filepath}): "
                  f"RMSE={rmse:.4f} ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"  run {j+1}/{len(train_runs)} ({run.n_samples} steps, {run.filepath}): "
                  f"RAISED after {elapsed:.2f}s — {e}")

    print(f"\nTotal: {time.monotonic() - t_total:.1f}s for this one perturbed evaluation "
          f"across all {len(train_runs)} runs. Compare against the nominal point's "
          f"confirmed ~96-100s.")


if __name__ == "__main__":
    main()