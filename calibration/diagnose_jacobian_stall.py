#!/usr/bin/env python3
"""
Isolates which (if any) of the 22 parameters' PERTURBED evaluations —
exactly what scipy's Jacobian estimation actually does, one parameter at
a time — is slow or hanging. The nominal starting point was already
confirmed fast (~96s, time_offdiag_model.py) — but that never tests a
PERTURBED point, which is what the real Jacobian computation needs 22 of.

Uses a SMALL subset (3 runs) so each individual perturbed evaluation is
fast, and prints progress after every single one — no silent waits.

Usage:
    python diagnose_jacobian_stall.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0
"""
import argparse
import time

import numpy as np

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, joint_residuals
from optimizer import load_vehicle_spec, initial_parameters, parameter_bounds


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
    print(f"Using {len(train_runs)} training run(s) for this isolation test\n")

    loaded = load_vehicle_spec(args.spec)
    params = initial_parameters(loaded)
    vehicle = loaded.vehicle
    lb, ub = parameter_bounds(loaded)
    x0 = params.to_vector()
    param_names = params.param_names

    print("=== Baseline: nominal starting point (should be fast) ===")
    t0 = time.monotonic()
    joint_residuals(x0, train_runs, vehicle)
    print(f"Nominal point: {time.monotonic() - t0:.2f}s\n")

    print("=== Perturbing EACH parameter individually, exactly like scipy's own")
    print("    finite-difference Jacobian estimation does (2-point, relative step) ===")
    for i, name in enumerate(param_names):
        x_perturbed = x0.copy()
        # Match scipy's default relative step for '2-point' finite differences
        step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(x0[i]))
        x_perturbed[i] = np.clip(x0[i] + step, lb[i], ub[i])

        t0 = time.monotonic()
        try:
            joint_residuals(x_perturbed, train_runs, vehicle)
            elapsed = time.monotonic() - t0
            print(f"  [{i+1}/{len(param_names)}] {name:<18} perturbed to {x_perturbed[i]:.6g}: "
                  f"{elapsed:.2f}s")
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"  [{i+1}/{len(param_names)}] {name:<18} perturbed to {x_perturbed[i]:.6g}: "
                  f"RAISED after {elapsed:.2f}s — {e}")

    print("\nIf every row above completed in a time comparable to the baseline, no single")
    print("parameter's perturbation is the problem — the stall must be somewhere else")
    print("(scipy's own internals, or something not exercised by this direct test).")
    print("If ANY row took dramatically longer or never printed, THAT parameter is the cause.")


if __name__ == "__main__":
    main()