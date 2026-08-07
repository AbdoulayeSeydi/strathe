#!/usr/bin/env python3
"""
General parameter sensitivity sweep (item #11 from the original
diagnostic checklist - never built until now): perturbs EACH of the 22
calibration parameters by a small percentage (default 5%) individually,
measures the resulting change in RMSE, on a SMALL, FAST subset of real
data (default 3 runs) - deliberately avoiding the full-dataset
performance mystery currently under separate live investigation.

Directly relevant to tonight's Ieff_roll/Ieff_pitch finding: if RMSE is
highly sensitive to these two specifically, that's consistent with the
optimizer using them as compensation for something else (a parameter
doing a lot of "work" doesn't necessarily mean it's PHYSICALLY correct,
just that the objective is sensitive to it).

Usage:
    python check_parameter_sensitivity.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --calibrated-params ../calibration_output_holoocean_coriolis/calibrated_parameters.yaml
"""
import argparse

import numpy as np
import yaml

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, mean_run_rmse
from optimizer import load_vehicle_spec, parameter_bounds
from dynamics_model import CalibrationParameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--calibrated-params", required=True)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--perturb-frac", type=float, default=0.05)
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    runs = [load_run(fp, dataset_config) for fp in split.train_files[:args.n_runs]]
    print(f"Using {len(runs)} training run(s) for this fast sensitivity sweep\n")

    loaded = load_vehicle_spec(args.spec)
    vehicle = loaded.vehicle
    lb, ub = parameter_bounds(loaded)

    with open(args.calibrated_params, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    key = "parameters" if "parameters" in data else "best_parameters_so_far"
    base_params = CalibrationParameters(**data[key])
    param_names = base_params.param_names
    x0 = base_params.to_vector()

    baseline_rmse = mean_run_rmse(base_params, runs, vehicle)
    print(f"Baseline RMSE at current calibrated point: {baseline_rmse:.4f}\n")

    print(f"{'Parameter':<18}{'Base value':>14}{'Perturbed +5%':>16}{'RMSE change':>16}{'Sensitivity':>14}")
    results = []
    for i, name in enumerate(param_names):
        x_perturbed = x0.copy()
        step = x0[i] * args.perturb_frac
        x_perturbed[i] = np.clip(x0[i] + step, lb[i], ub[i])
        perturbed_params = CalibrationParameters.from_vector(x_perturbed)
        perturbed_rmse = mean_run_rmse(perturbed_params, runs, vehicle)
        rmse_change_pct = (perturbed_rmse - baseline_rmse) / baseline_rmse * 100
        # Sensitivity: how much RMSE moves per 1% parameter change - lets
        # us compare across parameters of very different natural scales
        sensitivity = abs(rmse_change_pct) / (args.perturb_frac * 100) if step != 0 else 0.0
        results.append((name, x0[i], x_perturbed[i], rmse_change_pct, sensitivity))
        print(f"{name:<18}{x0[i]:>14.4f}{x_perturbed[i]:>16.4f}{rmse_change_pct:>+15.2f}%{sensitivity:>14.3f}")

    print("\nRanked by sensitivity (highest = RMSE most affected by this parameter):")
    for name, base_val, pert_val, rmse_change, sens in sorted(results, key=lambda r: -r[4])[:8]:
        print(f"  {name:<18} sensitivity={sens:.3f}  (base={base_val:.4f})")

    print("\nHow to read this: HIGH sensitivity means the objective cares a lot about this")
    print("parameter's exact value - it's doing real 'work' in the fit. This does NOT by")
    print("itself confirm the value is physically correct (see PROJECT_STATUS.md's")
    print("Ieff_roll/Ieff_pitch finding - independently-sourced real values are ~0.23-0.37,")
    print("far below what recent fits have converged to, suggesting compensation for")
    print("missing physics rather than genuine inertia).")


if __name__ == "__main__":
    main()