#!/usr/bin/env python3
"""
Cheap, fast (seconds) directional sanity check before committing to
another ~11 hour HoloOcean run: takes the current checkpoint's
parameters, manually pushes Ieff_roll and Kp further into the NEWLY
widened range (the direction they were straining toward when pinned),
and checks whether p's RMSE actually improves — using the FAST stand-in
backend on real held-out data.

HONEST LIMITATION: the stand-in and real HoloOcean can disagree on which
parameters are actually better (see PROJECT_STATUS.md finding #24) — this
can only rule out an OBVIOUSLY wrong direction, not confirm the real
HoloOcean result will improve. Still worth checking before another long
commitment.

Usage:
    python check_widened_bounds_direction.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --calibrated-params ../calibration_output_holoocean_coriolis/calibrated_parameters.yaml
"""
import argparse

import numpy as np
import yaml

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, rmse_by_dof
from optimizer import load_vehicle_spec, parameter_bounds
from dynamics_model import CalibrationParameters

DOF_LABELS = ["u", "v", "w", "p", "q", "r"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--calibrated-params", required=True)
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    heldout_runs = [load_run(fp, dataset_config) for fp in split.heldout_files]

    loaded = load_vehicle_spec(args.spec)
    vehicle = loaded.vehicle
    lb, ub = parameter_bounds(loaded)
    param_names = None

    with open(args.calibrated_params, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    key = "parameters" if "parameters" in data else "best_parameters_so_far"
    current = CalibrationParameters(**data[key])
    param_names = current.param_names

    def dof_rmse(params):
        return np.mean([rmse_by_dof(params, run, vehicle) for run in heldout_runs], axis=0)

    print("=== Baseline: current checkpoint parameters (fast stand-in, real held-out data) ===")
    baseline_dof = dof_rmse(current)
    for i, label in enumerate(DOF_LABELS):
        print(f"  {label}: {baseline_dof[i]:.4f}")

    # Push Ieff_roll and Kp halfway into the NEWLY available room beyond
    # their old bound (5.0 and -5.0 respectively) — a concrete, moderate
    # step in the direction they were straining toward, not the extreme edge.
    i_roll = param_names.index("Ieff_roll")
    i_kp = param_names.index("Kp")
    pushed_vec = current.to_vector().copy()
    pushed_vec[i_roll] = 7.5   # halfway between the old bound (5.0) and new bound (10.0)
    pushed_vec[i_kp] = -7.5    # halfway between the old bound (-5.0) and new bound (-10.0)
    pushed = CalibrationParameters.from_vector(pushed_vec)

    print(f"\n=== Pushed: Ieff_roll {current.Ieff_roll:.3f}->7.5, Kp {current.Kp:.3f}->-7.5 "
          f"(halfway into the newly available room) ===")
    pushed_dof = dof_rmse(pushed)
    for i, label in enumerate(DOF_LABELS):
        change = (baseline_dof[i] - pushed_dof[i]) / baseline_dof[i] * 100
        print(f"  {label}: {pushed_dof[i]:.4f}  ({change:+.1f}% vs baseline)")

    print(f"\nIf p improves here (even on the stand-in), that's a real, if not conclusive, green light.")
    print(f"If p gets WORSE here, that's a genuine red flag worth reconsidering before the long run.")
    print(f"Remember: stand-in and real HoloOcean CAN disagree (finding #24) — this rules out an")
    print(f"obviously wrong direction, it doesn't confirm the real HoloOcean result.")


if __name__ == "__main__":
    main()