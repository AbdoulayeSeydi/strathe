#!/usr/bin/env python3
"""
Directly compares TWO calibrated parameter sets against real held-out
data, using the FAST stand-in backend (seconds, not hours) — specifically
to test whether the incomplete HoloOcean refinement (finding #19,
segment 2, never converged, diminishing returns) moved p/r-relevant
parameters somewhere WORSE than where the stand-in's own CONVERGED
Coriolis fit (finding #18) already had them. If so, the Stage 2.3 mixed
rotational result (finding #20) may be an artifact of stopping HoloOcean
refinement early, not a fundamental limit of the Coriolis/calibration
approach itself.

Usage:
    python compare_two_calibrations.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --params-a ../calibration_output_10hz_coriolis/calibrated_parameters.yaml \
        --label-a "stand-in (converged)" \
        --params-b ../calibration_output_holoocean_coriolis/calibrated_parameters.yaml \
        --label-b "HoloOcean-refined (unconverged)"
"""
import argparse

import numpy as np
import yaml

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, rmse_by_dof
from optimizer import load_vehicle_spec
from dynamics_model import CalibrationParameters

DOF_LABELS = ["u", "v", "w", "p", "q", "r"]


def load_params(path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "parameters" in data:
        return CalibrationParameters(**data["parameters"])
    if "best_parameters_so_far" in data:
        return CalibrationParameters(**data["best_parameters_so_far"])
    raise ValueError(f"{path}: no 'parameters' or 'best_parameters_so_far' key found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--params-a", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--params-b", required=True)
    parser.add_argument("--label-b", default="B")
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    heldout_runs = [load_run(fp, dataset_config) for fp in split.heldout_files]
    print(f"Loaded {len(heldout_runs)} held-out runs (stand-in backend — fast, seconds not hours)\n")

    loaded = load_vehicle_spec(args.spec)
    vehicle = loaded.vehicle

    params_a = load_params(args.params_a)
    params_b = load_params(args.params_b)

    dof_a = np.mean([rmse_by_dof(params_a, run, vehicle) for run in heldout_runs], axis=0)
    dof_b = np.mean([rmse_by_dof(params_b, run, vehicle) for run in heldout_runs], axis=0)

    print(f"{'DOF':<6}{args.label_a:>22}{args.label_b:>25}{'change (A->B)':>18}")
    for i, label in enumerate(DOF_LABELS):
        pct_change = (dof_a[i] - dof_b[i]) / dof_a[i] * 100
        print(f"{label:<6}{dof_a[i]:>22.4f}{dof_b[i]:>25.4f}{pct_change:>+17.1f}%")

    print(f"\nIf p/r show B WORSE than A here (negative %change), the incomplete HoloOcean")
    print(f"refinement itself moved those parameters in the wrong direction — a fixable,")
    print(f"specific issue, not a fundamental limit of Coriolis/calibration in general.")


if __name__ == "__main__":
    main()