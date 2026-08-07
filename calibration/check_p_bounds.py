#!/usr/bin/env python3
"""
Fast, cheap check (no simulation, no optimizer — just comparing values to
bounds): is roll-relevant damping (Kp, Kpp) pinned at its bound in the
latest calibration result? If so, that's a real, well-motivated,
cheap-to-fix explanation for why p (roll rate) remains stuck, matching
exactly the same pattern that already fixed a real rotational problem
earlier tonight (the zero-damping bound issue).

Usage:
    python check_p_bounds.py \
        --spec vehicle_spec.yaml \
        --calibrated-params ../calibration_output_holoocean_coriolis/calibrated_parameters.yaml
"""
import argparse
import yaml

from optimizer import load_vehicle_spec, parameter_bounds, initial_parameters
from dynamics_model import CalibrationParameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--calibrated-params", required=True)
    args = parser.parse_args()

    loaded = load_vehicle_spec(args.spec)
    lb, ub = parameter_bounds(loaded)
    param_names = initial_parameters(loaded).param_names

    with open(args.calibrated_params, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    key = "parameters" if "parameters" in data else "best_parameters_so_far"
    calibrated = CalibrationParameters(**data[key])
    calibrated_vec = calibrated.to_vector()

    print(f"{'Parameter':<18}{'Value':>14}{'Lower bound':>14}{'Upper bound':>14}{'% from nearest bound':>22}")
    for i, name in enumerate(param_names):
        val = calibrated_vec[i]
        width = ub[i] - lb[i]
        dist_to_lower = (val - lb[i]) / width * 100 if width > 0 else float("nan")
        dist_to_upper = (ub[i] - val) / width * 100 if width > 0 else float("nan")
        nearest = min(dist_to_lower, dist_to_upper)
        flag = "  <-- PINNED (within 5%)" if nearest < 5 else ""
        print(f"{name:<18}{val:>14.4f}{lb[i]:>14.4f}{ub[i]:>14.4f}{nearest:>21.1f}%{flag}")


if __name__ == "__main__":
    main()