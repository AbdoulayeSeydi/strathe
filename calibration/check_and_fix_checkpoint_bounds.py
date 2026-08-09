#!/usr/bin/env python3
"""
Checks EVERY parameter in a checkpoint against the CURRENT
vehicle_spec.yaml bounds (not just the ones we expected to have
changed) - a general fix for the real mistake found tonight: only
Ieff_roll/Ieff_pitch were manually corrected, but net_buoyancy_z's
bounds were ALSO tightened earlier the same night and its old checkpoint
value (~9.99) was missed, causing a real crash.

For any out-of-bounds parameter, clamps it to the NEAREST bound edge by
default - a safe, general fallback. Known cases we already have real,
confirmed-good fixed values for (Ieff_roll, Ieff_pitch, net_buoyancy_z)
use those specific values instead of just clamping.

Usage:
    python check_and_fix_checkpoint_bounds.py \
        --checkpoint ../calibration_output_holoocean_coriolis/checkpoint.yaml \
        --spec vehicle_spec.yaml \
        --output ../calibration_output_holoocean_coriolis/checkpoint_bounds_fixed.yaml
"""
import argparse

import yaml

from optimizer import load_vehicle_spec, parameter_bounds, initial_parameters

# Known, confirmed-good values from tonight's real investigations -
# used instead of a bare clamp for these specific parameters.
KNOWN_GOOD_VALUES = {
    "Ieff_roll": 0.26,
    "Ieff_pitch": 0.23,
    "net_buoyancy_z": 0.0,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    loaded = load_vehicle_spec(args.spec)
    lb, ub = parameter_bounds(loaded)
    param_names = initial_parameters(loaded).param_names

    with open(args.checkpoint, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    print(f"{'Parameter':<18}{'Value':>14}{'Lower':>12}{'Upper':>12}{'Status':>20}")
    any_fixed = False
    for i, name in enumerate(param_names):
        val = data["best_parameters_so_far"].get(name)
        if val is None:
            continue
        if lb[i] <= val <= ub[i]:
            print(f"{name:<18}{val:>14.4f}{lb[i]:>12.4f}{ub[i]:>12.4f}{'OK':>20}")
            continue

        any_fixed = True
        if name in KNOWN_GOOD_VALUES:
            new_val = KNOWN_GOOD_VALUES[name]
            reason = "known-good value"
        else:
            new_val = min(max(val, lb[i]), ub[i])
            reason = "clamped to nearest bound"
        print(f"{name:<18}{val:>14.4f}{lb[i]:>12.4f}{ub[i]:>12.4f}"
              f"{'OUT OF BOUNDS -> ' + f'{new_val:.4f} (' + reason + ')':>20}")

        for key in ("current_parameters", "best_parameters_so_far"):
            if name in data[key]:
                data[key][name] = new_val

    if not any_fixed:
        print("\nNo out-of-bounds parameters found - checkpoint is already valid "
              "against the current vehicle_spec.yaml.")
    else:
        print(f"\nFixed the out-of-bounds parameter(s) listed above.")

    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Saved to: {args.output} (original checkpoint untouched)")


if __name__ == "__main__":
    main()