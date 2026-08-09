#!/usr/bin/env python3
"""
Corrects Ieff_roll/Ieff_pitch in a checkpoint file to the new,
physically-real fixed values, keeping all other 20 already-converged
parameters exactly as they were - a clean, precise test of "does fixing
these two specifically change r," without disturbing anything else.

Saves as a NEW file (never overwrites the original checkpoint - tonight's
hard-learned lesson about not risking real, valuable progress).

Usage:
    python fix_checkpoint_inertia.py \
        --checkpoint ../calibration_output_holoocean_coriolis/checkpoint.yaml \
        --output ../calibration_output_holoocean_coriolis/checkpoint_ieff_fixed.yaml
"""
import argparse

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ieff-roll", type=float, default=0.26)
    parser.add_argument("--ieff-pitch", type=float, default=0.23)
    args = parser.parse_args()

    with open(args.checkpoint, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    print(f"Original: Ieff_roll={data['best_parameters_so_far']['Ieff_roll']}, "
          f"Ieff_pitch={data['best_parameters_so_far']['Ieff_pitch']}")

    for key in ("current_parameters", "best_parameters_so_far"):
        data[key]["Ieff_roll"] = args.ieff_roll
        data[key]["Ieff_pitch"] = args.ieff_pitch

    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Corrected: Ieff_roll={args.ieff_roll}, Ieff_pitch={args.ieff_pitch}")
    print(f"All other 20 parameters left EXACTLY as they were.")
    print(f"Saved to: {args.output} (original checkpoint untouched)")


if __name__ == "__main__":
    main()