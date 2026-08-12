#!/usr/bin/env python3
"""
Physical/units plausibility checks (item #2 follow-up) - deliberately
split into two confidence tiers, NOT one blended threshold:

FAIL (high confidence, NOT tied to this dataset's own observed data):
  - Angular rate values exceeding an absolute, physically-motivated
    ceiling (15 rad/s - comfortably above this project's own real
    confirmed maximum, ~4 rad/s, so it will never false-positive
    against genuine data, while reliably catching a degrees/s-vs-
    radians/s mixup for anything but extremely slow real motion).
    This is a well-known, common real mistake, detected directly via
    an absolute physical bound - NOT inferred from a statistical
    threshold, which would conflate "impossible" with "unusually
    energetic."

WARNING (medium confidence, genuinely just an anomaly flag):
  - Any DOF (u,v,w,p,q,r) with values beyond 8x this project's own
    real, observed training-data standard deviation. This is NOT a
    rejection criterion - a future, legitimately more aggressive real
    dataset could exceed this without being wrong. It exists to flag
    "worth a human look," nothing stronger.

Deliberately deferred, not implemented as FAIL yet: linear-velocity
limits tied to the CONFIGURED VEHICLE's actual capability. No
authoritative limit currently exists in vehicle_spec.yaml - per the
explicit design decision (see PROJECT_STATUS.md), this stays
WARNING-only (statistical) until real, vehicle-specific limits are
added there, at which point this should be upgraded to a real FAIL
check using those limits directly, not left as a permanent TODO.

Usage:
    python check_physical_plausibility.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --sampling-rate-hz 10.0
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from data_split import discover_csv_files
from objective import DatasetConfig, load_run

DOF_LABELS = ["u", "v", "w", "p", "q", "r"]
ANGULAR_DOF_INDICES = {3, 4, 5}  # p, q, r

# Absolute, physically-motivated ceiling for angular rates - NOT derived
# from this dataset's own statistics. See module docstring for reasoning.
ANGULAR_RATE_FAIL_CEILING_RAD_S = 15.0

# Real, confirmed training-data standard deviations from this project's
# own diagnostic runs (PROJECT_STATUS.md) - used ONLY for the WARNING-
# tier anomaly check, never as a rejection criterion.
REFERENCE_STD = {"u": 0.376, "v": 0.2223, "w": 0.166, "p": 0.1246, "q": 0.2012, "r": 0.5625}
WARNING_MULTIPLE = 8.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    if not csv_paths:
        print(f"ERROR: no CSV files found under {args.data_dir}")
        sys.exit(1)

    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    print(f"Checking physical plausibility of {len(csv_paths)} file(s)...\n")

    any_failed = False
    any_warned = False
    for filepath in csv_paths:
        filename = Path(filepath).name
        try:
            run = load_run(filepath, dataset_config)
        except Exception as e:
            print(f"  ERROR: {filename}: could not be loaded at all ({type(e).__name__}: {e})")
            any_failed = True
            continue

        for i, label in enumerate(DOF_LABELS):
            values = run.real_nu[:, i]

            if i in ANGULAR_DOF_INDICES:
                over_ceiling = np.sum(np.abs(values) > ANGULAR_RATE_FAIL_CEILING_RAD_S)
                if over_ceiling > 0:
                    any_failed = True
                    print(f"  FAIL: {filename}: {over_ceiling} sample(s) of '{label}' exceed "
                          f"{ANGULAR_RATE_FAIL_CEILING_RAD_S} rad/s (max observed: {np.abs(values).max():.2f}). "
                          f"This commonly indicates a units mismatch (degrees/s supplied where "
                          f"radians/s is expected). Verify the units of the input data.")

            warning_ceiling = WARNING_MULTIPLE * REFERENCE_STD[label]
            over_warning = np.sum(np.abs(values) > warning_ceiling)
            if over_warning > 0:
                any_warned = True
                print(f"  WARNING: {filename}: {over_warning} sample(s) of '{label}' exceed "
                      f"{WARNING_MULTIPLE}x this project's previously-observed range "
                      f"({warning_ceiling:.2f}). This is an anomaly flag, not a rejection - "
                      f"could be genuine, unusually energetic real motion. Worth a look, not "
                      f"necessarily a problem.")

    print()
    if any_failed:
        print("FAIL: one or more files have physically implausible values. Review the "
              "messages above before proceeding.")
        sys.exit(1)
    elif any_warned:
        print("PASS WITH WARNINGS: no definite errors found, but some values are unusually "
              "large relative to this project's prior real data. Review above; not blocking.")
        sys.exit(0)
    else:
        print(f"PASS: all {len(csv_paths)} file(s) are within physically plausible ranges, "
              f"no anomalies flagged.")
        sys.exit(0)


if __name__ == "__main__":
    main()