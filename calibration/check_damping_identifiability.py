#!/usr/bin/env python3
"""
Checks whether the linear-vs-quadratic damping correlations found earlier
(Xu/Xuu ~-0.93, Yv/Yvv ~-0.92, etc.) are actually explained by the training
data's velocity distribution — vehicle_spec.yaml's identifiability_note has
CITED narrow speed range as the likely cause, but never verified it against
real data. This does that directly.

The two damping terms being fit are v (linear) and v*|v| (quadratic). If
these two basis functions are themselves highly correlated in the REAL
LOGGED velocity data, no fitting method — however good — could ever
separate them for that axis. This is a property of the data's velocity
distribution, checked independently of the optimizer and independently of
the earlier Jacobian analysis (which measured correlation in the FIT
residuals; this measures correlation in the raw kinematics that produced
them).

Usage:
    python3 check_damping_identifiability.py \\
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \\
        --spec vehicle_spec.yaml \\
        --sampling-rate-hz 50 \\
        --heldout-dates 16-02-2024 19-02-2024
"""
from __future__ import annotations

import argparse

import numpy as np

from data_split import split_by_date, discover_csv_files
from objective import DatasetConfig, load_run

DOF_LABELS = ["u", "v", "w", "p", "q", "r"]

# Which damping-parameter pair each DOF corresponds to, for cross-reference
# against the earlier Jacobian correlation findings.
PARAM_PAIRS = ["Xu/Xuu", "Yv/Yvv", "Zw/Zww", "Kp/Kpp", "Mq/Mqq", "Nr/Nrr"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=None)
    args = parser.parse_args()

    csv_paths, duplicates = discover_csv_files(args.data_dir)
    for dup_path, kept_path in duplicates:
        print(f"Duplicate filename skipped: {dup_path} (kept {kept_path})")

    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files]

    all_nu = np.concatenate([r.real_nu for r in train_runs], axis=0)

    print("=" * 100)
    print("DAMPING IDENTIFIABILITY: is linear (v) vs quadratic (v*|v|) damping separable")
    print("GIVEN THE TRAINING DATA'S ACTUAL VELOCITY DISTRIBUTION, independent of optimizer/model")
    print("=" * 100)
    print(f"{'DOF':<6}{'param pair':<12}{'min':>9}{'max':>9}{'std':>9}{'corr(v,v|v|)':>15}   assessment")

    for i, (label, pair) in enumerate(zip(DOF_LABELS, PARAM_PAIRS)):
        v = all_nu[:, i]
        v_quad = v * np.abs(v)
        corr = np.corrcoef(v, v_quad)[0, 1]
        vmin, vmax, vstd = v.min(), v.max(), v.std()

        if abs(corr) >= 0.9:
            assessment = "SEVERE — v and v|v| nearly collinear in this data"
        elif abs(corr) >= 0.75:
            assessment = "poor — consistent with the damping-pair correlation seen"
        else:
            assessment = "reasonable separation"

        print(f"{label:<6}{pair:<12}{vmin:>9.3f}{vmax:>9.3f}{vstd:>9.3f}{corr:>15.3f}   {assessment}")

    print("\nIf corr(v, v|v|) is high for an axis, linear and quadratic damping CANNOT be")
    print("reliably separated there with this data, no matter how good the optimizer or")
    print("how many restarts — it's a property of the velocity distribution itself (values")
    print("clustering too much within one sign/magnitude band), not a fitting limitation.")
    print("Compare against the earlier Jacobian-measured correlations for the same pairs:")
    print("Xu/Xuu -0.93, Mq/Mqq -0.93, Yv/Yvv -0.92, Zw/Zww -0.88, Nr/Nrr -0.87, Kp/Kpp -0.84.")
    print("If this check's per-axis ranking roughly matches that ranking, the")
    print("identifiability_note's claimed mechanism is CONFIRMED, not just plausible.")


if __name__ == "__main__":
    main()