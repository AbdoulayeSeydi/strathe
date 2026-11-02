#!/usr/bin/env python3
"""
Checks, using ONLY real logged velocity data (pure kinematics — no
simulation, no model, no optimizer involved), whether the training data
actually excites the specific velocity PRODUCTS Coriolis's rotational
equations depend on. Same methodology as finding #10's independent
damping-identifiability check (corr(v, v|v|) from real data alone),
applied to the new Coriolis terms.

Per dynamics_model.py's _coriolis_centripetal_force derivation (finding
#16), the rotational-DOF Coriolis contributions are:
    roll:  (m3-m2)*v*w + (m6-m5)*Q*R
    pitch: (m1-m3)*u*w + (m4-m6)*P*R
    yaw:   (m2-m1)*u*v + (m5-m4)*P*Q

If these products have near-zero variance in the real data, or their
paired components are so highly correlated that the product barely
varies independently, no amount of further model complexity can identify
the corresponding mass-difference coefficients FROM THIS DATASET — a
genuine data-coverage ceiling, not a missing-physics gap. If the products
ARE well-excited but the fit is still weak, that points the other way:
toward a genuine remaining model limitation worth investigating further.

Usage:
    python check_coriolis_excitation.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --sampling-rate-hz 10.0
"""
from __future__ import annotations

import argparse

import numpy as np

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run

DOF_LABELS = ["u", "v", "w", "P", "Q", "R"]

# (label, index_a, index_b) — the exact velocity-product pairs Coriolis's
# rotational equations depend on, per dynamics_model.py's derivation.
PRODUCT_DEFINITIONS = [
    ("v*w  (-> roll,  coeff (m3-m2))", 1, 2),
    ("Q*R  (-> roll,  coeff (m6-m5))", 4, 5),
    ("u*w  (-> pitch, coeff (m1-m3))", 0, 2),
    ("P*R  (-> pitch, coeff (m4-m6))", 3, 5),
    ("u*v  (-> yaw,   coeff (m2-m1))", 0, 1),
    ("P*Q  (-> yaw,   coeff (m5-m4))", 3, 4),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=None,
                        help="recording dates to hold out (DD-MM-YYYY); default: the same split calibration uses")
    args = parser.parse_args()

    csv_paths, duplicates = discover_csv_files(args.data_dir)
    for dup_path, kept_path in duplicates:
        print(f"Duplicate filename skipped: {dup_path} (kept {kept_path})")

    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files]

    # Concatenate REAL LOGGED velocity across all training runs — pure
    # data, no model, no simulation.
    all_nu = np.concatenate([run.real_nu for run in train_runs], axis=0)
    print(f"Loaded {len(train_runs)} training runs, {all_nu.shape[0]} total real samples\n")

    print("=" * 70)
    print("1. INDIVIDUAL COMPONENT EXCITATION (reference baseline)")
    print("=" * 70)
    print(f"{'DOF':<6}{'std':>10}{'min':>10}{'max':>10}")
    component_stds = {}
    unmeasured = [label for i, label in enumerate(DOF_LABELS) if not np.isfinite(all_nu[:, i]).any()]
    for i, label in enumerate(DOF_LABELS):
        col = all_nu[:, i]
        col = col[np.isfinite(col)]   # a DVL with gaps reads only where it locked (#96)
        if not len(col):
            component_stds[label] = float("nan")
            print(f"{label:<6}{'not measured (no DVL)':>30}")
            continue
        component_stds[label] = float(np.std(col))
        print(f"{label:<6}{np.std(col):>10.4f}{np.min(col):>10.4f}{np.max(col):>10.4f}")

    print()
    print("=" * 70)
    print("2. CORIOLIS VELOCITY-PRODUCT EXCITATION (the actual question)")
    print("=" * 70)
    print(f"{'Product':<32}{'std':>10}{'min':>10}{'max':>10}{'corr(a,b)':>12}")
    product_stds = {}
    for label, i, j in PRODUCT_DEFINITIONS:
        a, b = all_nu[:, i], all_nu[:, j]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            product_stds[label] = float("nan")
            print(f"{label:<32}{'not measured (no DVL)':>30}")
            continue
        a, b = a[ok], b[ok]
        product = a * b
        product_std = float(np.std(product))
        product_stds[label] = product_std
        corr_ab = float(np.corrcoef(a, b)[0, 1])
        print(f"{label:<32}{product_std:>10.5f}{np.min(product):>10.5f}{np.max(product):>10.5f}{corr_ab:>12.3f}")

    if unmeasured:
        print(f"\n{', '.join(unmeasured)}: no DVL in these logs, so their excitation can't be measured from velocity. "
              f"The sensor suite fits them to position fixes and depth instead, and says which axes those sensors "
              f"see (SENSORS.md).")
    print()
    print("How to read this:")
    print("  - std column is the primary answer: the ACTUAL excitation of each Coriolis-relevant")
    print("    product in real data, in the product's own natural units. Compare the 6 rows against")
    print("    EACH OTHER — orders-of-magnitude differences reveal which rotational axes have real")
    print("    data support for their new coupling term, and which are starved of it. A product")
    print("    near-zero across the whole dataset means that coupling term genuinely can't be")
    print("    constrained from this data, regardless of how sophisticated the model is.")
    print("  - corr(a,b) is supplementary context (do the two components tend to move together or")
    print("    independently) — NOT a direct measure of whether the product itself is well-excited.")
    print("    A worked check confirms strong correlation (either sign) tends to slightly INCREASE")
    print("    a product's variance relative to what independence would predict, not suppress it —")
    print("    so don't read a high |corr| here as a red flag on its own; the std column is what")
    print("    actually answers the identifiability question.")


if __name__ == "__main__":
    main()