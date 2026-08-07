#!/usr/bin/env python3
"""
Computes a CUSTOM dof_scales array for the optimizer - NOT the raw
compute_dof_scales() output, because that would help p (genuinely
under-weighted, small natural scale) while making r WORSE (r already has
the LARGEST natural scale of all 6 DOFs, so scale-normalization would
de-emphasize it further, not help it).

Applies a deliberate, documented boost to p and r beyond their natural
scales:
  - p: boosted because it's confirmed under-weighted by raw scale alone
  - r: boosted BEYOND its already-large natural scale, because it kept
    regressing across multiple real HoloOcean segments tonight even
    under the current large-scale-implies-more-weight regime - the
    problem for r is empirically NOT "not enough weight", so this is a
    stronger, deliberate intervention, not just correcting an oversight

Usage:
    python compute_custom_dof_weights.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0
"""
import argparse

import numpy as np

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, compute_dof_scales

DOF_LABELS = ["u", "v", "w", "p", "q", "r"]

# How much to shrink the effective scale (= how much to BOOST weight) for
# each problem DOF, relative to its own natural training-data scale.
# 0.5 = weight it as if its scale were half its real value (2x more weight
# than plain normalization would give it).
P_BOOST_FACTOR = 0.3   # p: strongly boosted - confirmed under-weighted
R_BOOST_FACTOR = 0.15  # r: even more strongly boosted - already large
                        # scale wasn't enough, needs deliberate extra weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files]

    natural_scales = compute_dof_scales(train_runs)
    print("Natural (unmodified) per-DOF scales, from TRAINING data only:")
    for i, label in enumerate(DOF_LABELS):
        print(f"  {label}: {natural_scales[i]:.4f}")

    custom_scales = natural_scales.copy()
    p_idx, r_idx = DOF_LABELS.index("p"), DOF_LABELS.index("r")
    custom_scales[p_idx] *= P_BOOST_FACTOR
    custom_scales[r_idx] *= R_BOOST_FACTOR

    print(f"\nCustom scales (p boosted x{1/P_BOOST_FACTOR:.1f}, r boosted x{1/R_BOOST_FACTOR:.1f} "
          f"beyond natural scale):")
    for i, label in enumerate(DOF_LABELS):
        marker = "  <-- boosted" if i in (p_idx, r_idx) else ""
        print(f"  {label}: {custom_scales[i]:.4f}{marker}")

    print(f"\nRelative weight each DOF gets in the objective (1/scale, normalized so u=1.0):")
    weights = (1.0 / custom_scales) / (1.0 / custom_scales[0])
    for i, label in enumerate(DOF_LABELS):
        print(f"  {label}: {weights[i]:.2f}x")

    np.save("custom_dof_scales.npy", custom_scales)
    print(f"\nSaved to custom_dof_scales.npy — pass this to CalibrationConfig(dof_scales=...)")


if __name__ == "__main__":
    main()