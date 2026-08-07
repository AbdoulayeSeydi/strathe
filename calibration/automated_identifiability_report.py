#!/usr/bin/env python3
"""
Automated identifiability reporting (item #17) - wraps
diagnose_calibration.py's identifiability_analysis() and
report_correlated_pairs() (both imported directly, NOT reimplemented)
into one command that produces a structured, machine-readable report -
not just printed text a human has to read through the full
diagnose_calibration.py CLI (which also does profiling, per-DOF
worst-run analysis, and plotting - heavier than just this one question).

HONEST COST: this runs the real Jacobian evaluation - confirmed ~15-17
minutes on the full real dataset (PROJECT_STATUS.md finding #18). Not
instant. Use --n-runs to test on a small subset first.

Usage:
    python automated_identifiability_report.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --calibrated-params ../calibration_output_holoocean_coriolis/calibrated_parameters.yaml
"""
import argparse
import json

import yaml

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run
from optimizer import load_vehicle_spec
from dynamics_model import CalibrationParameters
from diagnose_calibration import identifiability_analysis, report_correlated_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--calibrated-params", required=True)
    parser.add_argument("--correlation-threshold", type=float, default=0.8)
    parser.add_argument("--n-runs", type=int, default=None,
                         help="Limit to this many training runs for a faster (but less "
                              "complete) check. Default: use all training runs.")
    parser.add_argument("--output", default="identifiability_report.json",
                         help="Where to write the structured JSON report.")
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_files = split.train_files[:args.n_runs] if args.n_runs else split.train_files
    train_runs = [load_run(fp, dataset_config) for fp in train_files]
    print(f"Using {len(train_runs)} training run(s) "
          f"({'LIMITED via --n-runs' if args.n_runs else 'full training set'})\n")

    loaded = load_vehicle_spec(args.spec)
    vehicle = loaded.vehicle

    with open(args.calibrated_params, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    key = "parameters" if "parameters" in data else "best_parameters_so_far"
    calibrated = CalibrationParameters(**data[key])
    param_vec = calibrated.to_vector()
    param_names = calibrated.param_names

    print("Running the real Jacobian-based identifiability analysis - this reuses "
          "diagnose_calibration.py's own function, not a reimplementation. Real cost: "
          "confirmed ~15-17 minutes on the full dataset (fewer runs = faster).\n")
    analysis = identifiability_analysis(param_vec, train_runs, vehicle)

    correlated_pairs = []
    if analysis["correlation"] is not None:
        pairs = report_correlated_pairs(
            analysis["correlation"], param_names, threshold=args.correlation_threshold
        )
        correlated_pairs = [
            {"param_a": a, "param_b": b, "correlation": float(c)} for a, b, c in pairs
        ]

    report = {
        "condition_number": analysis["condition_number"],
        "correlated_pairs": correlated_pairs,
        "near_null_directions": analysis["near_null_directions"],
        "n_train_runs_used": len(train_runs),
        "correlation_threshold": args.correlation_threshold,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 70}\nAUTOMATED IDENTIFIABILITY REPORT SUMMARY\n{'=' * 70}")
    print(f"Condition number: {report['condition_number']:.3e}")
    print(f"Correlated pairs (>= {args.correlation_threshold}): {len(correlated_pairs)}")
    for p in correlated_pairs:
        print(f"  {p['param_a']:<18} <-> {p['param_b']:<18} corr={p['correlation']:+.3f}")
    print(f"\nStructured report written to: {args.output}")


if __name__ == "__main__":
    main()