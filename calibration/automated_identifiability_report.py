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

Usage (recommended: a finished calibration, analysed with its own objective settings):
    python automated_identifiability_report.py --result ../calibration_output_x \
        --data-dir <logs> --spec vehicle_spec.yaml --workers 8

Older usage (default objective, whole-run replay, no torque shift or current):
    python automated_identifiability_report.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --calibrated-params ../calibration_output_holoocean_coriolis/calibrated_parameters.yaml
"""
import argparse
import json
import os
import time

import numpy as np
import yaml

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run
from optimizer import load_vehicle_spec
from dynamics_model import CalibrationParameters
from diagnose_calibration import identifiability_analysis, report_correlated_pairs


# a parameter combination the data constrains this many times less than the best-constrained one is
# reported as "barely pinned down" (singular values of the bounds-width-scaled Jacobian)
WEAK_COMBINATION_RATIO = 1e-3


def identifiability_for_result(result_dir: str, data_dir: str, spec_path: str, workers: int = 1,
                               threshold: float = 0.8, output: str = None) -> dict:
    """Identifiability of a finished calibration, with the SAME objective it was fitted with
    (metrics.json's objective block: sampling rate, torque reference point, water current, axis
    scaling, excluded axes, prediction windows, held parameters) on the same training dates.
    Jacobian of the residuals over the free parameters at the calibrated point, evaluated in
    parallel. Writes <result_dir>/identifiability_report.json by default."""
    from scipy.optimize._numdiff import approx_derivative
    from objective import ParallelJointResiduals, compute_zero_baseline_scales, joint_residuals
    from optimizer import parameter_bounds
    from run_calibration import excluded_axis_setup
    with open(os.path.join(result_dir, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    obj = metrics.get("objective") or {}
    with open(os.path.join(result_dir, "calibrated_parameters.yaml"), encoding="utf-8") as f:
        params = CalibrationParameters(**yaml.safe_load(f)["parameters"])
    rate = float(obj.get("sampling_rate_hz") or 10.0)
    offset = tuple(obj.get("torque_reference_offset_m") or (0.0, 0.0, 0.0))
    current = obj.get("current_world_mps")
    by_file = obj.get("current_by_file")
    dc = DatasetConfig(sampling_rate_hz=rate, torque_reference_offset_m=offset,
                       current_world_mps=None if not current or not any(current) else tuple(current),
                       current_by_file=None if not by_file else tuple((f, tuple(c)) for f, c in by_file))
    csv_paths, _ = discover_csv_files(data_dir)
    split = split_by_date(csv_paths, heldout_dates=metrics["dataset"]["heldout_dates"])
    train_runs = [load_run(fp, dc) for fp in split.train_files]
    loaded = load_vehicle_spec(spec_path)
    vehicle = loaded.vehicle
    scales = compute_zero_baseline_scales(train_runs) if obj.get("axis_scaling") == "zero_rms" else None
    held = dict(obj.get("held_parameters") or {})
    if obj.get("excluded_axes"):
        from optimizer import initial_parameters
        scales, held_axes = excluded_axis_setup(obj["excluded_axes"], scales, initial_parameters(loaded))
        held.update(held_axes)
    horizon = obj.get("prediction_horizon_samples")
    names = params.param_names
    lb, ub = parameter_bounds(loaded)
    full = params.to_vector()
    free = np.array([n not in held and ub[i] > lb[i] for i, n in enumerate(names)])
    free_names = [n for n, f in zip(names, free) if f]

    t0 = time.monotonic()
    evaluator = ParallelJointResiduals(train_runs, vehicle, workers) if workers >= 2 else None
    try:
        def residuals(x_free):
            x = full.copy()
            x[free] = x_free
            fn = evaluator if evaluator is not None else joint_residuals
            return fn(x, train_runs, vehicle, scales, True, None, horizon)
        J = approx_derivative(residuals, full[free], method="2-point", bounds=(lb[free], ub[free]))
    finally:
        if evaluator is not None:
            evaluator.close()
    elapsed = time.monotonic() - t0

    width = (ub - lb)[free]
    Js = J * width  # each parameter in units of its allowed range, so no parameter dominates by its units
    S = np.linalg.svd(Js, compute_uv=False)
    _, _, Vt = np.linalg.svd(Js, full_matrices=False)
    cond = float("inf") if S[-1] < 1e-12 * S[0] else float(S[0] / S[-1])
    weak = []
    for i in range(1, len(S) + 1):
        if S[-i] > WEAK_COMBINATION_RATIO * S[0]:
            break
        v = Vt[-i]
        members = [free_names[j] for j in np.argsort(-np.abs(v)) if abs(v[j]) >= 0.3]
        weak.append({"parameters": members, "relative_strength": float(S[-i] / S[0])})
    corr = None
    try:
        cov = np.linalg.inv(J.T @ J)
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
    except np.linalg.LinAlgError:
        pass
    pairs = [{"param_a": a, "param_b": b, "correlation": float(c)}
             for a, b, c in report_correlated_pairs(corr, tuple(free_names), threshold=threshold)] if corr is not None else []
    report = {"condition_number": cond, "correlated_pairs": pairs, "weak_combinations": weak,
              "free_parameters": free_names, "held_parameters": sorted(held),
              "n_train_runs_used": len(train_runs), "correlation_threshold": threshold,
              "objective": {"sampling_rate_hz": rate, "torque_reference_offset_m": list(offset),
                            "current_world_mps": current, "axis_scaling": obj.get("axis_scaling"),
                            "excluded_axes": obj.get("excluded_axes") or [], "prediction_horizon_samples": horizon},
              "elapsed_s": round(elapsed, 1)}
    output = output or os.path.join(result_dir, "identifiability_report.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", help="a calibration output folder: analyse it with the exact objective it was "
                                         "fitted with (recommended; what `strathe calibrate` runs)")
    parser.add_argument("--workers", type=int, default=1, help="with --result: parallel worker processes")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, default=10.0)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--calibrated-params")
    parser.add_argument("--correlation-threshold", type=float, default=0.8)
    parser.add_argument("--n-runs", type=int, default=None,
                         help="Limit to this many training runs for a faster (but less "
                              "complete) check. Default: use all training runs.")
    parser.add_argument("--output", default="identifiability_report.json",
                         help="Where to write the structured JSON report.")
    args = parser.parse_args()
    if args.result:
        r = identifiability_for_result(args.result, args.data_dir, args.spec, args.workers,
                                       args.correlation_threshold,
                                       None if args.output == "identifiability_report.json" else args.output)
        print(f"Identifiability of {args.result}: {len(r['free_parameters'])} free parameters, condition number "
              f"{r['condition_number']:.3g}, {len(r['correlated_pairs'])} correlated pair(s), "
              f"{len(r['weak_combinations'])} barely-pinned combination(s) ({r['elapsed_s']:.0f} s)")
        return
    if not args.calibrated_params:
        parser.error("give --result <calibration folder> or --calibrated-params <yaml>")

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