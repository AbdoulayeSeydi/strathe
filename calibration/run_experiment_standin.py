#!/usr/bin/env python3
"""
Runs a stand-in (fast, non-MuJoCo) calibration on the FULL training set,
for the parameter-reduction experiments (A, B1, B2, B3) testing whether
Xu_dot/Yv_dot/Zw_dot's near-total three-way correlation (>=0.916 pairwise,
confirmed via diagnose_calibration.py's Jacobian on the real dataset) means
they aren't independently identifiable from this data -- not evidence of
missing sway physics (that hypothesis was separately closed this session).

DELIBERATELY uses the stand-in, not MuJoCo: teacher-forced MuJoCo is
confirmed mathematically identical to the stand-in's own output and
gradient, so there is no reason to pay MuJoCo's overhead for this
exploration -- MuJoCo's role is validating whichever structure this
experiment settles on, once, not iterating through four candidates.

DELIBERATELY uses the FULL training set, not a subset: unlike the earlier
MuJoCo backend-comparison experiments (where a 10-run subset was a
reasonable cost/time tradeoff for a DIFFERENT question), an
identifiability question is specifically about what THIS DATASET can
constrain -- a smaller subset would only worsen identifiability and bias
the answer toward "fixing helps", regardless of what the full dataset
would actually show.

COST WARNING (measured directly, not assumed): the stand-in's own
per-call cost on the full 25 training runs is ~65s (confirmed from
diagnose_calibration.py's own Jacobian eval timings), not fast in
absolute terms even though it is much faster than MuJoCo proportionally.
A full run needing several hundred real calls (trf's own Jacobian
overhead applies here identically to the MuJoCo experiments -- expect
real call counts well over --max-nfev) can still take multiple hours.
Defaults --max-nfev to a modest 30, not the hundreds implied by "run to
strict convergence" -- this is a comparative, diagnostic experiment, not
a final production fit; a partial but well-progressed result is
sufficient to see whether held-out RMSE moves.

Usage:
    python3 run_experiment_standin.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset/dataset \
        --spec vehicle_spec_expA_fix_all_added_mass.yaml \
        --sampling-rate-hz 10.0 \
        --warm-start-yaml calibrated_parameters_a1.yaml \
        --max-nfev 30 \
        --output-dir ../calibration_output_expA

*** UNVERIFIED against real data in this sandbox -- no real CSV files,
no real calibrated_parameters_a1.yaml, no real mlflow_logging, in this
sandbox. This has NOT been run end-to-end against real data before being
handed off -- syntax-checked and core logic tested against a fake
harness (see conversation), matching this session's established
discipline for every prior handoff of this kind.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, mean_run_rmse
from optimizer import load_vehicle_spec, calibrate, CalibrationConfig, parameter_bounds, initial_parameters
from dynamics_model import CalibrationParameters


def make_checkpoint_callback(checkpoint_path):
    """Same discipline as run_calibration_mujoco.py's own callback --
    overwrite after every evaluation, separately track the best point
    seen, so an interrupted run never loses more than its most recent
    evaluation's progress."""
    best = {"cost": float("inf"), "param_vec": None}

    def callback(call_count, param_vec, cost):
        if cost < best["cost"]:
            best["cost"] = cost
            best["param_vec"] = param_vec.copy()
        current_params = CalibrationParameters.from_vector(param_vec)
        best_params = (CalibrationParameters.from_vector(best["param_vec"])
                        if best["param_vec"] is not None else current_params)
        checkpoint = {
            "call_count": call_count,
            "current_cost": cost,
            "current_parameters": {n: float(getattr(current_params, n)) for n in current_params.param_names},
            "best_cost_so_far": best["cost"],
            "best_parameters_so_far": {n: float(getattr(best_params, n)) for n in best_params.param_names},
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            yaml.dump(checkpoint, f, sort_keys=False)

    return callback


def load_warm_start(yaml_path: str) -> np.ndarray:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    params = CalibrationParameters(**data["parameters"])
    return params.to_vector()


def reconcile_warm_start_with_bounds(x0: np.ndarray, loaded_spec, warm_start_path: str) -> np.ndarray:
    """
    A warm-start vector was fit under a DIFFERENT spec's bounds (e.g.
    calibrated_parameters_a1.yaml's own free fit) than the one this run
    is actually using -- when an experiment narrows a parameter's bounds
    around its literature value, the warm-start's OWN fitted value for
    that same parameter can land far outside the new range (confirmed
    directly: A1's Xu_dot/Yv_dot/Zw_dot sit 20-25 units past this
    experiment's narrow bounds around their literature values -- not a
    rounding issue). scipy's least_squares refuses to start outside its
    own bounds. Clamps any out-of-range component to the NEAREST bound
    of the CURRENT spec, printing exactly what changed -- a silent clamp
    here would hide a real, sometimes large adjustment.
    """
    lb, ub = parameter_bounds(loaded_spec)
    names = initial_parameters(loaded_spec).param_names
    x0 = x0.copy()
    adjusted = []
    for i, (name, lo, hi) in enumerate(zip(names, lb, ub)):
        if x0[i] < lo or x0[i] > hi:
            old_val = x0[i]
            x0[i] = min(max(x0[i], lo), hi)
            adjusted.append((name, old_val, x0[i], lo, hi))
    if adjusted:
        print(f"\nWARM-START RECONCILIATION: {warm_start_path}'s own fitted value(s) fall outside "
              f"this experiment's bounds -- clamped to fit, not silently:")
        for name, old_val, new_val, lo, hi in adjusted:
            print(f"  {name:<16} {old_val:>12.4f} -> {new_val:>12.4f}  (bounds [{lo}, {hi}])")
    return x0


def count_pinned(params: CalibrationParameters, loaded_spec, tol_frac: float = 0.01) -> list:
    """Matches diagnose_calibration.py's own active_bounds_report exactly
    -- same tolerance, same logic -- so pinning counts are comparable
    across this whole investigation, not a subtly different definition."""
    lb, ub = parameter_bounds(loaded_spec)
    names = initial_parameters(loaded_spec).param_names
    hits = []
    for name, lo, hi in zip(names, lb, ub):
        val = getattr(params, name)
        width = hi - lo
        tol = tol_frac * width if width > 0 else 0
        if abs(val - lo) <= tol:
            hits.append((name, "lower", lo, val))
        elif abs(val - hi) <= tol:
            hits.append((name, "upper", hi, val))
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--warm-start-yaml", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--max-nfev", type=int, default=30,
                         help="Modest by design -- see this script's own COST WARNING. "
                              "This is a comparative experiment, not a final production fit.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    csv_paths, duplicates = discover_csv_files(args.data_dir)
    for dup_path, kept_path in duplicates:
        print(f"Duplicate filename skipped: {dup_path} (kept {kept_path})")

    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)

    print(f"Loading ALL {len(split.train_files)} training runs at full length "
          f"(no truncation -- identifiability needs the real dataset, not a subset)...")
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files]
    total_steps = sum(r.n_samples for r in train_runs)
    print(f"Loaded. Total steps: {total_steps}")

    heldout_runs = [load_run(fp, dataset_config) for fp in split.heldout_files]
    print(f"Held-out runs for validation: {len(heldout_runs)}")

    loaded_spec = load_vehicle_spec(args.spec)
    vehicle = loaded_spec.vehicle

    print(f"\nLoading warm-start parameters from {args.warm_start_yaml}...")
    x0_warm_start = load_warm_start(args.warm_start_yaml)
    x0_warm_start = reconcile_warm_start_with_bounds(x0_warm_start, loaded_spec, args.warm_start_yaml)
    warm_start_params = CalibrationParameters.from_vector(x0_warm_start)

    checkpoint_path = args.checkpoint_path or str(Path(args.output_dir) / "checkpoint.yaml")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Checkpointing every evaluation to: {checkpoint_path}")

    print("\nEvaluating BASELINE (warm-start parameters, stand-in, full training set)...")
    t0 = time.monotonic()
    baseline_train_rmse = mean_run_rmse(warm_start_params, train_runs, vehicle, simulate_fn=None)
    baseline_time = time.monotonic() - t0
    print(f"Baseline train RMSE: {baseline_train_rmse:.4f}  (took {baseline_time:.1f}s -- this IS "
          f"the real per-call cost on this full dataset, not a smaller-subset number)")
    baseline_heldout_rmse = mean_run_rmse(warm_start_params, heldout_runs, vehicle, simulate_fn=None)
    print(f"Baseline held-out RMSE: {baseline_heldout_rmse:.4f}")

    est_hours_per_300_calls = (baseline_time * 300) / 3600
    print(f"\nRough scale check: 300 real calls at this per-call cost would take "
          f"~{est_hours_per_300_calls:.1f} hours. Real call count for --max-nfev={args.max_nfev} "
          f"is NOT {args.max_nfev} -- expect substantially more (trf's own Jacobian overhead, "
          f"confirmed repeatedly this session). Checkpointing is on regardless.\n")

    config = CalibrationConfig(
        n_restarts=1, seed=42, max_nfev=args.max_nfev,
        x0_override=x0_warm_start, initialization_strategy="fixed",
        on_iteration=make_checkpoint_callback(checkpoint_path),
        simulate_fn=None,  # the stand-in, deliberately -- see module docstring
        verbose=2,
    )

    print("Running calibration against the stand-in (full training set)...")
    t0 = time.monotonic()
    converged = True
    try:
        result = calibrate(loaded_spec, train_runs, config)
        elapsed = time.monotonic() - t0
        best_x = result.restarts[0].x
        best_cost = result.restarts[0].cost
        best_nfev = result.restarts[0].nfev
    except RuntimeError as e:
        elapsed = time.monotonic() - t0
        converged = False
        print(f"\nDid not fully converge within max_nfev ({e}) -- expected for a "
              f"budget-limited comparative run. Falling back to the best checkpointed point.")
        with open(checkpoint_path, encoding="utf-8") as f:
            checkpoint_data = yaml.safe_load(f)
        best_params_from_checkpoint = CalibrationParameters(**checkpoint_data["best_parameters_so_far"])
        best_x = best_params_from_checkpoint.to_vector()
        best_cost = checkpoint_data["best_cost_so_far"]
        best_nfev = checkpoint_data["call_count"]

    print(f"\nCalibration finished in {elapsed:.1f}s ({elapsed/60:.1f} min, {elapsed/3600:.2f} hours)")

    fitted_params = CalibrationParameters.from_vector(best_x)

    print("\nEvaluating fitted parameters...")
    fitted_train_rmse = mean_run_rmse(fitted_params, train_runs, vehicle, simulate_fn=None)
    fitted_heldout_rmse = mean_run_rmse(fitted_params, heldout_runs, vehicle, simulate_fn=None)

    pinned = count_pinned(fitted_params, loaded_spec)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Converged (scipy's strict criteria): {converged}, cost: {best_cost:.6f}, nfev: {best_nfev}")
    print(f"Train RMSE:    baseline={baseline_train_rmse:.4f}, fitted={fitted_train_rmse:.4f}")
    print(f"Held-out RMSE: baseline={baseline_heldout_rmse:.4f}, fitted={fitted_heldout_rmse:.4f}")
    if baseline_heldout_rmse > 0:
        pct = (baseline_heldout_rmse - fitted_heldout_rmse) / baseline_heldout_rmse * 100
        print(f"Held-out improvement over warm-start: {pct:+.1f}%")
    print(f"\nPinned parameters ({len(pinned)}/{len(fitted_params.param_names)}):")
    for name, side, bound_val, actual_val in pinned:
        print(f"  {name:<16} at {side} bound ({bound_val})")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "calibrated_parameters.yaml", "w", encoding="utf-8") as f:
        yaml.dump({
            "parameters": {n: float(getattr(fitted_params, n)) for n in fitted_params.param_names},
            "note": f"Parameter-reduction experiment (see conversation), spec={args.spec}, "
                    f"stand-in backend, full training set.",
        }, f, sort_keys=False)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "spec": args.spec,
            "baseline_train_rmse": baseline_train_rmse, "fitted_train_rmse": fitted_train_rmse,
            "baseline_heldout_rmse": baseline_heldout_rmse, "fitted_heldout_rmse": fitted_heldout_rmse,
            "n_pinned": len(pinned), "pinned_params": [p[0] for p in pinned],
            "converged": converged, "cost": best_cost, "nfev": best_nfev, "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()