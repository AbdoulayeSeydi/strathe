#!/usr/bin/env python3
"""
The real proof-of-concept: runs the actual scipy.optimize.least_squares
calibration against REAL HoloOcean physics (via the Custom Dynamics
integration in holoocean_backend.py), instead of the analytical stand-in.

Deliberately scaled down from the full stand-in calibration setup, given
HoloOcean's confirmed real per-step cost (~22ms/step, ~450x slower than
the stand-in on this machine — see time_holoocean_backend.py's output):
  - Each of the 25 training runs is TRUNCATED to --n-steps-per-run
    (default 150) instead of its full ~4160-step length
  - Warm-started from a PREVIOUSLY-CALIBRATED parameter set (the 10Hz
    stand-in result) rather than vehicle_spec.yaml's naive initial guess,
    since expensive evaluations make a good starting point far more
    valuable than usual
  - Single restart, modest max_nfev — this is a first proof-of-concept
    ("does calibrating against real HoloOcean physics produce a sensible
    result at all"), not the full, rigorous validation the stand-in
    calibration received. Treat results accordingly.

CHECKPOINTING: each real evaluation costs real wall-clock time (measured:
~124s standalone, observed: much higher during an actual optimizer run —
see PROJECT_STATUS.md's runtime-mystery note) and there's no protection
against losing hours of progress to a sleep/shutdown/crash mid-run. Uses
CalibrationConfig's existing on_iteration hook (previously unwired) to
save the BEST parameter vector seen so far to --checkpoint-path after
EVERY single evaluation, overwriting one file each time (not one file per
call — a crash only ever loses the most recent single evaluation's
progress, not everything). Pass --resume-from-checkpoint to continue from
a previous run's checkpoint instead of --warm-start-yaml.

Usage:
    python run_calibration_holoocean.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --warm-start-yaml ../calibration_output_10hz/calibrated_parameters.yaml \
        --n-steps-per-run 150 \
        --max-nfev 25 \
        --output-dir ../calibration_output_holoocean_poc

    # To resume an interrupted run instead:
    python run_calibration_holoocean.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --resume-from-checkpoint ../calibration_output_holoocean_poc/checkpoint.yaml \
        --n-steps-per-run 150 \
        --max-nfev 25 \
        --output-dir ../calibration_output_holoocean_poc
"""
import argparse
import json
import time
from pathlib import Path

from stall_watchdog import StallWatchdog

import numpy as np
import yaml

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, mean_run_rmse, rmse_by_dof
from optimizer import load_vehicle_spec, parameter_bounds, calibrate, CalibrationConfig
from dynamics_model import CalibrationParameters
from holoocean_backend import make_holoocean_simulate_fn, make_deterministic_environment
from mlflow_logging import log_calibration_run
from mlflow_schema import STATUS_SUCCESS, STATUS_CHECKPOINT_FALLBACK
from run_calibration import _get_git_commit
from keep_awake import keep_awake


def make_checkpoint_callback(checkpoint_path):
    """
    Returns an on_iteration(call_count, param_vec, cost) callback (matching
    CalibrationConfig's existing hook signature) that overwrites
    checkpoint_path after EVERY evaluation — not just "improving" ones —
    so the file always reflects the most recent call, while separately
    tracking and saving the BEST (lowest-cost) point seen so far, since
    some evaluations during gradient/Jacobian estimation are deliberately
    worse-than-current probes, not real forward progress.

    IMPORTANT: `cost` here is PLAIN 0.5*sum(residuals**2) on the RAW
    residuals — computed in optimizer.py's _wrap_residual_fn, BEFORE
    scipy's own internal soft_l1 robust-loss transform is applied. This
    will NOT numerically match the "Cost" column scipy prints in its own
    iteration table (which IS soft_l1-robustified) — both are legitimate,
    consistent measures of the same fit, they're just not the same
    quantity. Field names below are explicit about this to avoid the
    confusion of comparing the two directly.
    """
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
            "current_parameters": {n: float(getattr(current_params, n))
                                    for n in current_params.param_names},
            "best_cost_so_far": best["cost"],
            "best_parameters_so_far": {n: float(getattr(best_params, n))
                                        for n in best_params.param_names},
            "cost_meaning_note": "cost/best_cost_so_far are PLAIN 0.5*sum(residuals**2), NOT "
                                  "scipy's own soft_l1-robustified printed 'Cost' column — "
                                  "see this function's docstring for why they legitimately differ.",
        }
        # Overwrite, not append — one file always reflecting latest state.
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            yaml.dump(checkpoint, f, sort_keys=False)

    return callback


def load_checkpoint_warm_start(checkpoint_path: str) -> np.ndarray:
    with open(checkpoint_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    params = CalibrationParameters(**data["best_parameters_so_far"])
    print(f"Resuming from checkpoint: call_count={data['call_count']}, "
          f"best_cost_so_far={data['best_cost_so_far']:.6f}")
    return params.to_vector()


def truncate_run(run, n_steps):
    n = min(n_steps, run.n_samples)
    return type(run)(
        filepath=run.filepath, forces_torques=run.forces_torques[:n],
        roll_pitch=run.roll_pitch[:n], real_nu=run.real_nu[:n], nu0=run.nu0,
        dt=run.dt, sample_rate_hz=run.sample_rate_hz, date=run.date,
    )


def load_warm_start(yaml_path: str) -> np.ndarray:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    params = CalibrationParameters(**data["parameters"])
    return params.to_vector()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--warm-start-yaml", default=None,
                         help="Path to a previously-calibrated calibrated_parameters.yaml "
                              "to warm-start from. Required unless --resume-from-checkpoint "
                              "is given instead.")
    parser.add_argument("--resume-from-checkpoint", default=None,
                         help="Path to a checkpoint.yaml from a previously-interrupted run — "
                              "resumes from its best-parameters-so-far instead of "
                              "--warm-start-yaml. Overrides --warm-start-yaml if both given.")
    parser.add_argument("--checkpoint-path", default=None,
                         help="Where to save the live checkpoint (overwritten after every "
                              "single evaluation). Defaults to <output-dir>/checkpoint.yaml.")
    parser.add_argument("--n-steps-per-run", type=int, default=150)
    parser.add_argument("--max-nfev", type=int, default=25)
    parser.add_argument("--output-dir", default="../calibration_output_holoocean_poc")
    parser.add_argument("--no-dof-weighting", action="store_true",
                         help="Disable the p/r DOF weighting (finding #27) for a clean, "
                              "isolated test of other changes.")
    args = parser.parse_args()

    if args.resume_from_checkpoint is None and args.warm_start_yaml is None:
        raise SystemExit("Must provide either --warm-start-yaml or --resume-from-checkpoint.")

    csv_paths, duplicates = discover_csv_files(args.data_dir)
    for dup_path, kept_path in duplicates:
        print(f"Duplicate filename skipped: {dup_path} (kept {kept_path})")

    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)

    print(f"Loading and truncating {len(split.train_files)} training runs to "
          f"{args.n_steps_per_run} steps each...")
    train_runs = [truncate_run(load_run(fp, dataset_config), args.n_steps_per_run)
                  for fp in split.train_files]
    total_steps = sum(r.n_samples for r in train_runs)
    print(f"Loaded. Total steps across all training runs: {total_steps}")

    loaded_spec = load_vehicle_spec(args.spec)
    vehicle = loaded_spec.vehicle

    if args.resume_from_checkpoint is not None:
        print(f"\nResuming from checkpoint {args.resume_from_checkpoint}...")
        x0_warm_start = load_checkpoint_warm_start(args.resume_from_checkpoint)
    else:
        print(f"\nLoading warm-start parameters from {args.warm_start_yaml}...")
        x0_warm_start = load_warm_start(args.warm_start_yaml)
    warm_start_params = CalibrationParameters.from_vector(x0_warm_start)

    checkpoint_path = args.checkpoint_path or str(Path(args.output_dir) / "checkpoint.yaml")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Checkpointing every evaluation to: {checkpoint_path}")

    print("\nCreating HoloOcean environment (real Unreal Engine startup)...")
    env = make_deterministic_environment("SimpleUnderwater-Hovering")
    simulate_fn = make_holoocean_simulate_fn(env)
    print("Environment created — all sensor noise zeroed for calibration (see "
          "make_deterministic_environment's docstring for why).\n")

    print("Evaluating BASELINE (warm-start parameters, before any HoloOcean-backed fitting)...")
    t0 = time.monotonic()
    baseline_rmse = mean_run_rmse(warm_start_params, train_runs, vehicle, simulate_fn=simulate_fn)
    baseline_time = time.monotonic() - t0
    print(f"Baseline RMSE (HoloOcean, warm-start params): {baseline_rmse:.4f}  "
          f"(took {baseline_time:.1f}s for this one evaluation)\n")

    naive_estimate = baseline_time * args.max_nfev
    print(f"NAIVE estimate ({args.max_nfev} evaluations x this baseline time): "
          f"~{naive_estimate/60:.1f} min")
    print("CAUTION: a prior real run measured standalone per-eval cost at ~124s, but the "
          "ACTUAL cost during a live optimizer run came in far higher (~900s/eval observed) — "
          "an unexplained gap (see PROJECT_STATUS.md's runtime-mystery note). Treat the naive "
          "estimate above as an optimistic floor, not a real prediction. Checkpointing is on, "
          "so an interrupted run can be resumed rather than needing to guess right upfront.\n")

    # --- Auto-backup: protect whatever already exists in the output dir
    # from a PRIOR segment before this run touches it at all. Fixes the
    # exact class of loss that happened to segment 3's real result
    # (silently overwritten in place, no versioning, nothing recoverable
    # afterward). Cheap - a plain file copy, only runs if something is
    # already there to protect.
    from pathlib import Path as _Path
    import shutil as _shutil
    import time as _time
    _output_dir = _Path(checkpoint_path).parent
    _backup_tag = _time.strftime("%Y%m%d_%H%M%S")
    for _fname in ("calibrated_parameters.yaml", "checkpoint.yaml"):
        _existing = _output_dir / _fname
        if _existing.exists():
            _backup_path = _output_dir / f"{_existing.stem}_backup_{_backup_tag}{_existing.suffix}"
            _shutil.copy2(_existing, _backup_path)
            print(f"Auto-backed up existing {_fname} -> {_backup_path.name} before this run starts")

    # --- Timing log: one line per real evaluation with a real timestamp -
    # same capability already built for run_calibration.py's stand-in
    # backend, now added here too, since the performance mystery just
    # resurfaced again in THIS (HoloOcean) backend specifically
    # (confirmed: 649s/eval vs the established 124.2s baseline, this
    # session). Wraps the existing checkpoint callback rather than
    # modifying it, so the proven checkpointing behavior is untouched.
    _timing_log_path = _output_dir / "timing_log_holoocean.csv"
    _timing_log_is_new = not _timing_log_path.exists()

    def _with_timing_log(inner_callback):
        def wrapped(call_count, param_vec, cost):
            inner_callback(call_count, param_vec, cost)
            with open(_timing_log_path, "a", encoding="utf-8") as f:
                if _timing_log_is_new and call_count == 1:
                    f.write("call_count,unix_timestamp,cost\n")
                f.write(f"{call_count},{_time.time()},{cost}\n")
        return wrapped

    # Custom per-DOF weighting - NOT standard scale-normalization.
    # Researched and confirmed (PROJECT_STATUS.md): p is genuinely
    # under-weighted by its small natural scale (0.125, smallest of all
    # 6 DOFs) - standard normalization helps it correctly. But r has the
    # LARGEST natural scale (0.5625) and still regressed across multiple
    # real segments - standard normalization would REDUCE r's weight
    # below even today's unweighted baseline (confirmed: 0.67x vs today's
    # 1.5x), making its regression worse, not better. This applies a
    # deliberate boost to BOTH p and r beyond their natural scales.
    from objective import compute_dof_scales as _compute_dof_scales
    _dof_labels = ["u", "v", "w", "p", "q", "r"]
    dof_scales = _compute_dof_scales(train_runs).copy()
    if getattr(args, "no_dof_weighting", False):
        dof_scales = None
        print("DOF weighting DISABLED (--no-dof-weighting) - standard unweighted objective, "
              "isolating other changes from the separately-tested weighting approach "
              "(PROJECT_STATUS.md finding #27)."
        )
    else:
        dof_scales[_dof_labels.index("p")] *= 0.3   # p: strongly boosted - confirmed under-weighted
        dof_scales[_dof_labels.index("r")] *= 0.15  # r: boosted BEYOND its already-large natural scale
        print(f"Custom DOF weighting active - p and r boosted beyond natural scale "
              f"(see PROJECT_STATUS.md for the research behind this): {dict(zip(_dof_labels, dof_scales))}")

    config = CalibrationConfig(
        n_restarts=1,
        seed=42,
        max_nfev=args.max_nfev,
        x0_override=x0_warm_start,
        initialization_strategy="fixed",  # start EXACTLY at the warm-start point, no perturbation
        on_iteration=_with_timing_log(make_checkpoint_callback(checkpoint_path)),
        stall_watchdog=StallWatchdog(str(_output_dir)),  # see stall_watchdog.py
        simulate_fn=simulate_fn,
        dof_scales=dof_scales,
        verbose=2,
    )

    print("Running calibration against REAL HoloOcean physics...")
    t0 = time.monotonic()
    converged = True
    try:
        result = calibrate(loaded_spec, train_runs, config)
        elapsed = time.monotonic() - t0
        best_x = result.restarts[0].x
        best_cost = result.restarts[0].cost
        best_nfev = result.restarts[0].nfev
    except RuntimeError as e:
        # calibrate() raises if scipy's own strict tolerances (ftol/xtol/gtol)
        # weren't satisfied within max_nfev — the RIGHT default in general,
        # but wrong here: for an expensive, budget-limited HoloOcean run,
        # hitting the evaluation cap mid-improvement is EXPECTED, not an
        # error. Real progress isn't lost — on_iteration already checkpointed
        # the best point seen, so fall back to that instead of crashing.
        elapsed = time.monotonic() - t0
        converged = False
        print(f"\nDid not fully converge within max_nfev ({e}) — this is expected for a "
              f"budget-limited run, not necessarily a problem. Falling back to the best "
              f"point from the checkpoint instead of losing the run's progress.")
        with open(checkpoint_path, encoding="utf-8") as f:
            checkpoint_data = yaml.safe_load(f)
        best_params_from_checkpoint = CalibrationParameters(**checkpoint_data["best_parameters_so_far"])
        best_x = best_params_from_checkpoint.to_vector()
        best_cost = checkpoint_data["best_cost_so_far"]
        best_nfev = checkpoint_data["call_count"]

    print(f"\nCalibration step finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    fitted_params = CalibrationParameters.from_vector(best_x)

    print("\nEvaluating fitted parameters...")
    fitted_rmse = mean_run_rmse(fitted_params, train_runs, vehicle, simulate_fn=simulate_fn)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Converged (scipy's strict criteria): {converged}, cost: {best_cost:.6f}, nfev: {best_nfev}")
    if not converged:
        print("NOTE: not fully converged — this is the best point found within the evaluation "
              "budget, not a final optimum. Resume with --resume-from-checkpoint and a higher "
              "--max-nfev to continue improving it.")
    print(f"RMSE (HoloOcean) — warm-start baseline: {baseline_rmse:.4f}")
    print(f"RMSE (HoloOcean) — after fitting:        {fitted_rmse:.4f}")
    if baseline_rmse > 0:
        print(f"Improvement: {(baseline_rmse - fitted_rmse) / baseline_rmse * 100:+.1f}%")

    print("\nParameter changes (warm-start -> HoloOcean-fitted):")
    for name in warm_start_params.param_names:
        before = getattr(warm_start_params, name)
        after = getattr(fitted_params, name)
        print(f"  {name:<18}{before:>12.4f} -> {after:>12.4f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "calibrated_parameters.yaml", "w", encoding="utf-8") as f:
        yaml.dump({
            "parameters": {n: float(getattr(fitted_params, n)) for n in fitted_params.param_names},
            "note": "PROOF-OF-CONCEPT: HoloOcean-backed, truncated runs, single restart — "
                    "not the full rigorous validation the stand-in calibration received",
        }, f, sort_keys=False)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "baseline_rmse_holoocean": baseline_rmse,
            "fitted_rmse_holoocean": fitted_rmse,
            "n_steps_per_run": args.n_steps_per_run,
            "n_train_runs": len(train_runs),
            "total_steps": total_steps,
            "converged": converged,
            "cost": best_cost,
            "nfev": best_nfev,
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"\nSaved results to {out}/")

    if converged:
        _status = STATUS_SUCCESS
        _status_reason = f"scipy converged, cost={best_cost:.6f}"
    else:
        _status = STATUS_CHECKPOINT_FALLBACK
        _status_reason = (f"maximum function evaluations reached, best checkpoint "
                           f"substituted (nfev={best_nfev})")

    log_calibration_run(
        script="run_calibration_holoocean",
        run_name=f"run_calibration_holoocean_{_time.strftime('%Y-%m-%d_%H%M')}",
        status=_status, status_reason=_status_reason,
        git_commit=_get_git_commit(), vehicle_spec_path=args.spec,
        params={
            "optimizer": "scipy.optimize.least_squares (trf)",
            "optimizer_settings": (f"n_steps_per_run={args.n_steps_per_run}, "
                                    f"dof_weighting={'disabled' if args.no_dof_weighting else 'enabled'}"),
            "vehicle_spec_version": loaded_spec.raw.get("schema_version", "unknown"),
            "dataset_path": args.data_dir,
            "dataset_file_count": len(train_runs),
            "seed": config.seed,
            "sampling_rate_hz": args.sampling_rate_hz,
            "max_nfev": args.max_nfev,
        },
        # NOTE: rmse_total only, deliberately - per-DOF metrics would need
        # a SEPARATE, expensive HoloOcean-backed evaluation loop on top of
        # the already-costly fitted_rmse call above, a real, bad cost
        # tradeoff for this specific script (see PROJECT_STATUS.md).
        metrics={
            "rmse_total": fitted_rmse,
            "rmse_baseline": baseline_rmse,
            "runtime_seconds": elapsed,
            "nfev": best_nfev,
        },
        artifacts=[args.spec, str(out / "calibrated_parameters.yaml"),
                   str(out / "metrics.json")],
        tags={"run_type": "real"},
    )


if __name__ == "__main__":
    with keep_awake("HoloOcean calibration"):  # PROJECT_STATUS.md #49
        main()
