#!/usr/bin/env python3
"""
Runs the actual scipy.optimize.least_squares calibration against MuJoCo's
teacher_forced=True backend (mujoco_backend.make_mujoco_calibration_simulate_fn),
adapted from run_calibration_holoocean.py (kept unchanged, still available
for Decision 2's occasional, independent physics cross-check role — see
phase2_config.py).

teacher_forced=True specifically, not forward-sim: confirmed (separate
session) its residual gradient is numerically IDENTICAL to the stand-in's
own (matching to 4 decimal places) at the exact point forward-sim's own
gradient was found unstable (>20x spread across finite-difference step
sizes) — calibration-safety confirmed directly, not assumed. Forward-sim
mode (mujoco_backend.make_mujoco_simulate_fn) is fine for held-out
EVALUATION of already-fitted parameters (see replay_comparison.py) but
NOT for fitting itself — see mujoco_backend.py's own docstring.

WHAT'S DIFFERENT FROM run_calibration_holoocean.py, and why:
  - No 150-step truncation by default (--n-steps-per-run defaults to 0,
    meaning full run length). HoloOcean needed truncation because of a
    confirmed IPC deadlock on episodes beyond a few hundred steps, plus
    its ~400x-the-stand-in overhead. MuJoCo has neither issue: no deadlock
    found in this investigation, and overhead measured at only ~1.9x the
    stand-in's (mujoco_calibration_experiment.py's own timing output) —
    full-length runs are tractable.
  - The p/r DOF-weighting boost (dof_scales[p] *= 0.3, [r] *= 0.15) is
    NOT carried over. That weighting was researched and tuned SPECIFICALLY
    for a rotational weakness confirmed in HoloOcean's own, separately-
    implemented hydrodynamics (PROJECT_STATUS.md finding #27) — MuJoCo
    runs your OWN dynamics_model.py equations, not a different physics
    engine, so there's no basis yet to assume the same weakness exists
    here. --dof-weighting-p/--dof-weighting-r are exposed if you want to
    test for it, but default to no special weighting.
  - Metrics/output field names say "_mujoco", not "_holoocean" — an
    honest label matters more here than a one-line diff.

CHECKPOINTING, auto-backup, and the timing log are unchanged in spirit
from run_calibration_holoocean.py (all genuinely useful regardless of
backend) — same on_iteration hook pattern, same "overwrite the best point
seen after every evaluation" discipline.

Usage:
    python run_calibration_mujoco.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --warm-start-yaml ../calibration_output_10hz/calibrated_parameters.yaml \
        --max-nfev 60 \
        --output-dir ../calibration_output_mujoco

    # To resume an interrupted run instead:
    python run_calibration_mujoco.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --resume-from-checkpoint ../calibration_output_mujoco/checkpoint.yaml \
        --max-nfev 60 \
        --output-dir ../calibration_output_mujoco

*** UNVERIFIED against real data or the real mlflow_logging/mlflow_schema/
run_calibration modules in this sandbox (none of those are available
here) -- syntax-checked and the core MuJoCo wiring tested directly
against a fake harness, but this exact file has NOT been run end-to-end
for real before being handed off. Please treat the first real run as a
first real run, not as something already proven.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
import mujoco

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, mean_run_rmse, rmse_by_dof
from optimizer import load_vehicle_spec, parameter_bounds, calibrate, CalibrationConfig
from dynamics_model import CalibrationParameters
from mujoco_backend import make_mujoco_calibration_simulate_fn
from mlflow_logging import log_calibration_run
from mlflow_schema import STATUS_SUCCESS, STATUS_CHECKPOINT_FALLBACK
from run_calibration import _get_git_commit


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
    if not n_steps:
        return run  # 0/None means "no truncation" — see module docstring for why this
                     # differs from run_calibration_holoocean.py's own default
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
    parser.add_argument("--mjcf-path", default="b0_single_body_placeholder.xml")
    parser.add_argument("--n-train-runs", type=int, default=None,
                         help="Limit to the first N training runs (sorted), for a "
                              "cost/time tradeoff at FULL trajectory length instead of "
                              "truncating every run's own length. Default: use all "
                              "available training runs.")
    parser.add_argument("--n-steps-per-run", type=int, default=0,
                         help="0 (default) means full run length — see module docstring for "
                              "why this differs from run_calibration_holoocean.py's 150-step "
                              "default. Set explicitly for a faster, truncated run if desired.")
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--output-dir", default="../calibration_output_mujoco")
    parser.add_argument("--dof-weight-p", type=float, default=1.0,
                         help="Multiplier on p's natural dof_scale — 1.0 (default) means no "
                              "special weighting. run_calibration_holoocean.py used 0.3, "
                              "tuned specifically for a HoloOcean-specific weakness — not "
                              "carried over here by default, see module docstring.")
    parser.add_argument("--dof-weight-r", type=float, default=1.0,
                         help="Same as --dof-weight-p, for r. run_calibration_holoocean.py "
                              "used 0.15.")
    args = parser.parse_args()

    if args.resume_from_checkpoint is None and args.warm_start_yaml is None:
        raise SystemExit("Must provide either --warm-start-yaml or --resume-from-checkpoint.")

    csv_paths, duplicates = discover_csv_files(args.data_dir)
    for dup_path, kept_path in duplicates:
        print(f"Duplicate filename skipped: {dup_path} (kept {kept_path})")

    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)

    train_files = sorted(split.train_files)
    if args.n_train_runs is not None:
        train_files = train_files[:args.n_train_runs]
        print(f"Limiting to the first {len(train_files)} training run(s) "
              f"(of {len(split.train_files)} available) — see --n-train-runs.")

    if args.n_steps_per_run:
        print(f"Loading and truncating {len(train_files)} training runs to "
              f"{args.n_steps_per_run} steps each...")
    else:
        print(f"Loading {len(train_files)} training runs at FULL length "
              f"(--n-steps-per-run=0, MuJoCo's confirmed ~1.9x overhead makes this "
              f"tractable, unlike HoloOcean's ~400x)...")
    train_runs = [truncate_run(load_run(fp, dataset_config), args.n_steps_per_run)
                  for fp in train_files]
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

    print("\nCreating MuJoCo model/data (teacher_forced=True — confirmed calibration-safe: "
          "gradient identical to the stand-in's own, to 4 decimal places, at the exact "
          "point forward-sim's own gradient was found unstable)...")
    model = mujoco.MjModel.from_xml_path(args.mjcf_path)
    data = mujoco.MjData(model)
    simulate_fn = make_mujoco_calibration_simulate_fn(model, data)
    print("Model created.\n")

    print("Evaluating BASELINE (warm-start parameters, before any MuJoCo-backed fitting)...")
    t0 = time.monotonic()
    baseline_rmse = mean_run_rmse(warm_start_params, train_runs, vehicle, simulate_fn=simulate_fn)
    baseline_time = time.monotonic() - t0
    print(f"Baseline RMSE (MuJoCo, warm-start params): {baseline_rmse:.4f}  "
          f"(took {baseline_time:.1f}s for this one evaluation)\n")

    naive_estimate = baseline_time * args.max_nfev
    print(f"NAIVE estimate ({args.max_nfev} evaluations x this baseline time): "
          f"~{naive_estimate/60:.1f} min")
    print("CAUTION: max_nfev does NOT cap real function-call volume — trf's own numerical "
          "Jacobian needs ~N+1 (N=22 free parameters) evaluations PER ITERATION, so real "
          "calls typically run well over 10x this naive estimate (confirmed directly: a "
          "max_nfev=10 smoke test on this script took 230 real calls, not 10 — see "
          "build_calibrated_parameters.py's own comment on this same gotcha). What IS "
          "confirmed reliable is the PER-CALL cost itself: unlike HoloOcean's own "
          "documented, unexplained gap between standalone and live-optimizer per-eval cost, "
          "MuJoCo's per-call cost during a live run matches its standalone baseline closely "
          "(confirmed: 4.71s/call live vs 4.8s standalone, in that same smoke test). "
          "Checkpointing is on regardless, so an interrupted run can always be resumed.\n")

    # --- Auto-backup: protect whatever already exists in the output dir
    # from a PRIOR run before this run touches it at all.
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

    # --- Timing log: one line per real evaluation with a real timestamp.
    _timing_log_path = _output_dir / "timing_log_mujoco.csv"
    _timing_log_is_new = not _timing_log_path.exists()

    def _with_timing_log(inner_callback):
        def wrapped(call_count, param_vec, cost):
            inner_callback(call_count, param_vec, cost)
            with open(_timing_log_path, "a", encoding="utf-8") as f:
                if _timing_log_is_new and call_count == 1:
                    f.write("call_count,unix_timestamp,cost\n")
                f.write(f"{call_count},{_time.time()},{cost}\n")
        return wrapped

    from objective import compute_dof_scales as _compute_dof_scales
    _dof_labels = ["u", "v", "w", "p", "q", "r"]
    dof_scales = _compute_dof_scales(train_runs).copy()
    if args.dof_weight_p != 1.0 or args.dof_weight_r != 1.0:
        dof_scales[_dof_labels.index("p")] *= args.dof_weight_p
        dof_scales[_dof_labels.index("r")] *= args.dof_weight_r
        print(f"Custom DOF weighting active (--dof-weight-p={args.dof_weight_p}, "
              f"--dof-weight-r={args.dof_weight_r}): {dict(zip(_dof_labels, dof_scales))}")
    else:
        print(f"No special DOF weighting (defaults) — natural training-derived scales: "
              f"{dict(zip(_dof_labels, dof_scales))}")

    config = CalibrationConfig(
        n_restarts=1,
        seed=42,
        max_nfev=args.max_nfev,
        x0_override=x0_warm_start,
        initialization_strategy="fixed",  # start EXACTLY at the warm-start point, no perturbation
        on_iteration=_with_timing_log(make_checkpoint_callback(checkpoint_path)),
        simulate_fn=simulate_fn,
        dof_scales=dof_scales,
        verbose=2,
    )

    print("Running calibration against MuJoCo (teacher_forced=True)...")
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
        # weren't satisfied within max_nfev — EXPECTED for a budget-limited
        # run on this 22-parameter model, not necessarily an error. Real
        # progress isn't lost — on_iteration already checkpointed the best
        # point seen, so fall back to that instead of crashing.
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
    print(f"RMSE (MuJoCo) — warm-start baseline: {baseline_rmse:.4f}")
    print(f"RMSE (MuJoCo) — after fitting:        {fitted_rmse:.4f}")
    if baseline_rmse > 0:
        print(f"Improvement: {(baseline_rmse - fitted_rmse) / baseline_rmse * 100:+.1f}%")

    print("\nParameter changes (warm-start -> MuJoCo-fitted):")
    for name in warm_start_params.param_names:
        before = getattr(warm_start_params, name)
        after = getattr(fitted_params, name)
        print(f"  {name:<18}{before:>12.4f} -> {after:>12.4f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "calibrated_parameters.yaml", "w", encoding="utf-8") as f:
        yaml.dump({
            "parameters": {n: float(getattr(fitted_params, n)) for n in fitted_params.param_names},
            "note": "MuJoCo-backed (teacher_forced=True), see run_calibration_mujoco.py's "
                    "own docstring for what differs from the earlier HoloOcean proof-of-concept.",
        }, f, sort_keys=False)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "baseline_rmse_mujoco": baseline_rmse,
            "fitted_rmse_mujoco": fitted_rmse,
            "n_steps_per_run": args.n_steps_per_run or "full",
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
        script="run_calibration_mujoco",
        run_name=f"run_calibration_mujoco_{_time.strftime('%Y-%m-%d_%H%M')}",
        status=_status, status_reason=_status_reason,
        git_commit=_get_git_commit(), vehicle_spec_path=args.spec,
        params={
            "optimizer": "scipy.optimize.least_squares (trf)",
            "optimizer_settings": (f"n_steps_per_run={args.n_steps_per_run or 'full'}, "
                                    f"dof_weight_p={args.dof_weight_p}, dof_weight_r={args.dof_weight_r}"),
            "vehicle_spec_version": loaded_spec.raw.get("schema_version", "unknown"),
            "dataset_path": args.data_dir,
            "dataset_file_count": len(train_runs),
            "seed": config.seed,
            "sampling_rate_hz": args.sampling_rate_hz,
            "max_nfev": args.max_nfev,
        },
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
    main()