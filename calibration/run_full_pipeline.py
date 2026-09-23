#!/usr/bin/env python3
"""
Phase 3 — the single-command demo pipeline: spec file + log folder in,
a ready-to-use Gymnasium environment + validation report out. Runnable
end-to-end without manual intervention partway through.

Uses the same proven lower-level functions (discover_csv_files,
split_by_date, load_run, calibrate) already exercised throughout tonight's
testing — NOT run_calibration.py's run_pipeline(), because that function
always uses the full dataset with no truncation option, which is exactly
what makes a demo command unsafe given the still-open performance
mystery. Adds: dataset truncation (--max-train-runs) as a REAL,
load-bearing safety mechanism (not just documented and unused), a
Gym-environment smoke test, and one combined summary report.

SCOPE, stated explicitly (see README.md / PROJECT_STATUS.md for why):
  - Uses the FAST STAND-IN calibration, not the (7-29 hour, less
    reliable) HoloOcean refinement. This command is designed to be
    watchable in a live demo, not run overnight.
  - The validation numbers here are a QUICK check on the fast stand-in
    backend, NOT the rigorous real-HoloOcean Claim A methodology
    documented in PROJECT_STATUS.md. Labeled clearly as such in the
    output - do not quote these numbers as the project's headline
    result without checking PROJECT_STATUS.md for the real, slower,
    HoloOcean-validated numbers.

Usage:
    python run_full_pipeline.py \
        --spec vehicle_spec.yaml \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --output-dir ../demo_pipeline_output \
        --sampling-rate-hz 10.0
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to vehicle_spec.yaml")
    parser.add_argument("--data-dir", required=True, help="Folder of real logged mission data")
    parser.add_argument("--output-dir", required=True, help="Where all outputs get written")
    parser.add_argument(
        "--sampling-rate-hz", type=float, default=10.0,
        help="Real sampling rate of the logged data. NOT auto-detected in this version — "
             "defaults to 10.0 (this project's own confirmed value for the reference "
             "dataset; see PROJECT_STATUS.md finding #11 for the validated method used to "
             "determine it). Override this if using a different dataset with a known, "
             "different rate.",
    )
    parser.add_argument("--max-nfev", type=int, default=15,
                         help="Evaluation budget for the calibration fit. Kept SMALL by "
                              "default (15) specifically for this demo script, given a "
                              "confirmed real precedent: --max-nfev 60 on this same "
                              "backend/dataset took ~6.5 hours (the displayed nfev "
                              "undercounts real cost ~21x for a 20+ parameter model - see "
                              "PROJECT_STATUS.md finding #14). Raise this for a more fully "
                              "converged result once you've confirmed the pipeline works "
                              "end-to-end at this fast, watchable setting first.")
    parser.add_argument("--max-train-runs", type=int, default=5,
                         help="Truncates to this many TRAINING runs (not steps) for the "
                              "demo fit, default 5. A single evaluation on the FULL real "
                              "dataset has been observed to occasionally take multiple "
                              "HOURS for unexplained reasons (PROJECT_STATUS.md's "
                              "still-open performance mystery) - this sidesteps that "
                              "entirely for demo purposes rather than waiting it out. Pass "
                              "a large number (e.g. 1000) to use the full dataset once the "
                              "mystery is understood or you have time to spare.")
    parser.add_argument("--skip-preflight", action="store_true",
                         help="Skip the automated data-quality pre-flight check (item #6 - "
                              "sampling-rate validation, Coriolis-excitation check). On by "
                              "default so real data-quality problems are caught before "
                              "committing time to a calibration that can't succeed. Skip "
                              "only if you've already verified this exact dataset recently.")
    parser.add_argument("--task", default="waypoint",
                         choices=["waypoint", "station_keeping", "docking", "obstacle_avoidance"],
                         help="Which Arena task the delivered Gym environment is built for "
                              "(item #12). 'waypoint' (default) - reach a target point once. "
                              "'station_keeping' - stay near the target for sustained steps, "
                              "not just touch it. 'docking' - reach the target aligned "
                              "(orientation) and slow (low speed) simultaneously. "
                              "'obstacle_avoidance' - route around a randomized obstacle to "
                              "reach a randomized goal; this task has a LARGER observation "
                              "space (15 dims, not 11) - see PROJECT_STATUS.md.")
    parser.add_argument("--heldout-dates", nargs="*",
                         default=["16-02-2024", "19-02-2024"],
                         help="Which collection dates to hold out for validation. Defaults "
                              "to Phase 1's EXACT established split (PROJECT_STATUS.md) - "
                              "the same dates every other script in this project uses - so "
                              "this demo's numbers are genuinely comparable, not just "
                              "'a demo that happens to run.' Override if pointing this at a "
                              "genuinely different dataset.")
    return parser


def smoke_test_gym_environment(calibrated_params_path: str, spec_path: str, task: str = "waypoint") -> dict:
    """
    Confirms the calibrated parameters actually produce a working,
    importable, steppable Gymnasium environment — not just that a YAML
    file exists. Returns a small dict describing what was checked, for
    the final report.
    """
    result = {"attempted": True, "success": False, "detail": None}
    try:
        import yaml
        from dynamics_model import CalibrationParameters
        from optimizer import load_vehicle_spec
        from arena import Arena

        with open(calibrated_params_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        params = CalibrationParameters(**data["parameters"])
        vehicle = load_vehicle_spec(spec_path).vehicle  # same real vehicle constants
        # used throughout calibration, not a hardcoded guess

        # Best-effort construction - Arena's exact constructor signature
        # should be verified against arena.py directly if this fails;
        # this smoke test is deliberately defensive rather than assuming.
        env = Arena(params=params, vehicle=vehicle, task=task)
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        result["success"] = True
        result["detail"] = f"Environment reset and stepped successfully. obs shape: {getattr(obs, 'shape', type(obs))}"
    except Exception as e:
        result["detail"] = f"Smoke test failed: {type(e).__name__}: {e}"
    return result


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    t_start = time.monotonic()

    print("=" * 70)
    print("PHASE 3 DEMO PIPELINE — spec + logs -> calibrated Gym environment")
    print("=" * 70)
    print(f"Spec:      {args.spec}")
    print(f"Data dir:  {args.data_dir}")
    print(f"Output:    {args.output_dir}")
    print(f"Sampling rate: {args.sampling_rate_hz} Hz (NOT auto-detected - see --help)")
    print()

    if not args.skip_preflight:
        print("PRE-FLIGHT: Running automated data-quality checks (item #6) before "
              "committing to a real calibration run...")
        preflight_result = subprocess.run(
            [sys.executable, "preflight_check.py",
             "--data-dir", args.data_dir, "--spec", args.spec],
        )
        if preflight_result.returncode != 0:
            print("\nPRE-FLIGHT FAILED — stopping before spending time on a calibration "
                  "run that likely can't succeed. Review the output above, or pass "
                  "--skip-preflight to proceed anyway (not recommended without "
                  "understanding why it failed).")
            sys.exit(1)
        print("PRE-FLIGHT PASSED — proceeding with calibration.\n")
    else:
        print("PRE-FLIGHT SKIPPED (--skip-preflight passed) — proceeding without "
              "automated data-quality validation.\n")

    print("STEP 1/2: Fitting calibration (fast stand-in backend)...")
    print(f"NOTE: truncated to {args.max_train_runs} training run(s) for demo speed - this "
          f"is what actually protects against the multi-hour full-dataset mystery, not just "
          f"a small --max-nfev (confirmed real risk tonight: a single full-dataset "
          f"evaluation has taken up to ~4 hours for unexplained reasons - see "
          f"PROJECT_STATUS.md's open performance-mystery note).")

    from data_split import discover_csv_files, split_by_date
    from objective import DatasetConfig, load_run, mean_run_rmse
    from optimizer import load_vehicle_spec, initial_parameters, CalibrationConfig as _CC, calibrate
    import yaml as _yaml
    from mlflow_logging import log_calibration_run
    from mlflow_schema import STATUS_SUCCESS, STATUS_CHECKPOINT_FALLBACK, STATUS_FAILED
    from run_calibration import _get_git_commit

    csv_paths, dup = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files[:args.max_train_runs]]
    heldout_runs = [load_run(fp, dataset_config) for fp in split.heldout_files]
    print(f"Loaded {len(train_runs)} training run(s) (truncated), {len(heldout_runs)} held-out run(s)")

    loaded_spec = load_vehicle_spec(args.spec)
    vehicle = loaded_spec.vehicle
    default_params = initial_parameters(loaded_spec)
    rmse_train_default = mean_run_rmse(default_params, train_runs, vehicle)
    rmse_heldout_default = mean_run_rmse(default_params, heldout_runs, vehicle)

    calibration_config = _CC(
        loss="soft_l1", n_restarts=1, seed=42, max_nfev=args.max_nfev,
        initialization_strategy="perturbed", verbose=2,
    )

    # Track the best (lowest-cost) point seen, same proven pattern as
    # run_calibration_holoocean.py's checkpoint callback - because
    # calibrate() raises RuntimeError (with NO usable result) if strict
    # scipy convergence isn't reached within max_nfev, and hitting that
    # budget without full convergence is EXPECTED and NOT a real failure
    # for a small demo budget (confirmed: --max-nfev 5 genuinely can't
    # reach strict convergence, but still makes real, useful progress -
    # see the cost trajectory printed above). Without this, the demo
    # would falsely "fail" on its own normal, expected behavior.
    from dynamics_model import CalibrationParameters as _CP
    _best = {"cost": float("inf"), "param_vec": None}

    def _track_best(call_count, param_vec, cost):
        if cost < _best["cost"]:
            _best["cost"] = cost
            _best["param_vec"] = param_vec.copy()
    calibration_config.on_iteration = _track_best
    # the ~4 h single-evaluation stall was last seen in THIS script -- time
    # every evaluation and capture stacks if it happens again (stall_watchdog.py)
    from stall_watchdog import StallWatchdog
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    calibration_config.stall_watchdog = StallWatchdog(args.output_dir)

    try:
        result = calibrate(loaded_spec, train_runs, calibration_config)
        best_params = result.parameters
        best_cost = result.cost
        converged = result.converged
    except RuntimeError as e:
        if _best["param_vec"] is None:
            print(f"\nPIPELINE FAILED at calibration step (no usable point found at all): {e}")
            log_calibration_run(
                script="run_full_pipeline",
                run_name=f"run_full_pipeline_{time.strftime('%Y-%m-%d_%H%M')}",
                status=STATUS_FAILED, status_reason=str(e),
                git_commit=_get_git_commit(), vehicle_spec_path=args.spec,
                tags={"run_type": "demo", "task": args.task},
            )
            sys.exit(1)
        print(f"\nDid not reach strict convergence within max_nfev={args.max_nfev} - this is "
              f"EXPECTED for a small demo budget, not a failure. Using the best point found "
              f"({_best['cost']:.6f}) instead. Raise --max-nfev for a more converged result.")
        best_params = _CP.from_vector(_best["param_vec"])
        best_cost = _best["cost"]
        converged = False

    rmse_train_calibrated = mean_run_rmse(best_params, train_runs, vehicle)
    rmse_heldout_calibrated = mean_run_rmse(best_params, heldout_runs, vehicle)
    improvement_heldout_pct_ = (
        (rmse_heldout_default - rmse_heldout_calibrated) / rmse_heldout_default * 100
    )

    print(f"\nConverged: {converged}, cost: {best_cost:.6f}")
    print(f"RMSE (train):   default={rmse_train_default:.4f}  calibrated={rmse_train_calibrated:.4f}")
    print(f"RMSE (heldout): default={rmse_heldout_default:.4f}  calibrated={rmse_heldout_calibrated:.4f}")
    print(f"Held-out improvement: {improvement_heldout_pct_:+.1f}%")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    calibrated_params_path = str(Path(args.output_dir) / "calibrated_parameters.yaml")
    params_dict = {n: float(getattr(best_params, n)) for n in best_params.param_names}
    with open(calibrated_params_path, "w", encoding="utf-8") as f:
        _yaml.dump({"parameters": params_dict}, f, sort_keys=False)

    class _Report:
        success = True  # we have a usable result either way at this point in the
        # code (the "no usable point at all" case already exited above) -
        # strict scipy convergence is a separate, honestly-reported detail
        # below, not what determines whether this demo run "worked"
        strict_converged = converged  # renamed to avoid a class-body self-reference NameError
        class rmse:
            default_train = rmse_train_default
            calibrated_train = rmse_train_calibrated
            default_heldout = rmse_heldout_default
            calibrated_heldout = rmse_heldout_calibrated
            improvement_heldout_pct = improvement_heldout_pct_
    report = _Report()

    print()
    print("STEP 2/2: Verifying the calibrated Gym environment actually works...")
    calibrated_params_path = str(Path(args.output_dir) / "calibrated_parameters.yaml")
    gym_check = smoke_test_gym_environment(calibrated_params_path, args.spec, task=args.task)
    if gym_check["success"]:
        print(f"  PASS — {gym_check['detail']}")
    else:
        print(f"  FAIL — {gym_check['detail']}")
        print("  (Calibration itself succeeded; only the Gym-environment smoke test failed. "
              "Check arena.py's Arena() constructor signature against this script's call.)")

    total_elapsed = time.monotonic() - t_start

    summary = {
        "pipeline_succeeded": report.success,
        "scipy_strict_convergence": report.strict_converged,
        "rmse_train_default": report.rmse.default_train,
        "rmse_train_calibrated": report.rmse.calibrated_train,
        "rmse_heldout_default": report.rmse.default_heldout,
        "rmse_heldout_calibrated": report.rmse.calibrated_heldout,
        "improvement_heldout_pct": report.rmse.improvement_heldout_pct,
        "gym_environment_check": gym_check,
        "total_elapsed_seconds": total_elapsed,
        "NOTE": "Validation numbers above are from the FAST STAND-IN backend, a quick check "
                "only - NOT the rigorous real-HoloOcean Claim A methodology in "
                "PROJECT_STATUS.md. Do not quote as the project's headline result.",
    }
    summary_path = Path(args.output_dir) / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if report.strict_converged:
        _status = STATUS_SUCCESS
        _status_reason = f"scipy converged, cost={best_cost:.6f}"
    else:
        _status = STATUS_CHECKPOINT_FALLBACK
        _status_reason = f"maximum function evaluations reached, best point substituted (cost={best_cost:.6f})"

    log_calibration_run(
        script="run_full_pipeline",
        run_name=f"run_full_pipeline_{time.strftime('%Y-%m-%d_%H%M')}",
        status=_status, status_reason=_status_reason,
        git_commit=_get_git_commit(), vehicle_spec_path=args.spec,
        params={
            "optimizer": "scipy.optimize.least_squares (trf)",
            "optimizer_settings": "loss=soft_l1, n_restarts=1, seed=42",
            "vehicle_spec_version": loaded_spec.raw.get("schema_version", "unknown"),
            "dataset_path": args.data_dir,
            "dataset_file_count": len(train_runs) + len(heldout_runs),
            "sampling_rate_hz": args.sampling_rate_hz,
            "max_nfev": args.max_nfev,
            "task": args.task,
            "max_train_runs": args.max_train_runs,
        },
        # NOTE: nfev deliberately omitted - this script never captures a
        # real call count anywhere (neither the try block nor the
        # fallback _best dict tracks it) - honestly omitted rather than
        # fabricated, see PROJECT_STATUS.md.
        metrics={
            "rmse_total": report.rmse.calibrated_heldout,
            "rmse_calibrated_train": report.rmse.calibrated_train,
            "rmse_default_heldout": report.rmse.default_heldout,
            "rmse_default_train": report.rmse.default_train,
            "runtime_seconds": total_elapsed,
        },
        artifacts=[args.spec, calibrated_params_path, str(summary_path)],
        tags={"run_type": "demo", "task": args.task,
              "gym_check_passed": str(gym_check["success"])},
    )

    print()
    print("=" * 70)
    print(f"DONE in {total_elapsed:.1f}s. Summary written to {summary_path}")
    print()
    print("Optional next step (item #17, NOT run automatically - real cost ~15-17 min on")
    print("the full dataset, would slow down this demo): a deeper, automated parameter")
    print("identifiability report (condition number, correlated pairs, multi-parameter")
    print("flat directions) is available via:")
    print(f"  python automated_identifiability_report.py --data-dir {args.data_dir} \\")
    print(f"    --spec {args.spec} --sampling-rate-hz {args.sampling_rate_hz} \\")
    print(f"    --calibrated-params {calibrated_params_path}")
    print("=" * 70)
    sys.exit(0 if (report.success and gym_check["success"]) else 1)


if __name__ == "__main__":
    main()