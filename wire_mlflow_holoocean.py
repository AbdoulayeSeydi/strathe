with open("calibration/run_calibration_holoocean.py", encoding="utf-8") as f:
    content = f.read()

replacements = []

# 1. Import the adapter, schema constants, and reuse run_calibration.py's
# existing _get_git_commit() rather than duplicating it.
old1 = '''from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, mean_run_rmse, rmse_by_dof
from optimizer import load_vehicle_spec, parameter_bounds, calibrate, CalibrationConfig
from dynamics_model import CalibrationParameters
from holoocean_backend import make_holoocean_simulate_fn, make_deterministic_environment'''
new1 = '''from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, mean_run_rmse, rmse_by_dof
from optimizer import load_vehicle_spec, parameter_bounds, calibrate, CalibrationConfig
from dynamics_model import CalibrationParameters
from holoocean_backend import make_holoocean_simulate_fn, make_deterministic_environment
from mlflow_logging import log_calibration_run
from mlflow_schema import STATUS_SUCCESS, STATUS_CHECKPOINT_FALLBACK
from run_calibration import _get_git_commit'''
replacements.append(("import mlflow adapter + reuse git_commit helper", old1, new1))

# 2. Wire logging in at the very END of main() - AFTER artifacts are
# genuinely saved to disk (real bug caught before shipping: an earlier
# draft of this patch placed the logging call BEFORE out.mkdir()/the
# actual file writes, meaning the artifact paths passed to
# log_calibration_run() would not have existed yet).
old2 = '''    with open(out / "metrics.json", "w", encoding="utf-8") as f:
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
    print(f"\\nSaved results to {out}/")'''
new2 = '''    with open(out / "metrics.json", "w", encoding="utf-8") as f:
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
    print(f"\\nSaved results to {out}/")

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
    )'''
replacements.append(("wire logging after artifacts genuinely saved", old2, new2))

missing = [label for label, old, new in replacements if content.count(old) != 1]
if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}")
    for label, old, new in replacements:
        print(f"  {label}: {content.count(old)} occurrence(s)")
else:
    for label, old, new in replacements:
        content = content.replace(old, new, 1)
    with open("calibration/run_calibration_holoocean.py", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"All {len(replacements)} changes applied successfully.")