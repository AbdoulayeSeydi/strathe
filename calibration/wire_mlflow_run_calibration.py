with open("run_calibration.py", encoding="utf-8") as f:
    content = f.read()

replacements = []

# 1. Import the adapter + schema constants
old1 = '''from optimizer import (
    LoadedVehicleSpec, load_vehicle_spec, initial_parameters,
    CalibrationConfig, calibrate, calibrate_parallel,
)'''
new1 = '''from optimizer import (
    LoadedVehicleSpec, load_vehicle_spec, initial_parameters,
    CalibrationConfig, calibrate, calibrate_parallel,
)
from mlflow_logging import log_calibration_run
from mlflow_schema import STATUS_SUCCESS, STATUS_FAILED'''
replacements.append(("import mlflow adapter", old1, new1))

# 2. Wire logging into main() - success path (after report is computed,
# before ConsoleReporter prints it) AND failure path (existing except block)
old2 = '''    try:
        report = run_pipeline(pipeline_config)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)

    ConsoleReporter().report(report)
    sys.exit(0 if report.success else 1)'''
new2 = '''    try:
        report = run_pipeline(pipeline_config)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        log_calibration_run(
            script="run_calibration", run_name=f"run_calibration_{_time.strftime('%Y-%m-%d_%H%M')}",
            status=STATUS_FAILED, status_reason=str(e),
            git_commit=_get_git_commit(), vehicle_spec_path=args.spec,
            tags={"run_type": "real"},
        )
        sys.exit(1)

    _dataset_dir = Path(args.data_dir)
    _dataset_files = list(_dataset_dir.rglob("*.csv")) if _dataset_dir.exists() else []
    _dataset_total_size = sum(f.stat().st_size for f in _dataset_files)
    _dataset_modified = max((f.stat().st_mtime for f in _dataset_files), default=0.0)

    log_calibration_run(
        script="run_calibration",
        run_name=f"run_calibration_{report.reproducibility.timestamp_utc[:16].replace(':', '').replace('T', '_')}",
        status=STATUS_SUCCESS,
        status_reason=f"optimizer converged, cost={report.optimizer.cost:.6g}",
        git_commit=report.reproducibility.git_commit,
        vehicle_spec_path=report.reproducibility.spec_path,
        params={
            "optimizer": pipeline_config.calibration_config.method,
            "optimizer_settings": (f"loss={pipeline_config.calibration_config.loss}, "
                                    f"n_restarts={pipeline_config.calibration_config.n_restarts}, "
                                    f"seed={pipeline_config.calibration_config.seed}"),
            "vehicle_spec_version": report.reproducibility.spec_version,
            "dataset_path": report.reproducibility.data_dir,
            "dataset_file_count": len(_dataset_files),
            "dataset_total_size_bytes": _dataset_total_size,
            "dataset_modified_time": _dataset_modified,
            "seed": pipeline_config.calibration_config.seed,
            "sampling_rate_hz": args.sampling_rate_hz,
            "max_nfev": args.max_nfev,
        },
        metrics={
            "rmse_total": report.rmse.calibrated_heldout,
            "rmse_calibrated_train": report.rmse.calibrated_train,
            "rmse_default_heldout": report.rmse.default_heldout,
            "rmse_default_train": report.rmse.default_train,
            "rmse_u": float(report.per_dof.calibrated_heldout[0]),
            "rmse_v": float(report.per_dof.calibrated_heldout[1]),
            "rmse_w": float(report.per_dof.calibrated_heldout[2]),
            "rmse_p": float(report.per_dof.calibrated_heldout[3]),
            "rmse_q": float(report.per_dof.calibrated_heldout[4]),
            "rmse_r": float(report.per_dof.calibrated_heldout[5]),
            "runtime_seconds": report.timing.total_s,
            "nfev": report.optimizer.nfev,
        },
        artifacts=[args.spec,
                   str(Path(args.output_dir) / "calibrated_parameters.yaml"),
                   str(Path(args.output_dir) / "metrics.json")],
        tags={"run_type": "real"},
    )

    ConsoleReporter().report(report)
    sys.exit(0 if report.success else 1)'''
replacements.append(("wire logging into main()", old2, new2))

missing = [label for label, old, new in replacements if content.count(old) != 1]
if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}")
    for label, old, new in replacements:
        print(f"  {label}: {content.count(old)} occurrence(s)")
else:
    for label, old, new in replacements:
        content = content.replace(old, new, 1)
    with open("run_calibration.py", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"All {len(replacements)} changes applied successfully.")