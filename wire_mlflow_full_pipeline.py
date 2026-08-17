with open("calibration/run_full_pipeline.py", encoding="utf-8") as f:
    content = f.read()

replacements = []

# 1. Import the adapter, schema constants, and reuse the git_commit helper
old1 = '''from data_split import discover_csv_files, split_by_date
    from objective import DatasetConfig, load_run, mean_run_rmse
    from optimizer import load_vehicle_spec, initial_parameters, CalibrationConfig as _CC, calibrate
    import yaml as _yaml'''
new1 = '''from data_split import discover_csv_files, split_by_date
    from objective import DatasetConfig, load_run, mean_run_rmse
    from optimizer import load_vehicle_spec, initial_parameters, CalibrationConfig as _CC, calibrate
    import yaml as _yaml
    from mlflow_logging import log_calibration_run
    from mlflow_schema import STATUS_SUCCESS, STATUS_CHECKPOINT_FALLBACK, STATUS_FAILED
    from run_calibration import _get_git_commit'''
replacements.append(("import mlflow adapter", old1, new1))

# 2. The genuine FAILED path - no usable point found at all, before any
# artifacts exist, so artifacts=[] here (matching run_calibration.py's
# own FAILED-path treatment, which also has no real artifacts by then).
old2 = '''    except RuntimeError as e:
        if _best["param_vec"] is None:
            print(f"\\nPIPELINE FAILED at calibration step (no usable point found at all): {e}")
            sys.exit(1)'''
new2 = '''    except RuntimeError as e:
        if _best["param_vec"] is None:
            print(f"\\nPIPELINE FAILED at calibration step (no usable point found at all): {e}")
            log_calibration_run(
                script="run_full_pipeline",
                run_name=f"run_full_pipeline_{time.strftime('%Y-%m-%d_%H%M')}",
                status=STATUS_FAILED, status_reason=str(e),
                git_commit=_get_git_commit(), vehicle_spec_path=args.spec,
                tags={"run_type": "demo", "task": args.task},
            )
            sys.exit(1)'''
replacements.append(("wire logging into genuine FAILED path", old2, new2))

# 3. SUCCESS/CHECKPOINT_FALLBACK - after summary_path is genuinely saved
# (same real lesson from Phase E's caught bug - never reference an
# artifact path before the file actually exists on disk).
old3 = '''    summary_path = Path(args.output_dir) / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print(f"DONE in {total_elapsed:.1f}s. Summary written to {summary_path}")'''
new3 = '''    summary_path = Path(args.output_dir) / "pipeline_summary.json"
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
    print(f"DONE in {total_elapsed:.1f}s. Summary written to {summary_path}")'''
replacements.append(("wire logging into SUCCESS/CHECKPOINT_FALLBACK path", old3, new3))

missing = [label for label, old, new in replacements if content.count(old) != 1]
if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}")
    for label, old, new in replacements:
        print(f"  {label}: {content.count(old)} occurrence(s)")
else:
    for label, old, new in replacements:
        content = content.replace(old, new, 1)
    with open("calibration/run_full_pipeline.py", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"All {len(replacements)} changes applied successfully.")