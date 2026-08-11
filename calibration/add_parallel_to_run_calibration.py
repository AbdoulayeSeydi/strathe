with open("run_calibration.py", encoding="utf-8") as f:
    content = f.read()

replacements = []

# 1. Import calibrate_parallel alongside calibrate
old1 = '''from optimizer import (
    LoadedVehicleSpec, load_vehicle_spec, initial_parameters,
    CalibrationConfig, calibrate,
)'''
new1 = '''from optimizer import (
    LoadedVehicleSpec, load_vehicle_spec, initial_parameters,
    CalibrationConfig, calibrate, calibrate_parallel,
)'''
replacements.append(("import calibrate_parallel", old1, new1))

# 2. PipelineConfig gains parallel + max_workers fields
old2 = '''@dataclass
class PipelineConfig:
    data_dir: str
    spec_path: str
    output_dir: str
    sampling_rate_hz: float
    heldout_dates: Optional[list]
    heldout_fraction: float
    calibration_config: CalibrationConfig
    workers: int = 4'''
new2 = '''@dataclass
class PipelineConfig:
    data_dir: str
    spec_path: str
    output_dir: str
    sampling_rate_hz: float
    heldout_dates: Optional[list]
    heldout_fraction: float
    calibration_config: CalibrationConfig
    workers: int = 4
    parallel: bool = False  # if True, restarts run across OS processes via
        # calibrate_parallel() instead of sequentially - stand-in backend
        # only. NOTE: checkpointing (on_iteration) is NOT supported in
        # parallel mode - calibrate_parallel() explicitly refuses it
        # (multiple processes writing one checkpoint file is a real race
        # condition) - main() skips setting it when parallel=True.
    max_workers: Optional[int] = None  # None = os.cpu_count() default'''
replacements.append(("PipelineConfig fields", old2, new2))

# 3. run_pipeline(): conditional calibrate call
old3 = '''    t_cal = time.monotonic()
    cal_result = calibrate(loaded_spec, train_runs, config.calibration_config)
    calibration_s = time.monotonic() - t_cal'''
new3 = '''    t_cal = time.monotonic()
    if config.parallel:
        cal_result = calibrate_parallel(loaded_spec, train_runs, config.calibration_config,
                                         max_workers=config.max_workers)
    else:
        cal_result = calibrate(loaded_spec, train_runs, config.calibration_config)
    calibration_s = time.monotonic() - t_cal'''
replacements.append(("conditional calibrate call", old3, new3))

# 4. run_pipeline(): OptimizerSummary must handle scipy_result=None (parallel case)
old4 = '''    scipy_result = cal_result.scipy_result
    optimizer_summary = OptimizerSummary(
        converged=cal_result.converged,
        cost=cal_result.cost,
        nfev=scipy_result.nfev,
        njev=getattr(scipy_result, "njev", None),
        status=scipy_result.status,
        message=scipy_result.message,
        optimality=float(scipy_result.optimality),
        n_restarts=config.calibration_config.n_restarts,
        n_converged=cal_result.stability["n_converged"],
        convergence_rate=cal_result.stability["convergence_rate"],
    )'''
new4 = '''    scipy_result = cal_result.scipy_result
    if scipy_result is not None:
        optimizer_summary = OptimizerSummary(
            converged=cal_result.converged,
            cost=cal_result.cost,
            nfev=scipy_result.nfev,
            njev=getattr(scipy_result, "njev", None),
            status=scipy_result.status,
            message=scipy_result.message,
            optimality=float(scipy_result.optimality),
            n_restarts=config.calibration_config.n_restarts,
            n_converged=cal_result.stability["n_converged"],
            convergence_rate=cal_result.stability["convergence_rate"],
        )
    else:
        # calibrate_parallel() (config.parallel=True) never populates
        # scipy_result - it deliberately doesn't ship the full scipy
        # OptimizeResult back across the process boundary. Fall back to
        # the best individual restart's own stored fields instead of
        # crashing on scipy_result.nfev. optimality has no substitute
        # (RestartResult doesn't store it) - NaN, not silently omitted.
        best_restart = min((r for r in cal_result.restarts if r.success), key=lambda r: r.cost)
        optimizer_summary = OptimizerSummary(
            converged=cal_result.converged,
            cost=cal_result.cost,
            nfev=cal_result.n_iterations,
            njev=None,
            status=best_restart.status,
            message=best_restart.message,
            optimality=float("nan"),
            n_restarts=config.calibration_config.n_restarts,
            n_converged=cal_result.stability["n_converged"],
            convergence_rate=cal_result.stability["convergence_rate"],
        )'''
replacements.append(("OptimizerSummary None-safe fallback", old4, new4))

# 5. CLI: add --parallel and --max-workers
old5 = '''    parser.add_argument("--workers", type=int, default=4)'''
new5 = '''    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--parallel", action="store_true",
                         help="Run calibration restarts across multiple OS processes "
                              "(genuine multi-core speedup) instead of sequentially. "
                              "Stand-in backend ONLY - not supported for HoloOcean-backed "
                              "runs. NOTE: checkpointing is disabled when this is set - "
                              "calibrate_parallel() cannot safely checkpoint across "
                              "multiple processes writing the same file. Only meaningful "
                              "with --n-restarts > 1.")
    parser.add_argument("--max-workers", type=int, default=None,
                         help="Max worker processes for --parallel. Defaults to "
                              "os.cpu_count() if not given.")'''
replacements.append(("CLI flags", old5, new5))

missing = [label for label, old, new in replacements if content.count(old) != 1]
if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}.")
else:
    for label, old, new in replacements:
        content = content.replace(old, new, 1)
    with open("run_calibration.py", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"All {len(replacements)} changes applied successfully.")

# --- Part 2: main()'s conditional checkpoint-skip and PipelineConfig wiring ---
with open("run_calibration.py", encoding="utf-8") as f:
    content = f.read()

old6 = '''    calibration_config = CalibrationConfig(
        loss=args.loss, n_restarts=args.n_restarts, seed=args.seed, max_nfev=args.max_nfev,
        initialization_strategy=args.initialization_strategy,
        perturbation_frac=args.perturbation_frac,
        verbose=2 if args.verbose else 0,  # scipy's own per-iteration progress
        # (cost, optimality, step norm each iteration) — previously --verbose
        # only affected Python logging.basicConfig, never scipy's own
        # verbose=, so runs gave zero visibility into whether they were
        # progressing or stuck. This is what actually answers that question.
        x0_override=x0_override,
        on_iteration=make_checkpoint_callback(checkpoint_path),
    )
    pipeline_config = PipelineConfig(
        data_dir=args.data_dir,
        spec_path=args.spec,
        output_dir=args.output_dir,
        sampling_rate_hz=args.sampling_rate_hz,
        heldout_dates=args.heldout_dates,
        heldout_fraction=args.heldout_fraction,
        calibration_config=calibration_config,
        workers=args.workers,
    )'''

new6 = '''    if args.parallel:
        print("--parallel is set: checkpointing DISABLED for this run - "
              "calibrate_parallel() cannot safely checkpoint across multiple "
              "processes writing the same file. An interrupted parallel run "
              "cannot be resumed; re-run from scratch if needed.")
    calibration_config = CalibrationConfig(
        loss=args.loss, n_restarts=args.n_restarts, seed=args.seed, max_nfev=args.max_nfev,
        initialization_strategy=args.initialization_strategy,
        perturbation_frac=args.perturbation_frac,
        verbose=2 if args.verbose else 0,
        x0_override=x0_override,
        on_iteration=None if args.parallel else make_checkpoint_callback(checkpoint_path),
    )
    pipeline_config = PipelineConfig(
        data_dir=args.data_dir,
        spec_path=args.spec,
        output_dir=args.output_dir,
        sampling_rate_hz=args.sampling_rate_hz,
        heldout_dates=args.heldout_dates,
        heldout_fraction=args.heldout_fraction,
        calibration_config=calibration_config,
        workers=args.workers,
        parallel=args.parallel,
        max_workers=args.max_workers,
    )'''

count = content.count(old6)
if count != 1:
    print(f"REFUSING TO PATCH part 2 - expected exactly 1, found {count}.")
else:
    content = content.replace(old6, new6, 1)
    with open("run_calibration.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Part 2 applied successfully - all 6 changes now complete.")