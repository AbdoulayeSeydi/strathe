# Safe, self-verifying patch: adds calibrate_parallel() to optimizer.py -
# a NEW, fully additive function. calibrate() itself is completely
# untouched. Confirmed against optimizer.py's real code before writing:
# Windows only supports the 'spawn' multiprocessing start method (every
# argument must be genuinely picklable, confirmed via Python's own
# official docs), and _make_fixed_params_wrapper's closure is real and
# always built whenever fixed_params is used - so this rebuilds that
# wrapping fresh inside each worker process from plain, picklable inputs
# (full_template, free_mask - both NumPy arrays) rather than shipping a
# closure across the process boundary.
#
# Deliberately, explicitly scoped to the stand-in backend only:
# config.simulate_fn (HoloOcean) and config.on_iteration (checkpointing)
# both refuse to run in parallel rather than silently attempt something
# genuinely unsafe (an unpicklable live HoloOcean environment; multiple
# processes racing to write the same checkpoint file).

with open("optimizer.py", encoding="utf-8") as f:
    content = f.read()

anchor = '''def calibrate('''

new_function = '''def _parallel_restart_worker(x0_free, lb_free, ub_free, x_scale_free,
                              full_template, free_mask, train_runs, vehicle,
                              dof_scales, normalize_by_run_length, method,
                              loss, f_scale, ftol, xtol, gtol, max_nfev):
    """
    Module-level (NOT a closure/lambda) so it's picklable under Windows'
    'spawn' multiprocessing start method - required, not a style choice.
    Rebuilds the fixed_params wrapping fresh in THIS process, rather than
    receiving a closure built elsewhere (which spawn cannot pickle).
    Deliberately does not support on_iteration or simulate_fn - see
    calibrate_parallel()'s docstring for why.
    """
    def wrapped_residual_fn(x_free, *args):
        full_x = full_template.copy()
        full_x[free_mask] = x_free
        return joint_residuals(full_x, *args)

    result = least_squares(
        fun=wrapped_residual_fn, x0=x0_free, bounds=(lb_free, ub_free),
        method=method, loss=loss, f_scale=f_scale, ftol=ftol, xtol=xtol,
        gtol=gtol, max_nfev=max_nfev, x_scale=x_scale_free, verbose=0,
        args=(train_runs, vehicle, dof_scales, normalize_by_run_length, None),
    )
    full_x0 = full_template.copy()
    full_x0[free_mask] = x0_free
    full_x = full_template.copy()
    full_x[free_mask] = result.x
    return RestartResult(
        x0=full_x0, x=full_x, cost=result.cost, success=result.success,
        status=result.status, nfev=result.nfev, message=result.message,
        raw_result=None,  # not shipped back across the process boundary
    )


def calibrate_parallel(
    loaded: LoadedVehicleSpec,
    train_runs: Sequence[RunData],
    config: Optional[CalibrationConfig] = None,
    max_workers: Optional[int] = None,
) -> CalibrationResult:
    """
    Parallel version of calibrate() for the STAND-IN backend ONLY - each
    of config.n_restarts independent fits runs in its own OS process
    (concurrent.futures.ProcessPoolExecutor), genuinely using multiple
    CPU cores rather than running restarts sequentially.

    Deliberately, explicitly refuses to run if config.simulate_fn or
    config.on_iteration is set:
      - config.simulate_fn (a real HoloOcean environment) almost
        certainly cannot be pickled across a process boundary - Windows'
        'spawn' start method requires every argument be genuinely
        picklable (confirmed via Python's own docs), and a live Unreal
        Engine-backed environment object is not a plain-data object.
      - config.on_iteration (checkpointing) would mean multiple processes
        racing to write the same checkpoint file concurrently - a real,
        confirmed race-condition risk, not addressed here.

    Use calibrate() (unchanged, still the default) for HoloOcean-backed
    or checkpointed runs. This is strictly a stand-in-backend speedup.

    max_workers defaults to os.cpu_count() (concurrent.futures' own
    default) if not given.
    """
    if config is None:
        config = CalibrationConfig()
    if config.simulate_fn is not None:
        raise ValueError(
            "calibrate_parallel() does not support config.simulate_fn (e.g. a "
            "real HoloOcean environment) - it almost certainly cannot be pickled "
            "across a process boundary. Use calibrate() for HoloOcean-backed runs."
        )
    if config.on_iteration is not None:
        raise ValueError(
            "calibrate_parallel() does not support config.on_iteration (checkpointing) "
            "- multiple processes writing the same checkpoint file concurrently is a "
            "real race condition. Use calibrate() for checkpointed runs."
        )

    vehicle = loaded.vehicle
    param_names = initial_parameters(loaded).param_names
    x0_base = config.x0_override if config.x0_override is not None else initial_parameters(loaded).to_vector()
    lb, ub = parameter_bounds(loaded)

    if config.x_scale is not None:
        x_scale = config.x_scale
    elif config.use_bounds_width_x_scale:
        x_scale = ub - lb
    else:
        x_scale = np.ones_like(x0_base)

    fixed_params = config.fixed_params or {}
    unknown = set(fixed_params) - set(param_names)
    if unknown:
        raise ValueError(f"fixed_params has unknown parameter name(s): {sorted(unknown)}")
    free_mask = np.array([name not in fixed_params for name in param_names])

    full_template = x0_base.copy()
    for name, val in fixed_params.items():
        idx = param_names.index(name)
        if not (lb[idx] <= val <= ub[idx]):
            raise ValueError(
                f"fixed_params[{name!r}]={val} is outside its own bounds "
                f"[{lb[idx]}, {ub[idx]}] in vehicle_spec.yaml."
            )
        full_template[idx] = val

    x0_base_free = full_template[free_mask]
    lb_free = lb[free_mask]
    ub_free = ub[free_mask]
    x_scale_free = x_scale[free_mask]

    rng = np.random.default_rng(config.seed)
    x0_list = [
        _initialize_x0(x0_base_free, lb_free, ub_free, rng,
                        config.initialization_strategy, config.perturbation_frac)
        for _ in range(config.n_restarts)
    ]

    from concurrent.futures import ProcessPoolExecutor
    start_time = time.monotonic()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _parallel_restart_worker, x0_free, lb_free, ub_free, x_scale_free,
                full_template, free_mask, train_runs, vehicle, config.dof_scales,
                config.normalize_by_run_length, config.method, config.loss,
                config.f_scale, config.ftol, config.xtol, config.gtol, config.max_nfev,
            )
            for x0_free in x0_list
        ]
        restarts = [f.result() for f in futures]
    runtime = time.monotonic() - start_time

    converged = [r for r in restarts if r.success]
    if not converged:
        raise RuntimeError(
            f"All {config.n_restarts} calibration restart(s) failed to converge. "
            f"Messages: {[r.message for r in restarts]}. Check vehicle_spec.yaml's "
            f"bounds, the dataset's sampling rate, or whether the training data "
            f"actually constrains these parameters (see the identifiability note "
            f"in vehicle_spec.yaml)."
        )
    best = min(converged, key=lambda r: r.cost)
    # Inlined rather than calling a guessed function name - matches the
    # mean/std-across-converged-x pattern already visible in optimizer.py,
    # without assuming an unconfirmed private function name exists.
    stability = None
    if len(converged) > 1:
        xs = np.array([r.x for r in converged])
        stability = {"mean": xs.mean(axis=0), "std": xs.std(axis=0),
                     "n_converged": len(converged), "n_restarts": config.n_restarts}

    return CalibrationResult(
        parameters=CalibrationParameters.from_vector(best.x), cost=best.cost,
        nfev=best.nfev, converged=True, restarts=restarts, stability=stability,
        n_restarts=config.n_restarts, n_converged=len(converged),
        best_cost=best.cost, best_nfev=best.nfev, runtime_seconds=runtime,
    )


'''

count = content.count(anchor)
if count != 1:
    print(f"REFUSING TO PATCH - expected exactly 1 occurrence of the anchor, found {count}.")
else:
    content = content.replace(anchor, new_function + anchor, 1)
    with open("optimizer.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("calibrate_parallel() added successfully - calibrate() itself untouched.")