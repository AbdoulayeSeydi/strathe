"""
Wraps scipy.optimize.least_squares around objective.py's joint_residuals().

## No duplicated schema
Previously this file had its own PARAM_NAMES + SECTION_FOR_PARAM list,
manually kept in sync with vehicle_spec.yaml AND dynamics_model.py's
CalibrationParameters. Three sources of truth that could silently drift.
Now there are exactly two, each authoritative for a different thing:
  - vehicle_spec.yaml supplies VALUES (initial guesses, bounds)
  - CalibrationParameters (dynamics_model.py) supplies the SCHEMA (which
    parameters must exist, and their vector ordering via to_vector())
This file only flattens the YAML's nested sections and cross-checks the
result against CalibrationParameters' actual dataclass fields — if they
disagree, construction fails immediately and specifically, rather than a
bare KeyError deep inside an optimizer call.

## Explicit optimizer configuration
Nothing here relies on scipy's defaults for anything that affects results
(method, loss, tolerances, scaling) — see CalibrationConfig. Relying on
scipy's defaults means results silently change if scipy's defaults ever
change; production calibration code shouldn't be implicitly coupled to a
dependency's version history.

## Robust loss by default
loss="soft_l1" (not scipy's default "linear"/pure-L2) — a robotics
telemetry log is exactly the kind of data that has occasional spikes
(dropped packets, a sensor glitch), and pure L2 lets one bad sample shift
the whole fit disproportionately.

## Restarts + actual convergence checking
calibrate() runs config.n_restarts independent fits from different initial
guesses and picks the best CONVERGED one — raising RuntimeError if every
restart fails to converge, rather than silently handing back whatever
scipy returned without anyone checking result.success. With
n_restarts > 1, the returned CalibrationResult.stability reports mean/std
across converged solutions and the convergence rate — actually measuring
the "consistent parameter set" claim instead of asserting it in a comment.

## Parameter scaling
Calibration targets span very different magnitudes (e.g. cg_cb_offset_z
~0.02, Zw ~-30) — least_squares converges better when variables are
similarly scaled. Default x_scale is each parameter's bounds width
(ub - lb), a scale directly tied to the physical range already specified
in vehicle_spec.yaml, rather than a generic heuristic.

## Fixed parameters (identifiability investigation support)
CalibrationConfig.fixed_params lets specific parameters be held EXACTLY
CONSTANT during optimization — removed from the search space entirely,
not just tightly bounded. Added specifically to test a real finding
(PROJECT_STATUS.md): Zw_dot pinned at its lower bound (0) after adding
Coriolis terms, with NO strong pairwise correlation found in the Jacobian
analysis — consistent with either a multi-parameter flat direction or a
genuine practical identifiability gap (this dataset's real maneuvers may
simply not excite heave acceleration enough to constrain it). Fixing
Zw_dot at a physically plausible value (e.g. its pre-Coriolis calibrated
value, ~14.6) and re-optimizing the other 19 parameters directly tests
this: if RMSE barely changes, the data genuinely doesn't constrain this
parameter; if it changes meaningfully, something else is going on.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, fields as dataclass_fields
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

import numpy as np
import yaml
from scipy.optimize import least_squares

from dynamics_model import CalibrationParameters, VehicleConstants
from objective import joint_residuals, mean_run_rmse, ParallelJointResiduals, RunData


# ---------------------------------------------------------------------------
# Loading + validating vehicle_spec.yaml
# ---------------------------------------------------------------------------

# Calibration targets a vehicle_spec.yaml may leave out: held at this value (zero-width bounds), which is the model
# before they existed. Lab specs written before #66 keep working unchanged.
OPTIONAL_CALIBRATION_TARGETS = {
    "thruster_time_constant_s": {
        "unit": "s", "initial_guess": 0.0, "bounds": [0.0, 0.0],
        "bounds_rationale": "not in this vehicle_spec.yaml: no thruster lag (add a thruster_dynamics section to fit it)",
        "source": "default", "confidence": "assumed"},
}
# torpedo-shaped hulls (#71): lift, lift moment, speed-dependent rotational damping and the x offset of the centre of
# gravity. Held at 0 unless the spec lists them (a hull_lift section; cg_cb_offset_x under buoyancy).
for _name, _unit, _what in (("cg_cb_offset_x", "m", "centre of gravity ahead of buoyancy"),
                            ("Zuw", "N/(m/s)^2", "hull lift"), ("Muw", "N*m/(m/s)^2", "hull lift moment"),
                            ("Muq", "N*m/((m/s)*(rad/s))", "speed-dependent pitch/yaw damping")):
    OPTIONAL_CALIBRATION_TARGETS[_name] = {
        "unit": _unit, "initial_guess": 0.0, "bounds": [0.0, 0.0],
        "bounds_rationale": f"not in this vehicle_spec.yaml: no {_what} (torpedo hulls: templates/vehicle_spec_template.yaml)",
        "source": "default", "confidence": "assumed"}


def held_by_bounds(lb, ub, param_names) -> dict:
    """{name: value} for parameters whose bounds have zero width: held there, never fitted (a zero-width range
    can't be scaled or stepped). OPTIONAL_CALIBRATION_TARGETS a spec leaves out are such parameters."""
    return {n: float(lb[i]) for i, n in enumerate(param_names) if ub[i] == lb[i]}


def _flatten_calibration_targets(raw_spec: dict) -> dict:
    """
    Flattens the YAML's nested calibration_targets (grouped into
    linear_damping/quadratic_damping/etc for human organization) into
    {param_name: {initial_guess, bounds, ...}}. Nothing downstream cares
    which section a parameter lives in — only its name, which must be
    unique across all sections.
    """
    flat = {}
    for section_name, section in raw_spec["calibration_targets"].items():
        for param_name, entry in section.items():
            if param_name in flat:
                raise ValueError(
                    f"Parameter {param_name!r} appears in multiple "
                    f"calibration_targets sections — names must be unique "
                    f"(clashes with section {section_name!r})."
                )
            flat[param_name] = entry
    return flat


def _validate_flat_targets(flat: dict) -> None:
    """
    Cross-checks the flattened YAML against CalibrationParameters'
    ACTUAL dataclass fields — the dataclass is the schema authority, not
    a hand-maintained list in this file.
    """
    required = {f.name for f in dataclass_fields(CalibrationParameters)}
    present = set(flat.keys())

    missing = required - present
    if missing:
        raise ValueError(
            f"vehicle_spec.yaml is missing calibration parameter(s) required "
            f"by CalibrationParameters: {sorted(missing)}"
        )
    extra = present - required
    if extra:
        raise ValueError(
            f"vehicle_spec.yaml has calibration parameter(s) not recognized "
            f"by CalibrationParameters: {sorted(extra)}. Either a typo in the "
            f"YAML, or CalibrationParameters needs a new field for these."
        )

    for name, entry in flat.items():
        lo, hi = entry["bounds"]
        if lo > hi:
            raise ValueError(f"{name}: bounds lower ({lo}) > upper ({hi})")
        guess = entry["initial_guess"]
        if not (lo <= guess <= hi):
            raise ValueError(
                f"{name}: initial_guess {guess} falls outside bounds [{lo}, {hi}]"
            )
        ref = entry.get("reference")
        if ref is not None:  # an independent measurement of this vehicle's value (calibration_report.py)
            if (not isinstance(ref, dict) or not isinstance(ref.get("value"), (int, float))
                    or isinstance(ref.get("value"), bool) or not ref["value"] or not ref.get("source")):
                raise ValueError(f"{name}: reference needs a nonzero value and a source, e.g. "
                                 f"reference: {{value: 6.3, source: \"CAD model\"}}")


@dataclass
class LoadedVehicleSpec:
    """
    Structured wrapper around the parsed YAML — not a bare dict passed
    around everywhere (that's how "spec['vehicl']"-style typos survive
    to runtime). `raw` retains the full parsed YAML for anything not yet
    promoted to a structured accessor below; vehicle_spec_schema.py has a
    stricter Pydantic-validated version of this same file for environments
    where pydantic is installed (couldn't be verified in this sandbox —
    no network access here — so this file's own validation in
    _validate_flat_targets() above is the one guaranteed to actually run).
    """
    raw: dict
    flat_calibration_targets: dict
    vehicle: VehicleConstants


def load_vehicle_spec(yaml_path: str) -> LoadedVehicleSpec:
    """Loads and validates vehicle_spec.yaml. Raises immediately and
    specifically on any missing/extra/malformed calibration parameter or
    bad bounds — before scipy ever sees any of it."""
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    flat = _flatten_calibration_targets(raw)
    for name, entry in OPTIONAL_CALIBRATION_TARGETS.items():
        flat.setdefault(name, dict(entry))
    _validate_flat_targets(flat)

    vehicle = VehicleConstants(mass_kg=raw["fixed_parameters"]["mass_kg"]["value"])
    vehicle.validate()

    return LoadedVehicleSpec(raw=raw, flat_calibration_targets=flat, vehicle=vehicle)


def initial_parameters(loaded: LoadedVehicleSpec) -> CalibrationParameters:
    """Builds CalibrationParameters directly from the YAML's initial
    guesses. Construction itself doubles as validation — a field
    CalibrationParameters requires but the YAML doesn't supply raises a
    clear TypeError here, on top of the checks already run in
    load_vehicle_spec()."""
    kwargs = {name: entry["initial_guess"] for name, entry in loaded.flat_calibration_targets.items()}
    params = CalibrationParameters(**kwargs)
    params.validate()
    return params


def parameter_bounds(loaded: LoadedVehicleSpec) -> tuple:
    """Returns (lower, upper) arrays in CalibrationParameters.to_vector()'s
    order — read from a real CalibrationParameters instance's param_names,
    never a separately-maintained ordering list."""
    order = initial_parameters(loaded).param_names
    flat = loaded.flat_calibration_targets
    lb = np.array([flat[n]["bounds"][0] for n in order])
    ub = np.array([flat[n]["bounds"][1] for n in order])
    return lb, ub


# ---------------------------------------------------------------------------
# Optimizer configuration, restarts, results
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    # scipy least_squares knobs — explicit, never left to scipy's defaults
    method: str = "trf"                     # required for bounded least_squares
    loss: str = "soft_l1"                   # robust to telemetry outliers, not pure L2
    f_scale: float = 1.0                    # residual magnitude marking the robust-loss transition
    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-10
    max_nfev: Optional[int] = None          # None = scipy's own size-based default
    use_bounds_width_x_scale: bool = True   # scale each param by (ub-lb) unless x_scale given
    x_scale: Optional[np.ndarray] = None
    verbose: int = 0                        # NOT hardcoded at the call site

    # objective knobs, passed through to joint_residuals
    dof_scales: Optional[np.ndarray] = None
    normalize_by_run_length: bool = True
    eval_workers: int = 1   # >1 = evaluate runs in that many worker processes
        # (objective.ParallelJointResiduals; identical results, stand-in physics only)
    horizon_samples: Optional[int] = None   # None = replay whole runs open-loop (original
        # objective); H = re-anchored H-sample prediction windows. See
        # objective.py "Prediction horizon" and PROJECT_STATUS.md #50.
    simulate_fn: Optional[Callable] = None  # None = the analytical stand-in (default,
        # unchanged behavior). Pass a HoloOcean-bound callable (see
        # holoocean_backend.make_holoocean_simulate_fn) to calibrate against
        # real HoloOcean physics instead — everything else about calibrate()
        # stays the same.

    # starting point
    x0_override: Optional[np.ndarray] = None  # None = use initial_parameters(loaded)
        # from vehicle_spec.yaml, as before. Pass an explicit parameter vector
        # (e.g. a previously-calibrated result) to warm-start from that instead
        # of the YAML's naive initial guess — useful when the naive guess is
        # far from the true optimum and evaluations are expensive (e.g. a
        # HoloOcean-backed run, where each evaluation costs real wall-clock
        # time, not milliseconds).

    # identifiability investigation support
    fixed_params: Optional[dict] = None  # {param_name: fixed_value} — see module
        # docstring's "Fixed parameters" section. None (default) = every
        # parameter is free, identical behavior to before this existed.
    prior_targets: Optional[dict] = None  # {param_name: published value}: drag_reference.prior_residual_wrapper
    prior_weight: float = 0.1             # adds sqrt(w) * ln(theta / target) per targeted parameter

    # reproducibility / restarts
    seed: Optional[int] = None
    initialization_strategy: str = "fixed"  # "fixed" | "perturbed" | "random"
    perturbation_frac: float = 0.1          # for "perturbed": +/- this fraction of bounds width
    n_restarts: int = 1

    # instrumentation
    on_iteration: Optional[Callable] = None  # called as on_iteration(call_count, param_vec, cost)
    log_path: Optional[str] = None           # if set, writes a JSON run log here
    stall_watchdog: Optional[object] = None  # stall_watchdog.StallWatchdog: times every evaluation
                                             # and dumps stacks if one runs far past normal


@dataclass
class RestartResult:
    x0: np.ndarray
    x: np.ndarray
    cost: float
    success: bool
    status: int
    nfev: int
    message: str
    raw_result: object  # the actual scipy OptimizeResult — has .optimality, .njev,
                         # .jac, etc. that the plain fields above don't capture


@dataclass
class CalibrationLog:
    timestamp_utc: str
    method: str
    loss: str
    n_restarts: int
    n_converged: int
    best_cost: float
    best_nfev: int
    runtime_seconds: float


@dataclass
class CalibrationResult:
    parameters: CalibrationParameters
    converged: bool
    cost: float
    n_iterations: int          # nfev of the best restart
    train_rmse: float          # mean_run_rmse on the runs calibrate() was given
    restarts: list             # list[RestartResult] — for stability inspection
    stability: dict            # {"mean", "std", "convergence_rate", "n_converged", "n_total"}
    log: CalibrationLog
    scipy_result: object       # raw best scipy OptimizeResult, full detail if needed


def _initialize_x0(x0_base, lb, ub, rng, strategy, perturbation_frac):
    if strategy == "fixed":
        return x0_base.copy()
    width = ub - lb
    if strategy == "perturbed":
        noise = rng.uniform(-perturbation_frac, perturbation_frac, size=x0_base.shape) * width
        return np.clip(x0_base + noise, lb, ub)
    if strategy == "random":
        return rng.uniform(lb, ub)
    raise ValueError(f"Unknown initialization_strategy: {strategy!r}")


def _wrap_residual_fn(base_fn: Callable, on_iteration: Optional[Callable],
                      watchdog: Optional[object] = None) -> Callable:
    """Scipy's least_squares has no native callback= — this wraps the
    residual function itself (which scipy calls every evaluation anyway)
    to report iteration count, current parameters, and cost. With a
    watchdog, every evaluation is also timed on three clocks and stack-dumped
    if it stalls (see stall_watchdog.py)."""
    if watchdog is not None:
        inner = base_fn

        def base_fn(param_vec, *args):
            with watchdog.measure("residual_evaluation"):
                return inner(param_vec, *args)
    if on_iteration is None:
        return base_fn

    call_count = {"n": 0}

    def wrapped(param_vec, *args):
        call_count["n"] += 1
        residuals = base_fn(param_vec, *args)
        cost = 0.5 * float(np.sum(residuals ** 2))  # scipy's own cost convention
        on_iteration(call_count["n"], param_vec.copy(), cost)
        return residuals

    return wrapped


def _make_fixed_params_wrapper(residual_fn: Callable, full_template: np.ndarray,
                                free_mask: np.ndarray) -> Callable:
    """
    Wraps a residual function that expects the FULL parameter vector so it
    instead accepts only the FREE-dimension subset scipy actually
    optimizes over — reconstructing the full vector (fixed entries from
    full_template, free entries from what scipy is proposing) before each
    call. This is the ONLY place fixed_params logic lives; everything else
    in calibrate() operates on full-length vectors as before.
    """
    def wrapped(x_free, *args):
        full_x = full_template.copy()
        full_x[free_mask] = x_free
        return residual_fn(full_x, *args)
    return wrapped


def _best_restart(restarts: list):
    """The restart with the lowest training cost, converged or not (#79). On hard calibrations restarts land in
    different minima; on URI's ALPHA RISE re-run the one with the lower training cost was the better one on unseen
    recordings every time, so choosing never needs the held-out data. A best restart that stopped at its evaluation
    budget is reported as not converged."""
    return min(restarts, key=lambda r: r.cost)


def _stability_report(all_restarts: list, converged_restarts: list) -> dict:
    """Actually measures parameter-set consistency across restarts,
    rather than asserting it in a docstring: mean/std of each parameter
    across CONVERGED solutions, plus what fraction of restarts converged."""
    xs = np.array([r.x for r in converged_restarts])
    return {
        "mean": xs.mean(axis=0),
        "std": xs.std(axis=0),
        "convergence_rate": len(converged_restarts) / len(all_restarts),
        "n_converged": len(converged_restarts),
        "n_total": len(all_restarts),
    }


def _write_log(path: str, log: CalibrationLog, restarts: list) -> None:
    payload = {
        "timestamp_utc": log.timestamp_utc, "method": log.method, "loss": log.loss,
        "n_restarts": log.n_restarts, "n_converged": log.n_converged,
        "best_cost": log.best_cost, "best_nfev": log.best_nfev,
        "runtime_seconds": log.runtime_seconds,
        "restarts": [
            {"cost": r.cost, "success": r.success, "status": r.status,
             "nfev": r.nfev, "message": r.message}
            for r in restarts
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _parallel_restart_worker(x0_free, lb_free, ub_free, x_scale_free,
                              full_template, free_mask, train_runs, vehicle,
                              dof_scales, normalize_by_run_length, method,
                              loss, f_scale, ftol, xtol, gtol, max_nfev, horizon_samples=None):
    """
    Module-level (NOT a closure/lambda) so it's picklable under Windows'
    'spawn' multiprocessing start method - required, not a style choice.
    Rebuilds the fixed_params wrapping fresh in THIS process, rather than
    receiving a closure built elsewhere (which spawn cannot pickle).
    Deliberately does not support on_iteration or simulate_fn - see
    calibrate_parallel()'s docstring for why.
    """
    import os as _os
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    print(f"  Worker starting (PID={_os.getpid()})...", flush=True)

    def wrapped_residual_fn(x_free, *args):
        full_x = full_template.copy()
        full_x[free_mask] = x_free
        return joint_residuals(full_x, *args)

    result = least_squares(
        fun=wrapped_residual_fn, x0=x0_free, bounds=(lb_free, ub_free),
        method=method, loss=loss, f_scale=f_scale, ftol=ftol, xtol=xtol,
        gtol=gtol, max_nfev=max_nfev, x_scale=x_scale_free, verbose=0,
        args=(train_runs, vehicle, dof_scales, normalize_by_run_length, None, horizon_samples),
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

    fixed_params = {**held_by_bounds(lb, ub, param_names), **(config.fixed_params or {})}
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
                config.horizon_samples,
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
    best = _best_restart(restarts)
    best_params = CalibrationParameters.from_vector(best.x)
    stability = _stability_report(restarts, converged)
    train_rmse = mean_run_rmse(best_params, train_runs, vehicle)

    log = CalibrationLog(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        method=config.method, loss=config.loss,
        n_restarts=config.n_restarts, n_converged=len(converged),
        best_cost=best.cost, best_nfev=best.nfev, runtime_seconds=runtime,
    )
    if config.log_path:
        _write_log(config.log_path, log, restarts)

    return CalibrationResult(
        parameters=best_params,
        converged=bool(best.success),
        cost=best.cost,
        n_iterations=best.nfev,
        train_rmse=train_rmse,
        restarts=restarts,
        stability=stability,
        log=log,
        scipy_result=None,
    )



def calibrate(
    loaded: LoadedVehicleSpec,
    train_runs: Sequence[RunData],
    config: Optional[CalibrationConfig] = None,
) -> CalibrationResult:
    """
    Runs config.n_restarts independent least-squares fits from different
    initial guesses (per config.initialization_strategy), returns the
    best CONVERGED result. Raises RuntimeError if every restart fails to
    converge — calibration code should fail loudly, not hand back an
    unchecked result (see module docstring).

    If config.fixed_params is set, those parameters are held EXACTLY
    constant (not part of the search space) — see module docstring's
    "Fixed parameters" section for why this exists.
    """
    config = config or CalibrationConfig()
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

    parallel_eval = None
    if config.eval_workers > 1 and config.simulate_fn is None:
        parallel_eval = ParallelJointResiduals(train_runs, vehicle, config.eval_workers)
    try:
        return _calibrate_with(loaded, train_runs, config, vehicle, param_names, x0_base, lb, ub, x_scale,
                               parallel_eval or joint_residuals)
    finally:
        if parallel_eval is not None:
            parallel_eval.close()


def _calibrate_with(loaded, train_runs, config, vehicle, param_names, x0_base, lb, ub, x_scale,
                    base_residual_fn) -> CalibrationResult:
    if config.prior_targets:  # --drag prior (drag_reference.py): part of the cost, so checkpoints include it
        from drag_reference import prior_residual_wrapper
        base_residual_fn = prior_residual_wrapper(base_residual_fn, list(param_names), config.prior_targets,
                                                  config.prior_weight)
    residual_fn = _wrap_residual_fn(base_residual_fn, config.on_iteration, config.stall_watchdog)

    # --- fixed_params: build the free/fixed split, subset everything the
    # optimizer actually sees to free dimensions only. Fully inert (every
    # array below is unchanged, mask is all-True) when fixed_params is None.
    fixed_params = {**held_by_bounds(lb, ub, param_names), **(config.fixed_params or {})}
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
    optimized_residual_fn = _make_fixed_params_wrapper(residual_fn, full_template, free_mask)

    rng = np.random.default_rng(config.seed)

    restarts = []
    start_time = time.monotonic()
    for _ in range(config.n_restarts):
        x0_free = _initialize_x0(x0_base_free, lb_free, ub_free, rng,
                                  config.initialization_strategy, config.perturbation_frac)
        result = least_squares(
            fun=optimized_residual_fn,
            x0=x0_free,
            bounds=(lb_free, ub_free),
            method=config.method,
            loss=config.loss,
            f_scale=config.f_scale,
            ftol=config.ftol,
            xtol=config.xtol,
            gtol=config.gtol,
            max_nfev=config.max_nfev,
            x_scale=x_scale_free,
            verbose=config.verbose,
            args=(train_runs, vehicle, config.dof_scales, config.normalize_by_run_length, config.simulate_fn,
                  config.horizon_samples),
        )
        # Reconstruct FULL-length x0/x — every downstream consumer
        # (CalibrationParameters.from_vector, checkpoint files, reporting)
        # expects a full param_names-length vector regardless of whether
        # fixed_params was used.
        full_x0 = full_template.copy()
        full_x0[free_mask] = x0_free
        full_x = full_template.copy()
        full_x[free_mask] = result.x
        restarts.append(RestartResult(
            x0=full_x0, x=full_x, cost=result.cost, success=result.success,
            status=result.status, nfev=result.nfev, message=result.message,
            raw_result=result,
        ))
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

    best = _best_restart(restarts)
    best_params = CalibrationParameters.from_vector(best.x)
    stability = _stability_report(restarts, converged)

    log = CalibrationLog(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        method=config.method, loss=config.loss,
        n_restarts=config.n_restarts, n_converged=len(converged),
        best_cost=best.cost, best_nfev=best.nfev, runtime_seconds=runtime,
    )
    if config.log_path:
        _write_log(config.log_path, log, restarts)

    train_rmse = mean_run_rmse(best_params, train_runs, vehicle)

    return CalibrationResult(
        parameters=best_params,
        converged=bool(best.success),
        cost=best.cost,
        n_iterations=best.nfev,
        train_rmse=train_rmse,
        restarts=restarts,
        stability=stability,
        log=log,
        scipy_result=best.raw_result,
    )