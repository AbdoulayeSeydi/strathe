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
from objective import joint_residuals, mean_run_rmse, RunData


# ---------------------------------------------------------------------------
# Loading + validating vehicle_spec.yaml
# ---------------------------------------------------------------------------

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

    # reproducibility / restarts
    seed: Optional[int] = None
    initialization_strategy: str = "fixed"  # "fixed" | "perturbed" | "random"
    perturbation_frac: float = 0.1          # for "perturbed": +/- this fraction of bounds width
    n_restarts: int = 1

    # instrumentation
    on_iteration: Optional[Callable] = None  # called as on_iteration(call_count, param_vec, cost)
    log_path: Optional[str] = None           # if set, writes a JSON run log here


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


def _wrap_residual_fn(base_fn: Callable, on_iteration: Optional[Callable]) -> Callable:
    """Scipy's least_squares has no native callback= — this wraps the
    residual function itself (which scipy calls every evaluation anyway)
    to report iteration count, current parameters, and cost."""
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
    """
    config = config or CalibrationConfig()
    vehicle = loaded.vehicle
    x0_base = initial_parameters(loaded).to_vector()
    lb, ub = parameter_bounds(loaded)

    if config.x_scale is not None:
        x_scale = config.x_scale
    elif config.use_bounds_width_x_scale:
        x_scale = ub - lb
    else:
        x_scale = np.ones_like(x0_base)

    residual_fn = _wrap_residual_fn(joint_residuals, config.on_iteration)
    rng = np.random.default_rng(config.seed)

    restarts = []
    start_time = time.monotonic()
    for _ in range(config.n_restarts):
        x0 = _initialize_x0(x0_base, lb, ub, rng, config.initialization_strategy, config.perturbation_frac)
        result = least_squares(
            fun=residual_fn,
            x0=x0,
            bounds=(lb, ub),
            method=config.method,
            loss=config.loss,
            f_scale=config.f_scale,
            ftol=config.ftol,
            xtol=config.xtol,
            gtol=config.gtol,
            max_nfev=config.max_nfev,
            x_scale=x_scale,
            verbose=config.verbose,
            args=(train_runs, vehicle, config.dof_scales, config.normalize_by_run_length),
        )
        restarts.append(RestartResult(
            x0=x0, x=result.x, cost=result.cost, success=result.success,
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

    best = min(converged, key=lambda r: r.cost)
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
        converged=True,
        cost=best.cost,
        n_iterations=best.nfev,
        train_rmse=train_rmse,
        restarts=restarts,
        stability=stability,
        log=log,
        scipy_result=best.raw_result,
    )