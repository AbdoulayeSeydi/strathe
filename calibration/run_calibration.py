#!/usr/bin/env python3
"""
CLI -> Pipeline -> Calibration Engine -> Evaluation -> Reporting.

    args -> PipelineConfig -> run_pipeline() -> CalibrationReport -> ConsoleReporter

run_pipeline() is the one function that does the real work (dataset
discovery, splitting, loading, calibration, evaluation, saving artifacts)
and returns a single CalibrationReport. main() itself is intentionally
thin: parse args, build config, call run_pipeline(), print the report,
set an exit code. Nothing else lives in main().

## Sampling rate — empirically determined default, still overridable
--sampling-rate-hz defaults to 10.0 Hz, empirically determined (see
objective.py's module docstring and PROJECT_STATUS.md finding #11) — NOT
guessed. The dataset has no time column; an earlier 50Hz placeholder was
found to be wrong by a validated investigation. Override this flag if
using a different dataset with an unconfirmed or different rate.

## Output artifacts
Every run writes, under --output-dir:
    calibrated_parameters.yaml   - the fitted parameters, ready to reuse
    metrics.json                 - full report, machine-readable
    optimizer_result.json        - per-restart scipy diagnostics (written
                                    by optimizer.py's own logging hook)
Otherwise a calibration run's results only exist as terminal output and
disappear the moment the shell scrolls past them.

## Exit codes
0 = pipeline completed AND optimizer converged. Non-zero = something in
the pipeline failed (bad input, empty split, optimizer never converging) —
never silently exits 0 on a failure a caller might be scripting against.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from checkpoint_io import CheckpointError, read_checkpoint, write_checkpoint
from data_split import split_by_date, discover_csv_files, DEFAULT_HELDOUT_FRACTION
from steering import SteeringError
from dynamics_model import CalibrationParameters
from stall_watchdog import StallWatchdog
from objective import read_currents_file
from objective import (DatasetConfig, RunData, compute_zero_baseline_scales, load_run, mean_run_rmse,
                       rmse_by_dof, skill_vs_zero, stable_mean_run_rmse, stable_rmse_by_dof, zero_baseline_rmse)
from optimizer import (
    LoadedVehicleSpec, load_vehicle_spec, initial_parameters,
    CalibrationConfig, RestartsFailed, calibrate, calibrate_parallel, held_by_bounds, parameter_bounds,
    result_from_restarts,
)
from mlflow_logging import log_calibration_run
from mlflow_schema import STATUS_SUCCESS, STATUS_FAILED
from keep_awake import keep_awake

logger = logging.getLogger("calibration_pipeline")

DOF_LABELS = ["u", "v", "w", "p", "q", "r"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
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
    max_workers: Optional[int] = None  # None = os.cpu_count() default
    axis_scaling: str = "none"  # "none" = raw units (original objective); "zero_rms" = each
        # axis divided by its "predict zero" error on the training runs (PROJECT_STATUS.md #50)
    current_by_file: Optional[tuple] = None    # a current per recording (objective.read_currents_file)
    current_world_mps: Optional[tuple] = None  # known current during the logs (log world frame);
        # None = vehicle_spec.yaml's environment block (still water for the tank dataset)
    torque_reference_offset_m: Optional[tuple] = None  # None = vehicle_spec.yaml's
        # data_conventions.torque_reference_offset_m (or none); see objective.shift_torque_reference
    exclude_axes: tuple = ()  # subset of ("p", "q"): axes the logged inputs don't explain. Their
        # residuals get zero weight and their axis-only parameters are held at vehicle_spec.yaml's
        # initial_guess instead of being fitted (see EXCLUDABLE_AXIS_PARAMS)
    drag: str = "free"                     # drag_reference.DRAG_MODES: free | prior
    drag_reference: str = "bluerov2_wu2018"
    drag_prior_weight: float = 0.1
    free_decay: Optional[str] = None  # free_decay.yaml (free_decay.py): roll/pitch damping and righting arm
        # measured in free-decay tank tests, held fixed instead of fitted from the mission logs
    checkpoint_path: Optional[str] = None  # where a retried fit's record is noted, for a not-converged finish (#93)
    fin_flights: bool = True                # a fin vehicle's second stage (fin_fit.py); False only keeps unit tests quick
    # the sensor suite (PROJECT_STATUS.md #96, SENSORS.md): which sensors to fit to (objective.DatasetConfig.sensors),
    # the accelerometer as a channel, how many times to re-estimate the windows' starting velocities with the fitted
    # values and refit, and the state smoother's process noise (None: chosen from the data)
    sensors: tuple = ("auto",)
    use_accelerometer: bool = False
    unseen: str = "infer"   # surge and sway no sensor measures: infer them from the gyro and depth, or hold them (#96)
    anchor_rounds: int = 2
    smoother_accel_sigma: Optional[float] = None


# ---------------------------------------------------------------------------
# Report structures
# ---------------------------------------------------------------------------

@dataclass
class DatasetSummary:
    train_dates: list
    heldout_dates: list
    n_train_files: int
    n_heldout_files: int
    total_samples: int
    total_trajectories: int
    split_mode: str = "dates"              # "dates" | "recordings" | "time block" (data_split.py, #93)
    split_warning: Optional[str] = None


@dataclass
class TimingSummary:
    loading_s: float
    calibration_s: float
    evaluation_s: float
    total_s: float


@dataclass
class OptimizerSummary:
    converged: bool
    cost: float
    nfev: int
    njev: Optional[int]
    status: int
    message: str
    optimality: float
    n_restarts: int
    n_converged: int
    convergence_rate: float
    restart_costs: Optional[list] = None  # each starting point's training cost, in order (#79)
    auto_restart: Optional[dict] = None   # a failed single-start fit retried from more starting points (#93)


AUTO_RESTART_GAIN = 0.9   # a fit must end below this fraction of the starting description's training error
AUTO_RESTART_ROUGHNESS = 1e-9   # ...and where the training cost is smooth (optimizer.end_roughness)
AUTO_RESTART_EXACT = 1e-3   # a fit whose training error is below this fraction of the start's explains the data all
    # but exactly (synthetic data, #96): at a cost near 0 the roughness measure is rounding noise, not a chaotic replay
AUTO_RESTARTS = 3         # up to this many more starting points when it doesn't


def fin_flight_stage(config, loaded_spec, train_runs, first, fin_info: dict, stopped=None):
    """A fin vehicle's second stage: on from the first stage's values, with the model's own flights in the objective (fin_fit.py). first: the first stage's
    result, or None with stopped, its RestartsFailed, when it ran out of budget: it goes on from where it stopped, and a fit with nothing to fly stops as it would have."""
    import fin_fit
    from dataclasses import replace as _replace
    count = fin_fit.plan_flights(train_runs)
    fin_info["flights"] = count
    if not count:
        logger.warning("Fins: these recordings are too short to fly %.0f s, so the fit never saw the vehicle fly itself", fin_fit.FLIGHT_S)
        fin_info["plain"].append(f"These recordings are too short to fly {fin_fit.FLIGHT_S:.0f} s, so the fit never saw the vehicle fly itself.")
        if first is None:
            raise stopped
        return first
    moving = fin_fit.flights_at_speed(train_runs, loaded_spec.vehicle.actuators if loaded_spec is not None else None)
    fin_info["flights_at_speed"] = moving
    if moving < fin_fit.MIN_FLIGHTS_AT_SPEED:
        fin_fit.clear_flights(train_runs)
        logger.warning("Fins: only %d of the %d flights of %.0f s were flown at speed (over %.1f m/s; %d needed), so the fit keeps its short-window values: "
                       "flights where the fins have no authority would fit the hull to what the fins can't explain",
                       moving, count, fin_fit.FLIGHT_S, fin_fit.MIN_SPEED_MPS, fin_fit.MIN_FLIGHTS_AT_SPEED)
        fin_info["plain"].append(f"Only {moving} of the {count} flights of {fin_fit.FLIGHT_S:.0f} s were flown at speed (over {fin_fit.MIN_SPEED_MPS:g} m/s; "
                                 f"{fin_fit.MIN_FLIGHTS_AT_SPEED} needed), so the fit kept its short-window values. Logs with more time at speed let it check "
                                 f"the vehicle flying itself.")
        if first is None:
            raise stopped
        return first
    logger.info("Fins: fitting on, with %d flights of %.0f s scored as well", count, fin_fit.FLIGHT_S)
    fin_info["plain"].append(f"After fitting short windows, the fit went on with {count} flights of {fin_fit.FLIGHT_S:.0f} s, where the model flies itself.")
    cc = config.calibration_config
    start = first.parameters.to_vector() if first is not None else min(stopped.restarts, key=lambda r: r.cost).x
    # costs of the two stages can't be compared, so this stage's checkpoint starts afresh
    second = _replace(cc, x0_override=start, n_restarts=1, initialization_strategy="fixed",
                      max_nfev=min(cc.max_nfev, fin_fit.FLIGHT_MAX_NFEV) if cc.max_nfev else fin_fit.FLIGHT_MAX_NFEV,
                      on_iteration=make_checkpoint_callback(config.checkpoint_path) if cc.on_iteration and config.checkpoint_path else cc.on_iteration)
    try:
        result = calibrate_parallel(loaded_spec, train_runs, second, max_workers=config.max_workers) if config.parallel else calibrate(loaded_spec, train_runs, second)
        fin_info["flight_stage"] = {"converged": True, "nfev": int(result.n_iterations)}
        return result
    except RestartsFailed as e:
        fin_info["flight_stage"] = {"converged": False, "nfev": int(e.restarts[0].nfev)}
        if first is None:      # neither stage converged: finished from the checkpoint, as any fit out of budget is
            raise
        # it starts from a converged fit and its budget is deliberately short, so stopping there is how it usually ends, not a failed fit
        logger.info("Fins: the flight stage used its %d-step budget; keeping where it got to", second.max_nfev)
        best = min(e.restarts, key=lambda r: r.cost)
        return result_from_restarts([_replace(best, success=True)], second, train_runs, loaded_spec.vehicle, 0.0)
    finally:
        fin_fit.clear_flights(train_runs)


def fit_with_retry(loaded_spec, train_runs, cc: CalibrationConfig, default_params, vehicle,
                   checkpoint_path: Optional[str] = None) -> tuple:
    """calibrate(), and for a fit from one cold starting point, up to AUTO_RESTARTS more starts when the first fails.
    Returns (result, the retry's record or None).

    Two ways a single start fails on real recordings (PROJECT_STATUS.md #93), both from one cause: the start drifts to
    values where the simulated vehicle is barely damped, the long open-loop replay turns chaotic, and the training cost
    jumps for any change in the values, so the optimizer's finite-difference slopes are noise. It then either stalls
    (stuck: training cost 1599, no better than the starting description, against 20 for the same day's other split) or
    wanders until its evaluation budget runs out (training cost 103 where the same data from another start reached 39).
    Both are caught: a fit no better than the starting description on its training recordings, or one that ended
    where the cost is rough (end_roughness above AUTO_RESTART_ROUGHNESS; 14 sound fits measured 1.5e-15 to 1.2e-10, the
    stuck one 2e-3, the wandering one 7.6e-8). A fit that ran out of budget on a smooth cost is not retried: it was
    still descending, which more budget fixes, not another start (112 of 276 results on our machines ran out of budget,
    with the same median error as converged fits'; strathe finishes them from the checkpoint and says so). The starts
    are pooled and the lowest training cost kept, as --restarts does. A warm start (x0_override) or --restarts N is
    left as asked."""
    from dataclasses import replace as _replace
    if cc.n_restarts != 1 or cc.x0_override is not None:
        return calibrate(loaded_spec, train_runs, cc), None
    cc = _replace(cc, measure_roughness=True)
    try:
        first, failed = calibrate(loaded_spec, train_runs, cc), None
        first_restarts, first_rmse = first.restarts, first.train_rmse
    except RestartsFailed as e:   # no convergence within the budget: judged the same way, from where it stopped
        first, failed, first_restarts = None, e, e.restarts
        try:
            first_rmse = mean_run_rmse(CalibrationParameters.from_vector(e.restarts[0].x), train_runs, vehicle)
        except (FloatingPointError, ValueError):
            first_rmse = float("inf")
    start_rmse = mean_run_rmse(default_params, train_runs, vehicle)
    rough = first_restarts[0].roughness or 0.0
    if first_rmse < AUTO_RESTART_EXACT * start_rmse:
        rough = 0.0
    no_better = not first_rmse < AUTO_RESTART_GAIN * start_rmse
    if not no_better and not rough > AUTO_RESTART_ROUGHNESS:
        if failed is not None:
            raise failed   # ran out of budget at a smooth, better answer: finished from its checkpoint, as before
        return first, None
    why = "no_better" if no_better else "rough"
    first_cost = float(first_restarts[0].cost)
    logger.warning("The fit %s (training cost %.4g): trying up to %d more starting points and keeping the lowest "
                   "training cost",
                   f"ended no better than the starting description on its own training recordings (error "
                   f"{first_rmse:.3g} against {start_rmse:.3g})" if no_better else
                   f"ended where the simulation is too sensitive to fit (nudging each value by 1e-8 of its range "
                   f"changed the training cost's curvature by {rough:.1e} of it, against ~1e-15 for a sound fit)",
                   first_cost, AUTO_RESTARTS)
    t0 = time.monotonic()
    more, seeds = [], []
    for k in range(1, AUTO_RESTARTS + 1):
        # one start at a time, each with its own seed (so --seed still repeats all of it), stopping at the first sound
        # one: converged below the first fit's cost, better than the starting description, on a smooth cost
        seeds.append((cc.seed or 0) + k)
        try:
            r = calibrate(loaded_spec, train_runs, _replace(cc, seed=seeds[-1]))
        except RestartsFailed as e:
            more += e.restarts
            continue
        more += r.restarts
        if r.cost < first_cost and r.train_rmse < AUTO_RESTART_GAIN * start_rmse and \
                not (r.restarts[0].roughness or 0.0) > AUTO_RESTART_ROUGHNESS:
            break
    pooled = list(first_restarts) + list(more)
    best = min(pooled, key=lambda r: r.cost)
    auto = {"reason": ("the first fit ended no better than the starting description on its training recordings"
                       if no_better else "the first fit ended where the training cost is rough (a chaotic replay)"),
            "why": why, "first_cost": first_cost, "first_converged": failed is None,
            "first_train_rmse": float(first_rmse), "start_train_rmse": float(start_rmse),
            "first_roughness": float(rough), "more_starts": len(more), "seeds_of_more_starts": seeds,
            "costs_of_more_starts": [float(r.cost) for r in more],
            "roughness_of_more_starts": [None if r.roughness is None else float(r.roughness) for r in more],
            "best_cost": float(best.cost), "best_converged": bool(best.success)}
    logger.info("Kept the lowest training cost of %d starting points: %.6g (the first: %.6g)%s", len(pooled),
                best.cost, first_cost, "" if best.success else ", which did not converge either")
    if checkpoint_path and Path(checkpoint_path).exists():   # kept by a not-converged finish (finalize_checkpoint)
        try:
            write_checkpoint(checkpoint_path, {**read_checkpoint(checkpoint_path), "auto_restart": auto,
                                               "restart_costs": [float(r.cost) for r in pooled]})
        except (OSError, CheckpointError):
            pass
    try:
        result = result_from_restarts(pooled, cc, train_runs, vehicle, time.monotonic() - t0)
    except RestartsFailed as e:
        e.auto_restart = auto
        raise
    auto["best_train_rmse"] = float(result.train_rmse)
    return result, auto


@dataclass
class RmseSummary:
    default_train: float
    default_heldout: float
    calibrated_train: float
    calibrated_heldout: float

    @property
    def improvement_train_pct(self) -> float:
        return 100.0 * (self.default_train - self.calibrated_train) / self.default_train

    @property
    def improvement_heldout_pct(self) -> float:
        return 100.0 * (self.default_heldout - self.calibrated_heldout) / self.default_heldout

    @property
    def generalization_gap_default(self) -> float:
        """heldout/train RMSE ratio — how much worse the default params
        do on unseen sessions vs. seen ones. Not meaningful as "overfitting"
        for the default (unfitted) params, but useful as a baseline."""
        return self.default_heldout / self.default_train

    @property
    def generalization_gap_calibrated(self) -> float:
        """Same ratio for the CALIBRATED params — this is the one that
        actually signals overfitting. A gap near 1x means the fit
        generalizes to unseen sessions; a large gap (e.g. 3x) means the
        optimizer fit session-specific quirks rather than vehicle physics."""
        return self.calibrated_heldout / self.calibrated_train


@dataclass
class PerDofSummary:
    default_train: np.ndarray
    default_heldout: np.ndarray
    calibrated_train: np.ndarray
    calibrated_heldout: np.ndarray
    # "predict zero" baseline and skill = 1 - RMSE/RMSE_zero (PROJECT_STATUS.md #50);
    # skill <= 0 means no better than assuming no motion on that axis
    zero_heldout: Optional[np.ndarray] = None
    skill_heldout: Optional[np.ndarray] = None          # whole-run replay
    skill_heldout_short: Optional[np.ndarray] = None    # SKILL_EVAL_HORIZON_S windows
    diverged_heldout: Optional[dict] = None  # held-out runs where the calibrated simulation blew up:
        # {"full_replay": n, "short_windows": n, "runs": total}; their skill is over the rest


@dataclass
class ParameterDelta:
    name: str
    default: float
    calibrated: float

    @property
    def magnitude_ratio(self) -> Optional[float]:
        """|calibrated| / |default| — the honest way to describe change for
        a signed physical quantity like damping (which is conventionally
        negative). A raw signed percentage on a negative-going value like
        -0.5 -> -10.0 reports "+1900%", which reads as "got much bigger"
        when it actually got much MORE NEGATIVE — same underlying fact,
        opposite intuition. Undefined (None) when default is exactly 0."""
        if self.default == 0:
            return None
        return abs(self.calibrated) / abs(self.default)

    @property
    def crossed_zero(self) -> bool:
        """True if default and calibrated have opposite signs — a ratio
        doesn't meaningfully describe this case either."""
        return bool((self.default * self.calibrated) < 0)

    def format_change(self) -> str:
        if self.default == 0:
            return "was zero — no ratio defined"
        if self.crossed_zero:
            return "sign flipped"
        return f"{self.magnitude_ratio:.2f}x magnitude"


# Exit code when the fit stops because the simulation blew up (NaN/inf), not because the budget ran out (#76).
EXIT_BLEW_UP = 3

@dataclass
class ReproducibilitySummary:
    git_commit: Optional[str]
    spec_version: str
    spec_path: str
    data_dir: str
    timestamp_utc: str
    vehicle_mass_kg: Optional[float] = None  # the spec's mass, so tools reading the calibration (export-stonefish) have it
    seed: Optional[int] = None  # the starting point's random seed: --seed N repeats this fit exactly (#76)


@dataclass
class CalibrationReport:
    dataset: DatasetSummary
    timing: TimingSummary
    optimizer: OptimizerSummary
    rmse: RmseSummary
    per_dof: PerDofSummary
    parameter_deltas: list
    calibrated_parameters: CalibrationParameters
    reproducibility: ReproducibilitySummary
    success: bool
    objective_axis_scaling: str = "none"
    objective_horizon_samples: Optional[int] = None
    objective_excluded_axes: tuple = ()
    vehicle_type: Optional[str] = None      # steering.py: hover, fin, vectored, hybrid
    objective_torque_reference_offset_m: tuple = (0.0, 0.0, 0.0)
    objective_current_world_mps: Optional[tuple] = None
    objective_held_parameters: Optional[dict] = None
    objective_drag: Optional[dict] = None
    objective_free_decay: Optional[dict] = None
    objective_sampling_rate_hz: float = 10.0
    objective_current_by_file: Optional[tuple] = None
    sensor_suite: Optional[dict] = None   # what the sensors could calibrate, per sensor axis's fit (#96)
    fins: Optional[dict] = None           # what the logs covered and what was held (fin_fit.py, #97)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _load_runs_parallel(filepaths: list, dataset_config: DatasetConfig, workers: int) -> list:
    """I/O-bound CSV loading — ThreadPoolExecutor gives a real speedup
    here despite the GIL, since most of the time is spent waiting on file
    reads / pandas parsing, not pure Python compute. executor.map()
    preserves input order regardless of completion order."""
    if not filepaths:
        return []
    if workers <= 1:
        return [load_run(fp, dataset_config) for fp in filepaths]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda fp: load_run(fp, dataset_config), filepaths))


def _average_rmse_by_dof(params, runs: list, vehicle, horizon_samples: Optional[int] = None) -> np.ndarray:
    rows = [rmse_by_dof(params, run, vehicle, horizon_samples=horizon_samples) for run in runs]
    if any(run.channels is not None for run in runs):   # u, v, w NaN where no DVL read (#96)
        a = np.array(rows)
        return np.array([np.nanmean(a[:, j]) if np.isfinite(a[:, j]).any() else np.nan for j in range(6)])
    return np.mean(rows, axis=0)


# The sensor suite's default prediction window (PROJECT_STATUS.md #96): position and depth need time to move before a
# window says anything about drag, and a longer window makes its starting velocity matter less. --prediction-horizon-s
# overrides it. Chosen by the #96 sweep (stress/run_96.py).
SENSOR_HORIZON_S = 10.0


def _sensor_mounts(loaded_spec) -> Optional[dict]:
    """The vehicle description's sensors: section (lever arms, a DVL's track), or None (#96)."""
    from measurements import sensor_mounts
    return sensor_mounts(loaded_spec.raw) or None


def smoother_noise_of(info: dict) -> dict:
    """The reading noise a fit's smoother used, from its sensor_suite record (before 2 Nov 2026: noise_from_ingest)."""
    return dict(info.get("noise", info.get("noise_from_ingest")) or {})


def sensor_suite_setup(config: "PipelineConfig", train_runs: list, heldout_runs: list, default_params, vehicle,
                       loaded_spec) -> dict:
    """Everything the sensor suite needs before the fit (PROJECT_STATUS.md #96): the prediction window, which linear
    axes the sensors see (surge and sway inferred, the rest held at the description's values), the smoother's process
    noise, each window's starting velocity, and each sensor axis's scale. Returns what to report."""
    import measurements as ms
    import state_smoother as sm
    cc = config.calibration_config
    if cc.horizon_samples is None:
        cc.horizon_samples = max(2, int(round(SENSOR_HORIZON_S * config.sampling_rate_hz)))
        logger.info("Sensor suite: %.0f s prediction windows (--prediction-horizon-s to change)", SENSOR_HORIZON_S)
    H = cc.horizon_samples
    for r in train_runs + heldout_runs:
        r._horizon = H
    labels = list(dict.fromkeys(lab for r in train_runs for lab in ms.channel_labels(r.channels)))
    seen = ms.axes_seen(train_runs, H)
    from steering import check_speed_source
    check_speed_source(loaded_spec.raw, seen)
    names = list(default_params.param_names)
    lb, ub = parameter_bounds(loaded_spec)
    already = set((cc.fixed_params or {}))
    # not for fins: their force goes as speed squared, so speed inferred from the turning they cause would be circular
    inferred = ms.inferred_axes(seen) if config.unseen == "infer" and vehicle.actuators is None else set()
    turning = ms.turning_rms(train_runs)
    too_little_turning = bool(inferred) and turning < ms.MIN_TURNING_RAD_S
    if too_little_turning:
        inferred = set()
        logger.warning("Sensor suite: the vehicle turned too little (%.2f rad/s RMS, under %.2f) to infer surge and "
                       "sway: held at the description's values (SENSORS.md)", turning, ms.MIN_TURNING_RAD_S)
    held = {k: v for k, v in ms.unobservable_holds(seen, default_params, inferred).items()
            if k in names and k not in already and lb[names.index(k)] < ub[names.index(k)]}
    if held:
        cc.fixed_params = {**(cc.fixed_params or {}), **held}
    for line in ms.seen_words(seen):
        logger.info("Sensor suite: %s", line)
    noise, noise_source = ms.smoother_noise(config.data_dir, ms.sensor_mounts(loaded_spec.raw),
                                            train_runs + heldout_runs)
    for name, src in sorted(noise_source.items()):
        logger.info("Sensor suite: %s's reading noise: %s (%s)", name,
                    f"{noise[name]:.3g}" if name in noise else "the smoother's default", src)
    def pick_sigma():
        if config.smoother_accel_sigma is not None:
            return float(config.smoother_accel_sigma), None
        return sm.choose_accel_sigma(train_runs, default_params, vehicle, noise)
    sigma, scores = pick_sigma()
    # a late position sensor: the delay the lab gave, else one the fixes clearly show, taken off before the window
    # starts and the fit (state_smoother.fix_delays)
    delays = sm.fix_delays(train_runs, default_params, vehicle, sigma, noise)
    given = ms.given_delays(config.data_dir, ms.sensor_mounts(loaded_spec.raw))
    for name, d in sorted(delays.items()):
        if name in given:
            d["applied_s"], d["applied_from"] = given[name]
        elif d.get("late"):
            d["applied_s"], d["applied_from"] = d["delay_s"], "found in the fixes"
            logger.warning("Sensor suite: %s's readings match the motion best %s%.1f s late: taken off (give "
                           "delay_s: in its %s: block to set it yourself, 0 if it's on time; SENSORS.md)", name,
                           "at least " if d["at_the_scan_edge"] else "", d["delay_s"], d.get("kind", "position"))
        else:
            d["applied_s"] = 0.0
        if d["applied_s"]:
            sm.shift_fixes(train_runs + heldout_runs, name, d["applied_s"])
    for name in sorted(set(given) - set(delays)):
        logger.warning("Sensor suite: %s's delay_s (%s) isn't used: no position or depth sensor %s in these "
                       "recordings", name, given[name][1], name)
    if any(d["applied_s"] for d in delays.values()):
        sigma, scores = pick_sigma()   # late fixes look like unexplained acceleration
    logger.info("Sensor suite: state smoother's unexplained acceleration %.3g m/s^2%s", sigma,
                "" if scores is None else " (the most likely of %s)" % sorted(scores))
    smoothing = sm.set_anchors(train_runs + heldout_runs, default_params, vehicle, sigma, noise)
    for f in smoothing["fallbacks"]:
        logger.warning("Sensor suite: smoother fell back to the DVL (or rest) on %s", f)
    scales = ms.still_scales(train_runs, H)
    only_heldout = {}
    for r in heldout_runs:
        for lab in ms.channel_labels(r.channels):
            if lab not in scales:
                only_heldout.setdefault(lab, []).append(r)
    for lab, rs in only_heldout.items():   # a sensor only in held-out recordings: scored against its own still RMS
        scales[lab] = ms.still_scales(rs, H)[lab]
    for r in train_runs + heldout_runs:
        r.sensor_scales = scales
    return {"sensors_asked": list(config.sensors), "use_accelerometer": bool(config.use_accelerometer),
            "channels": labels, "horizon_s": H / config.sampling_rate_hz, "axes_seen_by": seen,
            "plain": ms.seen_words(seen, inferred), "held_unobserved": held, "inferred_axes": sorted(inferred),
            "turning_rms_rad_s": turning, "too_little_turning": too_little_turning,
            "noise": noise,
            "noise_source": noise_source,
            "smoother": {"accel_sigma_mps2": sigma, "loglik_by_sigma": scores, **smoothing},
            "fix_delays": delays, "scales": scales, "only_in_heldout": sorted(only_heldout), "rounds": []}


def sensor_runs_from_metrics(runs: list, metrics: dict, params, vehicle) -> Optional[dict]:
    """A finished calibration's sensor-suite setup on reloaded runs (identifiability, validation, #96): the fit's
    prediction windows and sensor scales, as metrics.json recorded them, and each window's start from the smoother with
    `params` and the fit's process noise. None for runs fitted to velocity."""
    if not runs or runs[0].channels is None:
        return None
    import measurements as ms
    import state_smoother as sm
    ss = metrics.get("sensor_suite") or {}
    H = (metrics.get("objective") or {}).get("prediction_horizon_samples")
    for r in runs:
        r._horizon = H
    for name, d in (ss.get("fix_delays") or {}).items():   # the delay the fit took off a late sensor's fixes
        if d.get("applied_s"):
            sm.shift_fixes(runs, name, d["applied_s"])
    scales = dict(ss.get("scales") or {})
    missing = [lab for r in runs for lab in ms.channel_labels(r.channels) if lab not in scales]
    if missing:   # a sensor the fit never scaled (only in these recordings): its own still RMS
        scales.update({k: v for k, v in ms.still_scales(runs, H).items() if k in missing})
    for r in runs:
        r.sensor_scales = scales
    sigma = float((ss.get("smoother") or {}).get("accel_sigma_mps2") or 0.1)
    return sm.set_anchors(runs, params, vehicle, sigma, smoother_noise_of(ss))


def sensor_anchor_rounds(config: "PipelineConfig", loaded_spec, train_runs: list, heldout_runs: list, cal_result,
                         vehicle, info: dict):
    """Re-estimate every window's starting velocity with the fitted values, refit from there, up to
    config.anchor_rounds in all (#96): better physics gives better starting velocities. Stops early once the values
    move by under 0.1% of their ranges."""
    import state_smoother as sm
    from dataclasses import replace as _replace
    lb, ub = parameter_bounds(loaded_spec)
    width = np.where(ub > lb, ub - lb, 1.0)
    sigma = info["smoother"]["accel_sigma_mps2"]
    for rnd in range(2, max(1, config.anchor_rounds) + 1):
        prev = cal_result.parameters
        sm_info = sm.set_anchors(train_runs + heldout_runs, prev, vehicle, sigma, smoother_noise_of(info))
        cc = _replace(config.calibration_config, x0_override=prev.to_vector(), n_restarts=1)
        cal_result = calibrate(loaded_spec, train_runs, cc)
        moved = float(np.max(np.abs(cal_result.parameters.to_vector() - prev.to_vector()) / width))
        info["rounds"].append({"round": rnd, "cost": float(cal_result.cost), "converged": bool(cal_result.converged),
                               "largest_move_of_range": moved, "smoother_fallbacks": sm_info["fallbacks"],
                               "smoother_nis_mean": sm_info["nis_mean"]})
        logger.info("Sensor suite round %d: re-anchored with the fitted values and refit, cost %.6g (largest move "
                    "%.2g of a parameter's range)", rnd, cal_result.cost, moved)
        if moved < 1e-3:
            break
    return cal_result


def sensor_fit_summary(default_params, cal, train_runs: list, heldout_runs: list, vehicle) -> dict:
    """Per sensor axis: the default and calibrated values' RMSE on training and held-out recordings, what predicting
    no motion gets wrong on the held-out ones, and the held-out skill (1 - RMSE / still RMSE)."""
    import measurements as ms

    def avg(params, runs):
        rows = []
        for r in runs:
            try:
                rows.append(ms.channel_rmse(params, r, vehicle, r._horizon))
            except FloatingPointError:
                pass
        return ms.summarise(rows)
    out = {"default_train": avg(default_params, train_runs), "default_heldout": avg(default_params, heldout_runs),
           "calibrated_train": avg(cal, train_runs), "calibrated_heldout": avg(cal, heldout_runs),
           "still_heldout": ms.summarise([ms.still_rmse(r, r._horizon) for r in heldout_runs])}
    out["skill_heldout"] = {k: (1.0 - out["calibrated_heldout"][k] / v if np.isfinite(v) and v > 0 and
                                np.isfinite(out["calibrated_heldout"].get(k, np.nan)) else float("nan"))
                            for k, v in out["still_heldout"].items()}
    out["skill_heldout_default"] = {k: (1.0 - out["default_heldout"][k] / v if np.isfinite(v) and v > 0 and
                                        np.isfinite(out["default_heldout"].get(k, np.nan)) else float("nan"))
                                    for k, v in out["still_heldout"].items()}
    return out


def spec_torque_reference_offset(loaded_spec) -> tuple:
    """vehicle_spec.yaml's data_conventions.torque_reference_offset_m, or (0, 0, 0)."""
    raw = (loaded_spec.raw.get("data_conventions") or {}).get("torque_reference_offset_m", [0.0, 0.0, 0.0])
    if len(raw) != 3:
        raise ValueError(f"data_conventions.torque_reference_offset_m must have 3 numbers, got {raw}")
    return tuple(float(x) for x in raw)


# Parameters that ONLY shape one rotational axis's own dynamics. When that axis
# is excluded from the objective nothing else constrains them, so they are
# held at vehicle_spec.yaml's initial_guess. cg_cb_offset_z (the righting
# moment) shapes both roll and pitch, so it is held only when both are
# excluded. PROJECT_STATUS.md #50: in the reference dataset roll and pitch
# motion is not driven by the logged thruster torques, and fitting it pushes
# every roll/pitch damping term to its bound.
EXCLUDABLE_AXIS_PARAMS = {"p": ["Kp", "Kpp", "Ieff_roll"], "q": ["Mq", "Mqq", "Ieff_pitch"]}


def free_decay_setup(path: str, loaded_spec, default_params) -> dict:
    """The values a free_decay.yaml says to hold (free_decay.py), clamped into vehicle_spec.yaml's bounds with a
    note: {"file", "held", "measured", "notes"}. They override --exclude-axes' held reference values."""
    import free_decay as fd
    names = list(default_params.param_names)
    lb, ub = parameter_bounds(loaded_spec)
    measured = fd.hold_values(path)
    if not measured:
        raise ValueError(f"{path} holds nothing: the free-decay tests gave no usable roll or pitch fit (its "
                         f"warnings say why)")
    held, notes = fd.within_bounds(measured, names, lb, ub)
    return {"file": str(Path(path).resolve()), "held": held, "measured": measured, "notes": notes}


def excluded_axis_setup(exclude_axes, dof_scales, default_params) -> tuple:
    """(dof_scales with excluded axes zero-weighted, {param: held value})."""
    exclude_axes = tuple(dict.fromkeys(exclude_axes))
    bad = set(exclude_axes) - set(EXCLUDABLE_AXIS_PARAMS)
    if bad:
        raise ValueError(f"only {sorted(EXCLUDABLE_AXIS_PARAMS)} can be excluded, got {sorted(bad)}")
    scales = np.ones(6) if dof_scales is None else np.array(dof_scales, dtype=float)
    held = {}
    for ax in exclude_axes:
        scales[DOF_LABELS.index(ax)] = np.inf  # residual / inf = 0: the axis carries no weight
        for name in EXCLUDABLE_AXIS_PARAMS[ax]:
            held[name] = float(getattr(default_params, name))
    if {"p", "q"} <= set(exclude_axes):
        held["cg_cb_offset_z"] = float(default_params.cg_cb_offset_z)
    return scales, held


# skill is also reported over short re-anchored windows: 3 s is the longest
# beacon interval the fleet studies use, i.e. how far ahead a teammate model
# actually has to predict in this product
SKILL_EVAL_HORIZON_S = 3.0


def _jsonable(x):
    """Numbers and containers only, NaN as None: metrics.json stays strict JSON with a sensor suite in it."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def _save_artifacts(report: CalibrationReport, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    params_dict = {
        name: float(getattr(report.calibrated_parameters, name))
        for name in report.calibrated_parameters.param_names
    }
    with open(out / "calibrated_parameters.yaml", "w", encoding="utf-8") as f:
        yaml.dump({
            "generated_at_utc": report.reproducibility.timestamp_utc,
            "spec_version": report.reproducibility.spec_version,
            "git_commit": report.reproducibility.git_commit,
            "vehicle_mass_kg": report.reproducibility.vehicle_mass_kg,
            "seed": report.reproducibility.seed,
            **({"vehicle_type": report.vehicle_type} if report.vehicle_type else {}),
            "parameters": params_dict,
            # the sensors it was calibrated from, and what they couldn't see (#96); absent for a velocity fit
            **({"sensors": {"channels": report.sensor_suite["channels"],
                            "held_unobserved": report.sensor_suite["held_unobserved"],
                            "inferred_axes": report.sensor_suite["inferred_axes"]}}
               if report.sensor_suite else {}),
        }, f, sort_keys=False)

    metrics = {
        # split_mode / split_warning only for a one-session split (#93): a split by dates writes what it always has
        "dataset": {k: v for k, v in asdict(report.dataset).items()
                    if report.dataset.split_mode != "dates" or k not in ("split_mode", "split_warning")},
        "timing": asdict(report.timing),
        "optimizer": {k: v for k, v in asdict(report.optimizer).items()
                      if not (k == "auto_restart" and v is None)},   # only when a failed fit was retried (#93)
        "rmse": {
            "default_train": report.rmse.default_train,
            "default_heldout": report.rmse.default_heldout,
            "calibrated_train": report.rmse.calibrated_train,
            "calibrated_heldout": report.rmse.calibrated_heldout,
            "improvement_train_pct": report.rmse.improvement_train_pct,
            "improvement_heldout_pct": report.rmse.improvement_heldout_pct,
            "generalization_gap_default": report.rmse.generalization_gap_default,
            "generalization_gap_calibrated": report.rmse.generalization_gap_calibrated,
        },
        "per_dof": {
            "default_train": report.per_dof.default_train.tolist(),
            "default_heldout": report.per_dof.default_heldout.tolist(),
            "calibrated_train": report.per_dof.calibrated_train.tolist(),
            "calibrated_heldout": report.per_dof.calibrated_heldout.tolist(),
        },
        "skill_vs_zero": None if report.per_dof.zero_heldout is None else {
            "zero_heldout": report.per_dof.zero_heldout.tolist(),
            "calibrated_heldout_full_replay": report.per_dof.skill_heldout.tolist(),
            "calibrated_heldout_short_windows": report.per_dof.skill_heldout_short.tolist(),
            "short_window_s": SKILL_EVAL_HORIZON_S,
            "diverged_heldout": report.per_dof.diverged_heldout,
            "meaning": "1 - RMSE/RMSE_zero per axis [u,v,w,p,q,r]; <= 0 = no better than predicting no motion",
        },
        "objective": {"axis_scaling": report.objective_axis_scaling,
                      "prediction_horizon_samples": report.objective_horizon_samples,
                      "excluded_axes": list(report.objective_excluded_axes),
                      "torque_reference_offset_m": list(report.objective_torque_reference_offset_m),
                      "current_world_mps": (None if report.objective_current_world_mps is None
                                            else list(report.objective_current_world_mps)),
                      "held_parameters": report.objective_held_parameters,
                      "drag": report.objective_drag,
                      "free_decay": report.objective_free_decay,
                      "sampling_rate_hz": report.objective_sampling_rate_hz,
                      "current_by_file": (None if not report.objective_current_by_file else
                                          [[f, list(c)] for f, c in report.objective_current_by_file])},
        "parameter_deltas": [
            {"name": d.name, "default": d.default, "calibrated": d.calibrated,
             "magnitude_ratio": d.magnitude_ratio, "crossed_zero": d.crossed_zero}
            for d in report.parameter_deltas
        ],
        "reproducibility": asdict(report.reproducibility),
        "success": report.success,
    }
    if report.fins is not None:
        metrics["fins"] = _jsonable(report.fins)
    if report.sensor_suite is not None:
        metrics["sensor_suite"] = _jsonable(report.sensor_suite)
        # what validation, finishing a checkpoint and the report reload the logs with
        metrics["objective"]["sensors"] = report.sensor_suite["sensors_asked"]
        metrics["objective"]["use_accelerometer"] = report.sensor_suite["use_accelerometer"]
        metrics["skill_vs_zero"]["meaning"] += ("; u, v, w are NaN where no DVL read: see sensor_suite.fit for each "
                                                "sensor's own skill")
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Artifacts written to %s", out.resolve())


def run_pipeline(config: PipelineConfig) -> CalibrationReport:
    """The entire pipeline. Raises on any fatal condition (empty dataset,
    empty split, optimizer never converging) rather than returning a
    report that silently papers over a failure."""
    t_start = time.monotonic()

    logger.info("Discovering CSV files under %s", config.data_dir)
    csv_paths, duplicates = discover_csv_files(config.data_dir)
    if not csv_paths:
        raise RuntimeError(f"No CSV files found under {config.data_dir}")
    for dup_path, kept_path in duplicates:
        logger.warning("Duplicate filename skipped: %s (kept %s)", dup_path, kept_path)
    logger.info("Found %d CSV file(s)%s", len(csv_paths),
                f" ({len(duplicates)} duplicate(s) skipped)" if duplicates else "")

    split = split_by_date(csv_paths, heldout_dates=config.heldout_dates,
                           heldout_fraction=config.heldout_fraction)
    overlap = set(split.train_groups) & set(split.heldout_groups)
    if overlap:
        # Should be impossible given split_by_date's own logic — checked
        # explicitly anyway, per "never trust input."
        raise RuntimeError(f"train/heldout date overlap detected: {overlap}")
    logger.info("Split (by %s): %d train group(s) %s, %d heldout group(s) %s", split.mode,
                len(split.train_groups), split.train_groups,
                len(split.heldout_groups), split.heldout_groups)
    if split.warning:
        logger.warning(split.warning)

    loaded_spec = load_vehicle_spec(config.spec_path)
    import steering
    logger.info(steering.banner(loaded_spec.raw))
    with open(csv_paths[0], encoding="utf-8") as f:
        steering.check_data(loaded_spec.raw, f.readline().strip().split(","))
    if config.torque_reference_offset_m is None:
        config.torque_reference_offset_m = spec_torque_reference_offset(loaded_spec)
    if config.current_world_mps is None:
        from current import current_from_spec
        spec_current = current_from_spec(loaded_spec)
        config.current_world_mps = None if spec_current is None else tuple(float(x) for x in spec_current)
    dataset_config = DatasetConfig(sampling_rate_hz=config.sampling_rate_hz,
                                   torque_reference_offset_m=tuple(config.torque_reference_offset_m),
                                   current_world_mps=(None if config.current_world_mps is None
                                                      or not any(config.current_world_mps)
                                                      else tuple(config.current_world_mps)),
                                   current_by_file=config.current_by_file,
                                   sensors=tuple(config.sensors), use_accelerometer=config.use_accelerometer,
                                   lever_arms=_sensor_mounts(loaded_spec), actuators=loaded_spec.vehicle.actuators)
    if config.current_by_file:
        logger.info("Each recording's own water current, for %d recording(s) (from a currents file)",
                    len(config.current_by_file))
    if dataset_config.current_world_mps is not None:
        logger.info("Logs recorded in a known water current of %s m/s (log world frame)",
                    list(dataset_config.current_world_mps))
    if any(config.torque_reference_offset_m):
        logger.info("Logged torques re-expressed about a point offset %s m from the log's reference",
                    list(config.torque_reference_offset_m))
    vehicle = loaded_spec.vehicle

    logger.info("Loading runs (%d worker thread(s))...", config.workers)
    t_load = time.monotonic()
    train_runs = _load_runs_parallel(split.train_files, dataset_config, config.workers)
    heldout_runs = _load_runs_parallel(split.heldout_files, dataset_config, config.workers)
    loading_s = time.monotonic() - t_load

    if not train_runs:
        raise RuntimeError("No training runs loaded — cannot calibrate against an empty training set.")
    if not heldout_runs:
        raise RuntimeError("No held-out runs loaded — cannot validate against an empty held-out set.")
    modes = {r.channels is not None for r in train_runs + heldout_runs}
    if modes == {True, False}:
        # some recordings have a DVL reading at every sample, others don't: one objective for all, the sensors'
        logger.info("Sensor suite: some recordings have no DVL reading at every sample; every recording is fitted "
                    "through its sensors")
        from dataclasses import replace as _dc_replace
        dataset_config = _dc_replace(dataset_config, sensors=("all",))
        train_runs = _load_runs_parallel(split.train_files, dataset_config, config.workers)
        heldout_runs = _load_runs_parallel(split.heldout_files, dataset_config, config.workers)
    sensor_mode = train_runs[0].channels is not None

    total_samples = sum(r.n_samples for r in train_runs + heldout_runs)
    logger.info("Loaded %d train run(s), %d heldout run(s), %d total samples (%.2fs)",
                len(train_runs), len(heldout_runs), total_samples, loading_s)

    default_params = initial_parameters(loaded_spec)
    # parameters vehicle_spec.yaml holds by giving them zero-width bounds (e.g. no thruster_dynamics section: no
    # thruster lag). The optimizer holds them anyway; listing them here records them in metrics.json, so the
    # identifiability step and the report treat them as held too
    by_bounds = held_by_bounds(*parameter_bounds(loaded_spec), list(default_params.param_names))
    if by_bounds:
        config.calibration_config.fixed_params = {**by_bounds, **(config.calibration_config.fixed_params or {})}

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.calibration_config.log_path = str(Path(config.output_dir) / "optimizer_result.json")
    if config.axis_scaling == "zero_rms":
        config.calibration_config.dof_scales = compute_zero_baseline_scales(train_runs)
        logger.info("Axis scaling: each axis divided by its 'predict zero' error on training runs %s",
                    np.round(config.calibration_config.dof_scales, 4).tolist())
    elif config.axis_scaling != "none":
        raise ValueError(f"axis_scaling must be 'none' or 'zero_rms', got {config.axis_scaling!r}")
    if config.exclude_axes:
        scales, held = excluded_axis_setup(config.exclude_axes, config.calibration_config.dof_scales,
                                           default_params)
        config.calibration_config.dof_scales = scales
        config.calibration_config.fixed_params = {**(config.calibration_config.fixed_params or {}), **held}
        logger.info("Excluded axes %s from the objective; holding %s at vehicle_spec.yaml values",
                    list(config.exclude_axes), held)
    free_decay_info = None
    if config.free_decay:
        free_decay_info = free_decay_setup(config.free_decay, loaded_spec, default_params)
        config.calibration_config.fixed_params = {**(config.calibration_config.fixed_params or {}),
                                                  **free_decay_info["held"]}
        for n in free_decay_info["notes"]:
            logger.warning("Free decay: %s", n)
        logger.info("Holding roll/pitch values from free-decay tests (%s): %s", config.free_decay,
                    free_decay_info["held"])
    drag_info = None
    if config.drag != "free":
        import drag_reference as dr
        if config.drag not in dr.DRAG_MODES:
            raise ValueError(f"drag must be one of {dr.DRAG_MODES}, got {config.drag!r}")
        if config.parallel:
            raise ValueError("--drag prior isn't supported with --parallel restarts")
        names = list(default_params.param_names)
        lb, ub = parameter_bounds(loaded_spec)
        targets, notes = dr.within_bounds(dr.reference_values(config.drag_reference), names, lb, ub)
        config.calibration_config.prior_targets = targets
        config.calibration_config.prior_weight = config.drag_prior_weight
        drag_info = {"mode": config.drag, "reference": dr.REFERENCES[config.drag_reference]["source"],
                     "targets": targets, "weight": config.drag_prior_weight, "notes": notes}
        for n in notes:
            logger.info("Drag reference: %s", n)
        logger.info("Drag: %s, toward %s", config.drag, dr.REFERENCES[config.drag_reference]["source"])
    sensor_info = None
    fin_info = None
    if vehicle.actuators is not None and vehicle.actuators.fins:
        import fin_fit
        fin_info = fin_fit.setup(train_runs, vehicle.actuators, default_params, config.calibration_config.fixed_params)
        config.calibration_config.fixed_params = {**(config.calibration_config.fixed_params or {}), **fin_info["held"]}
        if config.calibration_config.horizon_samples is None and train_runs[0].channels is None:
            config.calibration_config.horizon_samples = fin_fit.window_samples(train_runs)
            logger.info("Fins: %d-sample prediction windows (--prediction-horizon-s to change)", config.calibration_config.horizon_samples)
        for line in fin_info["plain"]:
            logger.info("Fins: %s", line)
        for s, why in fin_info["held_why"].items():
            logger.warning("Fins: %s: their lift keeps its geometry value", why)
        for word in fin_info["few_manoeuvres"]:
            if {"steering": "a", "pitching": "b"}.get(word) in fin_info["held_why"]:
                continue
            logger.warning("Fins: few %s manoeuvres in these logs: the fit can predict missions like these while its fin "
                           "value is well off (FINS.md)", word)
    if sensor_mode:
        if config.axis_scaling != "none":
            logger.info("Sensor suite: --axis-scaling doesn't apply (each sensor axis is scaled by what predicting no "
                        "motion gets wrong)")
        sensor_info = sensor_suite_setup(config, train_runs, heldout_runs, default_params, vehicle, loaded_spec)
    if config.calibration_config.horizon_samples is not None:
        logger.info("Objective: re-anchored %d-sample prediction windows (%.1f s)",
                    config.calibration_config.horizon_samples,
                    config.calibration_config.horizon_samples / config.sampling_rate_hz)

    logger.info("Running calibration: %d restart(s), loss=%s, seed=%s",
                config.calibration_config.n_restarts, config.calibration_config.loss,
                config.calibration_config.seed)
    t_cal = time.monotonic()
    stopped = None
    try:
        if config.parallel:
            cal_result = calibrate_parallel(loaded_spec, train_runs, config.calibration_config,
                                             max_workers=config.max_workers)
        else:
            cal_result, auto_restart = fit_with_retry(loaded_spec, train_runs, config.calibration_config,
                                                      default_params, vehicle, config.checkpoint_path)
    except RestartsFailed as e:
        if fin_info is None:
            raise
        cal_result, stopped = None, e
    if sensor_info is not None and config.anchor_rounds > 1 and cal_result is not None:
        cal_result = sensor_anchor_rounds(config, loaded_spec, train_runs, heldout_runs, cal_result, vehicle,
                                          sensor_info)
    if fin_info is not None:
        cal_result = fin_flight_stage(config, loaded_spec, train_runs, cal_result, fin_info, stopped) if config.fin_flights else cal_result
        if cal_result is None:
            raise stopped
    calibration_s = time.monotonic() - t_cal
    logger.info("Calibration converged: %s (%d/%d restarts), cost=%.6g (%.2fs)",
                cal_result.converged, cal_result.stability["n_converged"],
                cal_result.stability["n_total"], cal_result.cost, calibration_s)

    logger.info("Evaluating default vs. calibrated on train and heldout...")
    t_eval = time.monotonic()
    cal = cal_result.parameters
    # a calibrated set can blow up on recordings it wasn't fitted to: counted and reported, not a crash
    rmse = RmseSummary(
        default_train=mean_run_rmse(default_params, train_runs, vehicle),
        default_heldout=mean_run_rmse(default_params, heldout_runs, vehicle),
        calibrated_train=stable_mean_run_rmse(cal, train_runs, vehicle)[0],
        calibrated_heldout=stable_mean_run_rmse(cal, heldout_runs, vehicle)[0],
    )
    cal_heldout, diverged_full = stable_rmse_by_dof(cal, heldout_runs, vehicle)
    per_dof = PerDofSummary(
        default_train=_average_rmse_by_dof(default_params, train_runs, vehicle),
        default_heldout=_average_rmse_by_dof(default_params, heldout_runs, vehicle),
        calibrated_train=stable_rmse_by_dof(cal, train_runs, vehicle)[0],
        calibrated_heldout=cal_heldout,
    )
    per_dof.zero_heldout = zero_baseline_rmse(heldout_runs)
    per_dof.skill_heldout = skill_vs_zero(per_dof.calibrated_heldout, per_dof.zero_heldout)
    short = max(2, int(round(SKILL_EVAL_HORIZON_S * config.sampling_rate_hz)))
    cal_short, diverged_short = stable_rmse_by_dof(cal, heldout_runs, vehicle, short)
    per_dof.skill_heldout_short = skill_vs_zero(cal_short, per_dof.zero_heldout)
    per_dof.diverged_heldout = {"full_replay": diverged_full, "short_windows": diverged_short,
                                "runs": len(heldout_runs)}
    if diverged_full or diverged_short:
        logger.warning("With the calibrated values the simulation diverges on %d of %d held-out recordings "
                       "(whole-run replay) and %d (%.0f s windows); their skill is over the rest",
                       diverged_full, len(heldout_runs), diverged_short, SKILL_EVAL_HORIZON_S)
    if sensor_info is not None:
        sensor_info["fit"] = sensor_fit_summary(default_params, cal, train_runs, heldout_runs, vehicle)
    evaluation_s = time.monotonic() - t_eval
    total_s = time.monotonic() - t_start

    scipy_result = cal_result.scipy_result
    if scipy_result is not None:
        optimizer_summary = OptimizerSummary(
            converged=cal_result.converged,
            cost=cal_result.cost,
            nfev=scipy_result.nfev,
            njev=getattr(scipy_result, "njev", None),
            status=scipy_result.status,
            message=scipy_result.message,
            optimality=float(scipy_result.optimality),
            n_restarts=len(cal_result.restarts),   # with a retry's starts (#93)
            n_converged=cal_result.stability["n_converged"],
            convergence_rate=cal_result.stability["convergence_rate"],
            restart_costs=[float(r.cost) for r in cal_result.restarts],
            auto_restart=auto_restart,
        )
    else:
        # calibrate_parallel() (config.parallel=True) never populates
        # scipy_result - it deliberately doesn't ship the full scipy
        # OptimizeResult back across the process boundary. Fall back to
        # the best individual restart's own stored fields instead of
        # crashing on scipy_result.nfev. optimality has no substitute
        # (RestartResult doesn't store it) - NaN, not silently omitted.
        best_restart = min(cal_result.restarts, key=lambda r: r.cost)
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
            restart_costs=[float(r.cost) for r in cal_result.restarts],
        )

    parameter_deltas = [
        ParameterDelta(name=n, default=getattr(default_params, n), calibrated=getattr(cal_result.parameters, n))
        for n in default_params.param_names
    ]

    reproducibility = ReproducibilitySummary(
        git_commit=_get_git_commit(),
        spec_version=loaded_spec.raw.get("schema_version", "unknown"),
        spec_path=config.spec_path,
        data_dir=config.data_dir,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        vehicle_mass_kg=float(loaded_spec.vehicle.mass_kg),
        seed=config.calibration_config.seed,
    )

    report = CalibrationReport(
        vehicle_type=steering.steering_of(loaded_spec.raw),
        fins=fin_info,
        dataset=DatasetSummary(
            train_dates=split.train_groups, heldout_dates=split.heldout_groups,
            n_train_files=len(train_runs), n_heldout_files=len(heldout_runs),
            total_samples=total_samples, total_trajectories=len(train_runs) + len(heldout_runs),
            split_mode=split.mode, split_warning=split.warning,
        ),
        timing=TimingSummary(loading_s=loading_s, calibration_s=calibration_s,
                              evaluation_s=evaluation_s, total_s=total_s),
        optimizer=optimizer_summary,
        rmse=rmse,
        per_dof=per_dof,
        parameter_deltas=parameter_deltas,
        calibrated_parameters=cal_result.parameters,
        reproducibility=reproducibility,
        success=cal_result.converged,
        objective_axis_scaling=config.axis_scaling,
        objective_horizon_samples=config.calibration_config.horizon_samples,
        objective_excluded_axes=tuple(config.exclude_axes),
        objective_torque_reference_offset_m=tuple(config.torque_reference_offset_m),
        objective_current_world_mps=dataset_config.current_world_mps,
        objective_held_parameters=config.calibration_config.fixed_params,
        objective_drag=drag_info,
        objective_free_decay=free_decay_info,
        objective_sampling_rate_hz=config.sampling_rate_hz,
        objective_current_by_file=config.current_by_file,
        sensor_suite=sensor_info,
    )

    _save_artifacts(report, config.output_dir)
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class ConsoleReporter:
    def report(self, r: CalibrationReport) -> None:
        print("=" * 70)
        print("CALIBRATION REPORT")
        print("=" * 70)

        print("\nReproducibility:")
        print(f"  Timestamp (UTC):    {r.reproducibility.timestamp_utc}")
        print(f"  Git commit:         {r.reproducibility.git_commit or 'unknown (not a git repo, or git unavailable)'}")
        print(f"  Spec version:       {r.reproducibility.spec_version}")
        print(f"  Spec path:          {r.reproducibility.spec_path}")
        print(f"  Data dir:           {r.reproducibility.data_dir}")

        print("\nDataset:")
        print(f"  Train dates ({len(r.dataset.train_dates)}):     {r.dataset.train_dates}")
        print(f"  Heldout dates ({len(r.dataset.heldout_dates)}):   {r.dataset.heldout_dates}")
        print(f"  Train files:        {r.dataset.n_train_files}")
        print(f"  Heldout files:      {r.dataset.n_heldout_files}")
        print(f"  Total trajectories: {r.dataset.total_trajectories}")
        print(f"  Total samples:      {r.dataset.total_samples}")

        print("\nTiming:")
        print(f"  Loading:      {r.timing.loading_s:.2f}s")
        print(f"  Calibration:  {r.timing.calibration_s:.2f}s")
        print(f"  Evaluation:   {r.timing.evaluation_s:.2f}s")
        print(f"  Total:        {r.timing.total_s:.2f}s")

        print("\nOptimizer:")
        print(f"  Converged:              {r.optimizer.converged}")
        print(f"  Status / message:       {r.optimizer.status} — {r.optimizer.message}")
        print(f"  Cost:                   {r.optimizer.cost:.6g}")
        print(f"  Optimality (grad norm): {r.optimizer.optimality:.3e}")
        print(f"  nfev / njev:            {r.optimizer.nfev} / {r.optimizer.njev}")
        print(f"  Restarts converged:     {r.optimizer.n_converged}/{r.optimizer.n_restarts} "
              f"({r.optimizer.convergence_rate:.0%})")

        print("\nRMSE (mean per-run, unweighted, natural units):")
        print(f"  {'':<14}{'Train':>12}{'Heldout':>12}")
        print(f"  {'Default':<14}{r.rmse.default_train:>12.4f}{r.rmse.default_heldout:>12.4f}")
        print(f"  {'Calibrated':<14}{r.rmse.calibrated_train:>12.4f}{r.rmse.calibrated_heldout:>12.4f}")
        print(f"  Improvement:   train {r.rmse.improvement_train_pct:+.1f}%   "
              f"heldout {r.rmse.improvement_heldout_pct:+.1f}%")
        print(f"  Generalization gap (heldout/train): "
              f"default {r.rmse.generalization_gap_default:.2f}x, "
              f"calibrated {r.rmse.generalization_gap_calibrated:.2f}x")

        print("\nRMSE by DOF (calibrated — train / heldout):")
        for i, lbl in enumerate(DOF_LABELS):
            print(f"  {lbl}: {r.per_dof.calibrated_train[i]:.4f} / {r.per_dof.calibrated_heldout[i]:.4f}")

        if r.sensor_suite is not None:
            ss = r.sensor_suite
            print(f"\nSensor suite (calibrated from {', '.join(sorted({l.rsplit('.', 1)[0] for l in ss['channels']}))};"
                  f" {ss['horizon_s']:g} s windows):")
            for line in ss["plain"]:
                print(f"  {line}")
            if ss["held_unobserved"]:
                print(f"  Held at the description's values (no sensor sees them): {', '.join(ss['held_unobserved'])}")
            fit = ss.get("fit") or {}
            print(f"  Held-out skill per sensor axis (calibrated / default physics):")
            for lab, v in (fit.get("skill_heldout") or {}).items():
                d = (fit.get("skill_heldout_default") or {}).get(lab, float("nan"))
                print(f"    {lab:<14}{v:>+8.2f} / {d:>+6.2f}")
            print("  u, v, w below are the DVL's (nan: no DVL reading)")
        if r.per_dof.skill_heldout is not None:
            print(f"\nHeld-out skill vs predicting no motion (1 = perfect, <= 0 = no better than zero):")
            whole = "whole run" if r.sensor_suite is None else f"{r.sensor_suite['horizon_s']:g} s windows"
            print(f"  {'':<6}{whole:>11}{f'{SKILL_EVAL_HORIZON_S:g} s windows':>14}")
            for i, lbl in enumerate(DOF_LABELS):
                print(f"  {lbl:<6}{r.per_dof.skill_heldout[i]:>+11.2f}{r.per_dof.skill_heldout_short[i]:>+14.2f}")

        print("\nParameter changes (default -> calibrated):")
        for d in r.parameter_deltas:
            print(f"  {d.name:<16}{d.default:>10.4f} -> {d.calibrated:>10.4f}  ({d.format_change()})")

        print("=" * 70)


# ---------------------------------------------------------------------------
# CLI — intentionally thin
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", default="calibration_output")
    parser.add_argument(
        "--sampling-rate-hz", type=float, default=10.0,
        help="Defaults to 10.0 Hz — empirically determined via "
             "check_sampling_rate_windowed.py, not the dataset's own metadata "
             "(it has no time column). Validated by: synthetic ground-truth "
             "recovery, survival of a deliberate sensor-bias drift stress "
             "test, agreement across all 4 collection dates independently, "
             "stability across 3 window lengths, a split-half reproducibility "
             "check, per-run outlier screening (37/39 runs consistent), and "
             "confirmation that row counts cluster at clean round-number "
             "durations at 10Hz (200s/400s/600s) — the signature of a real "
             "fixed clock, not curve-fitting. Full writeup in "
             "PROJECT_STATUS.md. Override this if using a different dataset.",
    )
    parser.add_argument("--heldout-dates", "--heldout-recordings", dest="heldout_dates", nargs="*", default=None,
                         help='e.g. --heldout-dates 16-02-2024 19-02-2024; or recordings (one session, #93)')
    parser.add_argument("--heldout-fraction", type=float, default=DEFAULT_HELDOUT_FRACTION)
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=1,
                        help="evaluate the training runs in this many worker processes per optimizer "
                             "evaluation (identical results; ~N-fold faster). Not combinable with --parallel")
    parser.add_argument("--torque-reference-offset-m", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="re-express the logged torques about a point this far (body FRD, m) from the "
                             "log's own reference. Default: vehicle_spec.yaml's data_conventions "
                             "(0 0 -0.085 for the BlueROV2 Heavy dataset, PROJECT_STATUS.md #50); "
                             "0 0 0 = use the logged torques as they are")
    parser.add_argument("--current-world-mps", type=float, nargs=3, default=None, metavar=("VX", "VY", "VZ"),
                        help="known water current during the recordings, in the logs' world frame (m/s). "
                             "Default: vehicle_spec.yaml's environment block (still water for the tank "
                             "dataset); 0 0 0 = still water (PROJECT_STATUS.md #55)")
    parser.add_argument("--currents-file", default=None,
                        help="currents.csv: a water current for each recording (file, vx_mps, vy_mps, vz_mps), as "
                             "estimate_current.py writes. Recordings it lists use their own current")
    parser.add_argument("--axis-scaling", choices=["none", "zero_rms"], default="none",
                        help="'zero_rms' scores each axis relative to its 'predict zero' error on the "
                             "training runs, instead of raw m/s and rad/s (PROJECT_STATUS.md #50)")
    parser.add_argument("--exclude-axes", nargs="+", choices=["p", "q"], default=[],
                        help="leave these rotational axes out of the fit and hold their axis-only "
                             "parameters at vehicle_spec.yaml's initial_guess -- for data whose roll/pitch "
                             "motion isn't driven by the logged thrusters (PROJECT_STATUS.md #50)")
    parser.add_argument("--drag", choices=["free", "prior"], default="free",
                        help="free: fit every drag term (default). prior: fit all, gently pulled toward published "
                             "values, for data that can't separate linear from quadratic drag (drag_reference.py)")
    parser.add_argument("--drag-reference", default="bluerov2_wu2018",
                        help="published drag values for --drag (drag_reference.REFERENCES)")
    parser.add_argument("--drag-prior-weight", type=float, default=0.1,
                        help="with --drag prior: weight of the pull toward the published values")
    parser.add_argument("--free-decay", default=None,
                        help="free_decay.yaml from free_decay.py (`strathe free-decay`): hold the roll/pitch damping "
                             "and righting arm measured in free-decay tank tests instead of fitting them")
    parser.add_argument("--sensors", nargs="+", default=["auto"],
                        help="which sensors to fit to (SENSORS.md, PROJECT_STATUS.md #96). auto (default): the "
                             "original velocity fit when the logs have a DVL reading at every sample, otherwise every "
                             "sensor in them (depth, position fixes, a DVL with gaps); velocity: the original fit "
                             "only; all: every sensor, the DVL among them; or names: dvl, a position or depth sensor's "
                             "name, acc. The gyro is always used")
    parser.add_argument("--unseen", choices=("infer", "hold"), default="infer",
                        help="surge and sway when no sensor measures them but depth holds heave. infer (default): fit "
                             "them from how they move the turning rate and depth; hold: keep the vehicle "
                             "description's values (SENSORS.md)")
    parser.add_argument("--use-accelerometer", action="store_true",
                        help="fit to the accelerometer too (sensor suite; experimental: #96 measures whether it helps)")
    parser.add_argument("--anchor-rounds", type=int, default=2,
                        help="sensor suite: re-estimate each window's starting velocity with the fitted values and "
                             "refit, up to this many rounds in all (default 2)")
    parser.add_argument("--smoother-accel-sigma", type=float, default=None,
                        help="sensor suite: the state smoother's unexplained acceleration, m/s^2 (default: the most "
                             "likely value for the data)")
    parser.add_argument("--prediction-horizon-s", type=float, default=None,
                        help="score re-anchored prediction windows of this length instead of replaying "
                             "whole runs open-loop; stops unexplained motion (e.g. roll/pitch) from "
                             "accumulating and biasing damping to its bounds (PROJECT_STATUS.md #50)")
    parser.add_argument("--initialization-strategy", choices=["fixed", "perturbed", "random"],
                         default="perturbed",
                         help="How each restart's starting point is chosen. 'fixed' repeats the "
                              "SAME starting point every restart — with n-restarts>1 this silently "
                              "produces identical runs (a real bug this default was fixing). "
                              "'perturbed' (default) randomly offsets each restart from the initial "
                              "guess; 'random' starts uniformly within bounds.")
    parser.add_argument("--perturbation-frac", type=float, default=0.1,
                         help="For --initialization-strategy perturbed: +/- this fraction of each "
                              "parameter's bounds width.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--loss", default="soft_l1")
    parser.add_argument("--jacobian-step", type=float, default=None,
                        help="finite-difference step for the optimizer's slopes, relative to each value (default: "
                             "scipy's, ~1.5e-8)")
    parser.add_argument(
        "--max-nfev", type=int, default=300,
        help="Max function evaluations PER RESTART. Left uncapped, scipy "
             "defaults to roughly 100x the number of parameters (~2300 "
             "here), which at this model's real-data cost (~0.5s/eval on "
             "the full training set) means ~19 minutes PER RESTART with "
             "no visibility into how long it'll actually take. 300 is a "
             "reasonable bounded default (~2-3 min/restart); raise it if "
             "runs are hitting this cap without converging (--verbose "
             "shows the termination reason).",
    )
    parser.add_argument("--workers", type=int, default=4)
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
                              "os.cpu_count() if not given.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint-path", default=None,
                         help="Where to save the live checkpoint (overwritten after every "
                              "single evaluation, not just improving ones). Defaults to "
                              "<output-dir>/checkpoint.yaml. Added after a real, confirmed "
                              "latency spike (one evaluation took ~17 minutes vs. the normal "
                              "~9.5s) during a long real run with no prior protection.")
    parser.add_argument("--resume-from-checkpoint", default=None,
                         help="Path to a checkpoint.yaml from a previously-interrupted run — "
                              "resumes from its best-parameters-so-far instead of "
                              "vehicle_spec.yaml's naive initial guess.")
    return parser


def make_checkpoint_callback(checkpoint_path):
    """
    Same proven pattern as run_calibration_holoocean.py's checkpoint —
    saves the BEST (lowest-cost) parameter vector seen so far after EVERY
    evaluation (overwriting one file, not accumulating), so an
    interruption only ever loses the most recent single evaluation's
    progress, not the whole run. Built after run_calibration.py's first
    real 12+ hour attempt showed a genuine, unexplained latency spike
    (one evaluation ~17 minutes vs. the normal ~9.5s) with zero protection
    against losing everything to an interruption.
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
        }
        write_checkpoint(checkpoint_path, checkpoint)  # atomic: never left empty by an interruption

        # Timing log for the still-unexplained performance mystery
        # (PROJECT_STATUS.md) - appends ONE line per real evaluation with
        # a real timestamp, so a future long stand-in run gives a genuine
        # post-hoc distribution of per-call timing instead of another
        # guess. Separate from checkpoint.yaml so it never risks
        # corrupting the resumable state.
        timing_log_path = str(Path(checkpoint_path).with_name("timing_log.csv"))
        is_new = not Path(timing_log_path).exists()
        with open(timing_log_path, "a", encoding="utf-8") as f:
            if is_new:
                f.write("call_count,unix_timestamp,cost\n")
            f.write(f"{call_count},{__import__('time').time()},{cost}\n")

    return callback


def load_checkpoint_warm_start(checkpoint_path: str) -> np.ndarray:
    data = read_checkpoint(checkpoint_path)
    params = CalibrationParameters(**data["best_parameters_so_far"])
    print(f"Resuming from checkpoint: call_count={data['call_count']}, "
          f"best_cost_so_far={data['best_cost_so_far']:.6f}")
    return params.to_vector()


def note_restart_costs(checkpoint_path: str, error: Exception) -> None:
    """No start converged: add each start's cost to the checkpoint, so the result finished from it reports them (#79)."""
    costs = getattr(error, "restart_costs", None)
    if costs and Path(checkpoint_path).exists():
        try:
            write_checkpoint(checkpoint_path, {**read_checkpoint(checkpoint_path), "restart_costs": costs})
        except (OSError, CheckpointError):
            pass


def main() -> None:
    args = build_arg_parser().parse_args()

    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-8s %(message)s")

    checkpoint_path = args.checkpoint_path or str(Path(args.output_dir) / "checkpoint.yaml")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # Auto-backup: protect whatever already exists in the output dir from
    # a PRIOR run before this one touches it at all. Same fix already
    # applied to run_calibration_holoocean.py - this closes the identical
    # gap in this (stand-in) script, which never got it.
    import shutil as _shutil
    import time as _time
    _output_dir = Path(checkpoint_path).parent
    _backup_tag = _time.strftime("%Y%m%d_%H%M%S")
    for _fname in ("calibrated_parameters.yaml", "checkpoint.yaml", "metrics.json"):
        _existing = _output_dir / _fname
        if _existing.exists():
            _backup_path = _output_dir / f"{_existing.stem}_backup_{_backup_tag}{_existing.suffix}"
            _shutil.copy2(_existing, _backup_path)
            print(f"Auto-backed up existing {_fname} -> {_backup_path.name} before this run starts")

    print(f"Checkpointing every evaluation to: {checkpoint_path}")

    x0_override = None
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint {args.resume_from_checkpoint}...")
        try:
            x0_override = load_checkpoint_warm_start(args.resume_from_checkpoint)
        except CheckpointError as e:
            print(f"❌ can't resume: {e}")
            sys.exit(2)

    if args.parallel:
        print("--parallel is set: checkpointing DISABLED for this run - "
              "calibrate_parallel() cannot safely checkpoint across multiple "
              "processes writing the same file. An interrupted parallel run "
              "cannot be resumed; re-run from scratch if needed.")
    if args.seed is None and args.initialization_strategy != "fixed":
        # The start is randomly perturbed: draw the seed here, so it's recorded and `--seed N` repeats the fit (#76).
        import secrets
        args.seed = secrets.randbelow(2**31 - 1)
    calibration_config = CalibrationConfig(
        loss=args.loss, n_restarts=args.n_restarts, seed=args.seed, max_nfev=args.max_nfev,
        diff_step=args.jacobian_step,
        initialization_strategy=args.initialization_strategy,
        perturbation_frac=args.perturbation_frac,
        verbose=2 if args.verbose else 0,
        x0_override=x0_override,
        on_iteration=None if args.parallel else make_checkpoint_callback(checkpoint_path),
        # times every evaluation on 3 clocks and stack-dumps stalls into
        # output_dir/stall_log.txt (not supported with --parallel's worker processes)
        stall_watchdog=None if args.parallel else StallWatchdog(args.output_dir,
                                                                delegated=args.eval_workers > 1),
        eval_workers=1 if args.parallel else max(1, args.eval_workers),
        horizon_samples=(None if args.prediction_horizon_s is None
                         else max(2, int(round(args.prediction_horizon_s * args.sampling_rate_hz)))),
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
        axis_scaling=args.axis_scaling,
        exclude_axes=tuple(args.exclude_axes),
        drag=args.drag, drag_reference=args.drag_reference, drag_prior_weight=args.drag_prior_weight,
        free_decay=args.free_decay,
        torque_reference_offset_m=(None if args.torque_reference_offset_m is None
                                   else tuple(args.torque_reference_offset_m)),
        current_world_mps=None if args.current_world_mps is None else tuple(args.current_world_mps),
        current_by_file=None if not args.currents_file else read_currents_file(args.currents_file),
        checkpoint_path=None if args.parallel else checkpoint_path,
        sensors=tuple(args.sensors), use_accelerometer=args.use_accelerometer, unseen=args.unseen,
        anchor_rounds=args.anchor_rounds,
        smoother_accel_sigma=args.smoother_accel_sigma,
    )

    try:
        report = run_pipeline(pipeline_config)
    except SteeringError as e:   # a mismatch with a fix, not a failure of the fit
        print(f"❌ {e}", flush=True)
        sys.exit(2)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        note_restart_costs(checkpoint_path, e)
        blew_up = isinstance(e, FloatingPointError) or "NaN/inf" in str(e)
        log_calibration_run(
            script="run_calibration", run_name=f"run_calibration_{_time.strftime('%Y-%m-%d_%H%M')}",
            status=STATUS_FAILED, status_reason=str(e),
            git_commit=_get_git_commit(), vehicle_spec_path=args.spec,
            tags={"run_type": "real"},
        )
        sys.exit(EXIT_BLEW_UP if blew_up else 1)

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
    sys.exit(0 if report.success else 1)


if __name__ == "__main__":
    with keep_awake("stand-in calibration"):  # PROJECT_STATUS.md #49
        main()
