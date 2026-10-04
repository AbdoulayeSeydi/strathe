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

from data_split import split_by_date, discover_csv_files, DEFAULT_HELDOUT_FRACTION
from dynamics_model import CalibrationParameters
from stall_watchdog import StallWatchdog
from objective import read_currents_file
from objective import (DatasetConfig, RunData, compute_zero_baseline_scales, load_run, mean_run_rmse,
                       rmse_by_dof, skill_vs_zero, zero_baseline_rmse)
from optimizer import (
    LoadedVehicleSpec, load_vehicle_spec, initial_parameters,
    CalibrationConfig, calibrate, calibrate_parallel, parameter_bounds,
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


@dataclass
class ReproducibilitySummary:
    git_commit: Optional[str]
    spec_version: str
    spec_path: str
    data_dir: str
    timestamp_utc: str


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
    objective_torque_reference_offset_m: tuple = (0.0, 0.0, 0.0)
    objective_current_world_mps: Optional[tuple] = None
    objective_held_parameters: Optional[dict] = None
    objective_drag: Optional[dict] = None
    objective_free_decay: Optional[dict] = None
    objective_sampling_rate_hz: float = 10.0
    objective_current_by_file: Optional[tuple] = None


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
    return np.mean([rmse_by_dof(params, run, vehicle, horizon_samples=horizon_samples) for run in runs], axis=0)


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
            "parameters": params_dict,
        }, f, sort_keys=False)

    metrics = {
        "dataset": asdict(report.dataset),
        "timing": asdict(report.timing),
        "optimizer": asdict(report.optimizer),
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
    logger.info("Split: %d train date(s) %s, %d heldout date(s) %s",
                len(split.train_groups), split.train_groups,
                len(split.heldout_groups), split.heldout_groups)

    loaded_spec = load_vehicle_spec(config.spec_path)
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
                                   current_by_file=config.current_by_file)
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

    total_samples = sum(r.n_samples for r in train_runs + heldout_runs)
    logger.info("Loaded %d train run(s), %d heldout run(s), %d total samples (%.2fs)",
                len(train_runs), len(heldout_runs), total_samples, loading_s)

    default_params = initial_parameters(loaded_spec)

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
    if config.calibration_config.horizon_samples is not None:
        logger.info("Objective: re-anchored %d-sample prediction windows (%.1f s)",
                    config.calibration_config.horizon_samples,
                    config.calibration_config.horizon_samples / config.sampling_rate_hz)

    logger.info("Running calibration: %d restart(s), loss=%s, seed=%s",
                config.calibration_config.n_restarts, config.calibration_config.loss,
                config.calibration_config.seed)
    t_cal = time.monotonic()
    if config.parallel:
        cal_result = calibrate_parallel(loaded_spec, train_runs, config.calibration_config,
                                         max_workers=config.max_workers)
    else:
        cal_result = calibrate(loaded_spec, train_runs, config.calibration_config)
    calibration_s = time.monotonic() - t_cal
    logger.info("Calibration converged: %s (%d/%d restarts), cost=%.6g (%.2fs)",
                cal_result.converged, cal_result.stability["n_converged"],
                cal_result.stability["n_total"], cal_result.cost, calibration_s)

    logger.info("Evaluating default vs. calibrated on train and heldout...")
    t_eval = time.monotonic()
    rmse = RmseSummary(
        default_train=mean_run_rmse(default_params, train_runs, vehicle),
        default_heldout=mean_run_rmse(default_params, heldout_runs, vehicle),
        calibrated_train=mean_run_rmse(cal_result.parameters, train_runs, vehicle),
        calibrated_heldout=mean_run_rmse(cal_result.parameters, heldout_runs, vehicle),
    )
    per_dof = PerDofSummary(
        default_train=_average_rmse_by_dof(default_params, train_runs, vehicle),
        default_heldout=_average_rmse_by_dof(default_params, heldout_runs, vehicle),
        calibrated_train=_average_rmse_by_dof(cal_result.parameters, train_runs, vehicle),
        calibrated_heldout=_average_rmse_by_dof(cal_result.parameters, heldout_runs, vehicle),
    )
    per_dof.zero_heldout = zero_baseline_rmse(heldout_runs)
    per_dof.skill_heldout = skill_vs_zero(per_dof.calibrated_heldout, per_dof.zero_heldout)
    short = max(2, int(round(SKILL_EVAL_HORIZON_S * config.sampling_rate_hz)))
    per_dof.skill_heldout_short = skill_vs_zero(
        _average_rmse_by_dof(cal_result.parameters, heldout_runs, vehicle, short), per_dof.zero_heldout)
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
    )

    report = CalibrationReport(
        dataset=DatasetSummary(
            train_dates=split.train_groups, heldout_dates=split.heldout_groups,
            n_train_files=len(train_runs), n_heldout_files=len(heldout_runs),
            total_samples=total_samples, total_trajectories=len(train_runs) + len(heldout_runs),
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

        if r.per_dof.skill_heldout is not None:
            print(f"\nHeld-out skill vs predicting no motion (1 = perfect, <= 0 = no better than zero):")
            print(f"  {'':<6}{'whole run':>11}{f'{SKILL_EVAL_HORIZON_S:g} s windows':>14}")
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
    parser.add_argument("--heldout-dates", nargs="*", default=None,
                         help='e.g. --heldout-dates 16-02-2024 19-02-2024')
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
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            yaml.dump(checkpoint, f, sort_keys=False)

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
    with open(checkpoint_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    params = CalibrationParameters(**data["best_parameters_so_far"])
    print(f"Resuming from checkpoint: call_count={data['call_count']}, "
          f"best_cost_so_far={data['best_cost_so_far']:.6f}")
    return params.to_vector()


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
        x0_override = load_checkpoint_warm_start(args.resume_from_checkpoint)

    if args.parallel:
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
    )

    try:
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
    sys.exit(0 if report.success else 1)


if __name__ == "__main__":
    with keep_awake("stand-in calibration"):  # PROJECT_STATUS.md #49
        main()
