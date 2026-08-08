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
from objective import DatasetConfig, RunData, load_run, mean_run_rmse, rmse_by_dof
from optimizer import (
    LoadedVehicleSpec, load_vehicle_spec, initial_parameters,
    CalibrationConfig, calibrate,
)

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


def _average_rmse_by_dof(params, runs: list, vehicle) -> np.ndarray:
    return np.mean([rmse_by_dof(params, run, vehicle) for run in runs], axis=0)


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
    dataset_config = DatasetConfig(sampling_rate_hz=config.sampling_rate_hz)
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

    logger.info("Running calibration: %d restart(s), loss=%s, seed=%s",
                config.calibration_config.n_restarts, config.calibration_config.loss,
                config.calibration_config.seed)
    t_cal = time.monotonic()
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
    evaluation_s = time.monotonic() - t_eval
    total_s = time.monotonic() - t_start

    scipy_result = cal_result.scipy_result
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

    calibration_config = CalibrationConfig(
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
    )

    try:
        report = run_pipeline(pipeline_config)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)

    ConsoleReporter().report(report)
    sys.exit(0 if report.success else 1)


if __name__ == "__main__":
    main()