#!/usr/bin/env python3
"""
finalize_checkpoint.py -- turn an interrupted or stopped calibration into a
usable result: the best parameters in its checkpoint.yaml, evaluated on the
training and held-out runs, written in the same layout run_calibration.py
writes (calibrated_parameters.yaml + metrics.json), so calibration_report.py
and everything else can use it.

    python finalize_checkpoint.py --checkpoint ../calibration_output_x/checkpoint.yaml \
        --data-dir <logs> --heldout-dates 16-02-2024 19-02-2024 \
        [--axis-scaling zero_rms] [--exclude-axes p q] [--reason "stopped at a plateau"]

The result is always marked as NOT converged (metrics.json optimizer.converged
= false, with the reason and the evaluation count), and the report says so.
Pass the same objective options the run used; they are recorded, not re-run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import yaml

from data_split import discover_csv_files, split_by_date
from dynamics_model import CalibrationParameters
from objective import read_currents_file
from objective import (DatasetConfig, load_run, mean_run_rmse, rmse_by_dof, skill_vs_zero, stable_mean_run_rmse,
                       stable_rmse_by_dof, zero_baseline_rmse)
from optimizer import initial_parameters, load_vehicle_spec
from run_calibration import (SKILL_EVAL_HORIZON_S, _get_git_commit, excluded_axis_setup, free_decay_setup,
                             spec_torque_reference_offset)


def _per_dof(params, runs, vehicle, horizon=None):
    return np.mean([rmse_by_dof(params, r, vehicle, horizon_samples=horizon) for r in runs], axis=0)


def finalize(checkpoint: str, data_dir: str, spec_path: str, heldout_dates, out_dir: str, sampling_rate_hz: float,
             torque_offset=None, axis_scaling: str = "none", exclude_axes=(), horizon_samples=None,
             reason: str = "", current_world=None, current_by_file=None, free_decay=None) -> dict:
    with open(checkpoint, encoding="utf-8") as f:
        ck = yaml.safe_load(f)
    params = CalibrationParameters(**ck["best_parameters_so_far"])
    spec = load_vehicle_spec(spec_path)
    offset = tuple(torque_offset) if torque_offset is not None else spec_torque_reference_offset(spec)
    if current_world is None:
        from current import current_from_spec
        current_world = current_from_spec(spec)
    current_world = None if current_world is None or not np.any(current_world) else tuple(map(float, current_world))
    dc = DatasetConfig(sampling_rate_hz, offset, current_world, current_by_file)
    split = split_by_date(discover_csv_files(data_dir)[0], heldout_dates=heldout_dates)
    train = [load_run(f, dc) for f in split.train_files]
    held = [load_run(f, dc) for f in split.heldout_files]
    default = initial_parameters(spec)
    v = spec.vehicle
    zero = zero_baseline_rmse(held)
    cal_held, diverged_full = stable_rmse_by_dof(params, held, v)  # a blow-up is reported, not a crash
    short = max(2, int(round(SKILL_EVAL_HORIZON_S * sampling_rate_hz)))
    cal_short, diverged_short = stable_rmse_by_dof(params, held, v, short)
    n_evals = None
    timing = os.path.join(os.path.dirname(checkpoint), "eval_timing.csv")
    if os.path.exists(timing):
        with open(timing, encoding="utf-8") as f:
            n_evals = sum(1 for _ in f) - 1
    held_params = excluded_axis_setup(exclude_axes, None, default)[1] if exclude_axes else None
    free_decay_info = free_decay_setup(free_decay, spec, default) if free_decay else None
    if free_decay_info:  # the fit held these, so the checkpoint has them already; recorded for the report
        held_params = {**(held_params or {}), **free_decay_info["held"]}
    rmse = {"default_train": mean_run_rmse(default, train, v), "default_heldout": mean_run_rmse(default, held, v),
            "calibrated_train": stable_mean_run_rmse(params, train, v)[0],
            "calibrated_heldout": stable_mean_run_rmse(params, held, v)[0]}
    metrics = {
        "dataset": {"train_dates": split.train_groups, "heldout_dates": split.heldout_groups,
                    "n_train_files": len(train), "n_heldout_files": len(held)},
        "optimizer": {"converged": False, "cost": ck.get("best_cost_so_far"), "nfev": n_evals,
                      "message": f"finalized from checkpoint, not formally converged: {reason or 'run stopped'}"},
        "rmse": rmse,
        "per_dof": {"default_train": _per_dof(default, train, v).tolist(),
                    "default_heldout": _per_dof(default, held, v).tolist(),
                    "calibrated_train": stable_rmse_by_dof(params, train, v)[0].tolist(),
                    "calibrated_heldout": cal_held.tolist()},
        "skill_vs_zero": {"zero_heldout": zero.tolist(),
                          "calibrated_heldout_full_replay": skill_vs_zero(cal_held, zero).tolist(),
                          "calibrated_heldout_short_windows": skill_vs_zero(cal_short, zero).tolist(),
                          "short_window_s": SKILL_EVAL_HORIZON_S,
                          "diverged_heldout": {"full_replay": diverged_full, "short_windows": diverged_short,
                                               "runs": len(held)},
                          "meaning": "1 - RMSE/RMSE_zero per axis [u,v,w,p,q,r]; <= 0 = no better than "
                                     "predicting no motion"},
        "objective": {"axis_scaling": axis_scaling, "prediction_horizon_samples": horizon_samples,
                      "excluded_axes": list(exclude_axes), "torque_reference_offset_m": list(offset),
                      "current_world_mps": None if current_world is None else list(current_world),
                      "held_parameters": held_params, "free_decay": free_decay_info,
                      "sampling_rate_hz": sampling_rate_hz,
                      "current_by_file": None if not current_by_file else [[f, list(c)] for f, c in current_by_file]},
        "reproducibility": {"git_commit": _get_git_commit(), "spec_path": spec_path, "data_dir": data_dir,
                            "checkpoint": os.path.abspath(checkpoint),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat()},
        "success": False,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "calibrated_parameters.yaml"), "w", encoding="utf-8") as f:
        yaml.dump({"generated_at_utc": metrics["reproducibility"]["timestamp_utc"],
                   "spec_version": spec.raw.get("schema_version", "unknown"),
                   "git_commit": metrics["reproducibility"]["git_commit"],
                   "note": metrics["optimizer"]["message"] + (f" ({n_evals} evaluations)" if n_evals else ""),
                   "parameters": {n: float(getattr(params, n)) for n in params.param_names}}, f, sort_keys=False)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--spec", default="vehicle_spec.yaml")
    ap.add_argument("--heldout-dates", nargs="+", default=None,
                    help="default: the latest 30%% of recording dates (at least one), as run_calibration.py does")
    ap.add_argument("--output-dir", help="default: the checkpoint's folder")
    ap.add_argument("--sampling-rate-hz", type=float, default=10.0)
    ap.add_argument("--torque-reference-offset-m", type=float, nargs=3, default=None)
    ap.add_argument("--axis-scaling", choices=["none", "zero_rms"], default="none")
    ap.add_argument("--exclude-axes", nargs="+", choices=["p", "q"], default=[])
    ap.add_argument("--prediction-horizon-samples", type=int, default=None)
    ap.add_argument("--current-world-mps", type=float, nargs=3, default=None)
    ap.add_argument("--currents-file", default=None, help="currents.csv: a current per recording")
    ap.add_argument("--free-decay", default=None, help="the free_decay.yaml the fit held (run_calibration.py)")
    ap.add_argument("--reason", default="")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    m = finalize(args.checkpoint, args.data_dir, args.spec, args.heldout_dates, out, args.sampling_rate_hz,
                 args.torque_reference_offset_m, args.axis_scaling, tuple(args.exclude_axes),
                 args.prediction_horizon_samples, args.reason, args.current_world_mps,
                 None if not args.currents_file else read_currents_file(args.currents_file), args.free_decay)
    sk = m["skill_vs_zero"]["calibrated_heldout_full_replay"]
    print(f"Wrote {out}/calibrated_parameters.yaml and metrics.json (NOT converged: {m['optimizer']['message']})")
    print("Held-out skill vs predicting no motion [u v w p q r]: " + " ".join(f"{x:+.2f}" for x in sk))
    dv = m["skill_vs_zero"]["diverged_heldout"]
    if dv["full_replay"] or dv["short_windows"]:
        print(f"⚠️  With these values the simulation diverges on {dv['full_replay']} of {dv['runs']} held-out "
              f"recordings (whole-run replay) and {dv['short_windows']} ({SKILL_EVAL_HORIZON_S:g} s windows); the "
              f"skill above is over the rest. Don't simulate with this calibration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
