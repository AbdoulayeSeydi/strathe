#!/usr/bin/env python3
"""
Stage 2.3 — Claim A: the rigorous, direct MVP acceptance criterion.

Replays REAL held-out logged thruster commands OPEN-LOOP through both
DEFAULT and CALIBRATED physics, computes RMSE against real logged
trajectories for both, and produces the comparison chart the MVP plan
literally asks for.

This is NOT new simulation logic — it reuses objective.py's
already-validated rmse_by_dof()/rmse_for_run() directly (the same
machinery every Phase 1 finding is built on), just pointed at the SAME
held-out split (phase2_config.PHASE1_HELDOUT_DATES, enforced, not
re-decided) with two different parameter sets.

Per phase2_config.py's Decision 2, uses the REAL HoloOcean backend
(VALIDATION_BACKEND) — this is Claim A, the rigorous one, and deserves the
real physics from the start, not the fast stand-in.

Usage:
    python replay_comparison.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --calibrated-params ../calibration_output_holoocean_poc/checkpoint.yaml \
        --result-type placeholder \
        --output-dir replay_comparison
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display available on a headless/remote run
import matplotlib.pyplot as plt
import yaml

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, rmse_by_dof, rmse_for_run, compute_dof_scales
from optimizer import load_vehicle_spec, initial_parameters
from dynamics_model import CalibrationParameters
from phase2_config import (
    PHASE1_HELDOUT_DATES, assert_uses_phase1_split, check_result_type,
    output_dir_for, Claim,
)

DOF_LABELS = ["u", "v", "w", "p", "q", "r"]


def load_calibrated_params(path: str) -> CalibrationParameters:
    """Handles BOTH real output formats seen this session: a
    calibrated_parameters.yaml (top-level 'parameters' key) and a
    checkpoint.yaml (top-level 'best_parameters_so_far' key)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "parameters" in data:
        return CalibrationParameters(**data["parameters"])
    if "best_parameters_so_far" in data:
        return CalibrationParameters(**data["best_parameters_so_far"])
    raise ValueError(f"{path} doesn't look like either a calibrated_parameters.yaml "
                      f"or a checkpoint.yaml — no 'parameters' or 'best_parameters_so_far' key found.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--calibrated-params", required=True)
    parser.add_argument("--result-type", required=True, choices=["placeholder", "final"])
    parser.add_argument("--metrics-path", default=None,
                         help="Required if --result-type final — the source calibration's "
                              "own metrics.json, checked for converged=True before this script "
                              "will produce a report labeled final.")
    parser.add_argument("--use-holoocean", action="store_true", default=True)
    parser.add_argument("--no-holoocean", dest="use_holoocean", action="store_false",
                         help="Use the fast stand-in instead of real HoloOcean — for a quick "
                              "pipeline smoke test only, NOT for an actual Claim A result "
                              "(Decision 2: validation should use HoloOcean).")
    parser.add_argument("--output-dir", default="replay_comparison")
    args = parser.parse_args()

    # The safeguard — fails loudly before doing any real (possibly
    # expensive) work if this is mislabeled.
    check_result_type(args.result_type, metrics_path=args.metrics_path)
    output_dir = output_dir_for(args.output_dir, args.result_type)

    csv_paths, duplicates = discover_csv_files(args.data_dir)
    for dup_path, kept_path in duplicates:
        print(f"Duplicate filename skipped: {dup_path} (kept {kept_path})")

    split = split_by_date(csv_paths, heldout_dates=PHASE1_HELDOUT_DATES)
    assert_uses_phase1_split(split.heldout_groups)  # fails loudly if this ever drifts
    print(f"Held-out dates (Phase 1's exact split, enforced): {split.heldout_groups}")

    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    heldout_runs = [load_run(fp, dataset_config) for fp in split.heldout_files]
    print(f"Loaded {len(heldout_runs)} held-out runs, "
          f"{sum(r.n_samples for r in heldout_runs)} total steps")

    # dof_scales MUST be computed from TRAINING data, never held-out — see
    # compute_dof_scales' own docstring: using held-out data here would
    # leak held-out statistics into what's supposed to be an out-of-sample
    # check. Loaded here specifically for this, not used for anything else
    # in this script.
    train_runs_for_scales = [load_run(fp, dataset_config) for fp in split.train_files]
    dof_scales = compute_dof_scales(train_runs_for_scales)
    print(f"Per-DOF scales (std of REAL logged velocity, TRAINING data only): "
          f"{dict(zip(DOF_LABELS, [round(float(s), 4) for s in dof_scales]))}")

    loaded_spec = load_vehicle_spec(args.spec)
    vehicle = loaded_spec.vehicle
    default_params = initial_parameters(loaded_spec)
    calibrated_params = load_calibrated_params(args.calibrated_params)

    if args.use_holoocean:
        print("\nUsing REAL HoloOcean physics (Decision 2 — validation backend)...")
        from holoocean_backend import make_holoocean_simulate_fn, make_deterministic_environment
        env = make_deterministic_environment("SimpleUnderwater-Hovering")
        simulate_fn = make_holoocean_simulate_fn(env)
    else:
        print("\nWARNING: using the fast stand-in, NOT HoloOcean — pipeline smoke test only, "
              "not a real Claim A result (see Decision 2 in phase2_config.py).")
        simulate_fn = None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "replay_checkpoint.json"

    def evaluate_with_progress(params, label):
        """Evaluates one parameter set across all held-out runs, printing
        per-run progress and checkpointing after EVERY run — this script
        previously had NEITHER, a real gap given every real HoloOcean-
        backed cost this session has repeatedly exceeded standalone
        estimates. A crash/interruption now only loses the current run's
        partial progress, not the whole evaluation."""
        per_run_dof = []
        per_run_overall = []
        for i, run in enumerate(heldout_runs):
            t0 = __import__("time").monotonic()
            dof = rmse_by_dof(params, run, vehicle, simulate_fn=simulate_fn)
            # overall RMSE derived from per-DOF RMSE via sqrt(mean(dof**2))
            # — mathematically EXACT (confirmed via direct numeric test,
            # not assumed) whenever every DOF shares the same sample count,
            # which is always true here (same run, same length). NOT a
            # second call to rmse_for_run, which would silently re-run the
            # full (expensive) simulation a second time for no reason — a
            # real bug in an earlier version of this script that roughly
            # DOUBLED the true cost of every real HoloOcean-backed run.
            overall = float(np.sqrt(np.mean(dof ** 2)))
            elapsed = __import__("time").monotonic() - t0
            per_run_dof.append(dof)
            per_run_overall.append(overall)
            print(f"  [{label}] run {i+1}/{len(heldout_runs)} "
                  f"({Path(run.filepath).name}, {run.n_samples} steps): "
                  f"RMSE={overall:.4f} ({elapsed:.1f}s)")
            checkpoint = {
                "label": label, "completed_runs": i + 1, "total_runs": len(heldout_runs),
                "per_run_overall_so_far": [float(x) for x in per_run_overall],
            }
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2)
        return np.array(per_run_dof), per_run_overall

    print("\nEvaluating DEFAULT parameters on held-out runs...")
    default_dof, default_overall = evaluate_with_progress(default_params, "default")

    print("\nEvaluating CALIBRATED parameters on held-out runs...")
    calibrated_dof, calibrated_overall = evaluate_with_progress(calibrated_params, "calibrated")

    mean_default_dof = default_dof.mean(axis=0)
    mean_calibrated_dof = calibrated_dof.mean(axis=0)
    mean_default_overall = float(np.mean(default_overall))
    mean_calibrated_overall = float(np.mean(calibrated_overall))
    improvement_pct_naive = (mean_default_overall - mean_calibrated_overall) / mean_default_overall * 100

    # SCALE-NORMALIZED aggregate — divides each DOF's RMSE by its own
    # training-derived scale (dof_scales) BEFORE averaging across DOFs,
    # using the SAME compute_dof_scales() machinery objective.py's own
    # optimizer already relies on for this exact reason. The naive
    # unweighted average above lets whichever DOF has the largest raw
    # magnitude dominate regardless of physical importance — rotational
    # DOFs (rad/s) are numerically much larger than translational ones
    # (m/s) here, so a naive average can show an "improvement" that's
    # actually a real, substantial translational win being swamped by a
    # known, separate rotational weakness (see PROJECT_STATUS.md findings
    # #5, #10, #12, #13). This metric puts all 6 DOFs on comparable
    # footing before combining them, giving a precise, not-misleading
    # headline number.
    normalized_default_dof = mean_default_dof / dof_scales
    normalized_calibrated_dof = mean_calibrated_dof / dof_scales
    mean_default_normalized = float(np.mean(normalized_default_dof))
    mean_calibrated_normalized = float(np.mean(normalized_calibrated_dof))
    improvement_pct_normalized = (
        (mean_default_normalized - mean_calibrated_normalized) / mean_default_normalized * 100
    )

    print("\n" + "=" * 70)
    print(f"{Claim.A_OPEN_LOOP_REPLAY_ACCURACY}")
    print("=" * 70)
    print(f"NAIVE overall RMSE (unweighted across DOFs — can be dominated by whichever")
    print(f"  DOF has the largest raw magnitude, NOT a precise headline number):")
    print(f"  default: {mean_default_overall:.4f}, calibrated: {mean_calibrated_overall:.4f}, "
          f"improvement: {improvement_pct_naive:+.1f}%")
    print(f"\nSCALE-NORMALIZED overall RMSE (each DOF divided by its own training-derived")
    print(f"  scale before averaging — puts m/s and rad/s DOFs on comparable footing;")
    print(f"  THIS is the precise, not-misleading headline number):")
    print(f"  default: {mean_default_normalized:.4f}, calibrated: {mean_calibrated_normalized:.4f}, "
          f"improvement: {improvement_pct_normalized:+.1f}%")
    print(f"\nPer-DOF RMSE (raw units):")
    print(f"{'DOF':<6}{'Default':>12}{'Calibrated':>14}{'Improvement':>14}")
    for i, label in enumerate(DOF_LABELS):
        dof_improvement = (mean_default_dof[i] - mean_calibrated_dof[i]) / mean_default_dof[i] * 100
        print(f"{label:<6}{mean_default_dof[i]:>12.4f}{mean_calibrated_dof[i]:>14.4f}{dof_improvement:>13.1f}%")

    # The MVP's literal stated acceptance criterion: "a single comparison
    # chart (trajectory overlay, or RMSE bar chart)".
    fig, (ax_dof, ax_summary) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})

    x = np.arange(len(DOF_LABELS))
    width = 0.35
    ax_dof.bar(x - width/2, mean_default_dof, width, label="Default physics", color="#c0392b")
    ax_dof.bar(x + width/2, mean_calibrated_dof, width, label="Calibrated physics", color="#27ae60")
    ax_dof.set_xticks(x)
    ax_dof.set_xticklabels(DOF_LABELS)
    ax_dof.set_ylabel("RMSE (natural units)")
    ax_dof.set_title("Per-DOF RMSE (raw units — u,v,w in m/s, p,q,r in rad/s)")
    ax_dof.legend()

    # The summary panel exists specifically so the chart itself — not just
    # console text — makes clear the naive aggregate can be misleading
    # (dominated by whichever DOF has the largest raw magnitude) and the
    # scale-normalized one is the precise headline number.
    summary_x = np.arange(2)
    ax_summary.bar(summary_x - width/2, [mean_default_overall, mean_default_normalized],
                    width, label="Default", color="#c0392b")
    ax_summary.bar(summary_x + width/2, [mean_calibrated_overall, mean_calibrated_normalized],
                    width, label="Calibrated", color="#27ae60")
    ax_summary.set_xticks(summary_x)
    ax_summary.set_xticklabels(["Naive\n(unweighted)", "Scale-normalized\n(precise headline)"])
    ax_summary.set_title("Aggregate metrics — see labels\nfor which is misleading")
    ax_summary.legend()

    fig.suptitle(f"Claim A: default vs. calibrated physics — held-out real data ({args.result_type})")
    fig.tight_layout()
    chart_path = out / "claim_a_rmse_comparison.png"
    fig.savefig(chart_path, dpi=150)
    print(f"\nSaved chart: {chart_path}")

    report = {
        "claim": Claim.A_OPEN_LOOP_REPLAY_ACCURACY,
        "result_type": args.result_type,
        "used_holoocean": args.use_holoocean,
        "heldout_dates": split.heldout_groups,
        "n_heldout_runs": len(heldout_runs),
        "dof_scales_from_training_data": {label: float(v) for label, v in zip(DOF_LABELS, dof_scales)},
        "naive_unweighted": {
            "note": "Can be dominated by whichever DOF has the largest raw magnitude — "
                    "NOT the precise headline number, kept for reference/comparability only.",
            "mean_rmse_default": mean_default_overall,
            "mean_rmse_calibrated": mean_calibrated_overall,
            "improvement_pct": improvement_pct_naive,
        },
        "scale_normalized": {
            "note": "Each DOF divided by its own training-derived scale (compute_dof_scales) "
                    "before averaging — puts m/s and rad/s DOFs on comparable footing. "
                    "THIS is the precise, not-misleading headline number.",
            "mean_rmse_default": mean_default_normalized,
            "mean_rmse_calibrated": mean_calibrated_normalized,
            "improvement_pct": improvement_pct_normalized,
        },
        "per_dof_default": {label: float(v) for label, v in zip(DOF_LABELS, mean_default_dof)},
        "per_dof_calibrated": {label: float(v) for label, v in zip(DOF_LABELS, mean_calibrated_dof)},
    }
    with open(out / "claim_a_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report: {out / 'claim_a_report.json'}")


if __name__ == "__main__":
    main()