#!/usr/bin/env python3
"""
Re-validates a checkpoint's best-parameters-so-far under the CORRECTED
physics (after the set_physics_state degrees/sec unit fix) — the existing
checkpoint's stored cost was computed under the OLD, slightly-wrong
initial-angular-velocity handling, so resuming from it directly would mix
an old, no-longer-accurate cost with new, correct physics going forward.

This does NOT throw away the real optimization progress already made —
it re-evaluates the SAME best parameters (found via ~40+ minutes of real
HoloOcean-backed optimization) under the now-fixed backend, computes their
TRUE current cost, and writes a fresh, internally-consistent checkpoint to
resume from. The parameters themselves aren't discarded, only re-validated.

Usage:
    python revalidate_checkpoint.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --checkpoint-path ../calibration_output_holoocean_poc/checkpoint.yaml \
        --n-steps-per-run 150
"""
import argparse
import numpy as np
import yaml

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, joint_residuals
from optimizer import load_vehicle_spec
from dynamics_model import CalibrationParameters
from holoocean_backend import make_holoocean_simulate_fn, make_deterministic_environment


def truncate(run, n):
    n = min(n, run.n_samples)
    return type(run)(filepath=run.filepath, forces_torques=run.forces_torques[:n],
                      roll_pitch=run.roll_pitch[:n], real_nu=run.real_nu[:n], nu0=run.nu0,
                      dt=run.dt, sample_rate_hz=run.sample_rate_hz, date=run.date)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--n-steps-per-run", type=int, default=150)
    args = parser.parse_args()

    with open(args.checkpoint_path, encoding="utf-8") as f:
        checkpoint = yaml.safe_load(f)

    old_cost = checkpoint["best_cost_so_far"]
    params = CalibrationParameters(**checkpoint["best_parameters_so_far"])
    print(f"Loaded checkpoint: old (pre-fix) best_cost_so_far = {old_cost:.6f}, "
          f"from call_count={checkpoint['call_count']}\n")

    csv_paths, duplicates = discover_csv_files(args.data_dir)
    for dup_path, kept_path in duplicates:
        print(f"Duplicate filename skipped: {dup_path} (kept {kept_path})")
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [truncate(load_run(fp, dataset_config), args.n_steps_per_run)
                  for fp in split.train_files]

    loaded_spec = load_vehicle_spec(args.spec)
    vehicle = loaded_spec.vehicle

    print("Creating environment (with the FIXED set_physics_state units)...")
    env = make_deterministic_environment("SimpleUnderwater-Hovering")
    simulate_fn = make_holoocean_simulate_fn(env)

    print("Re-evaluating the checkpoint's best parameters under the FIXED physics...")
    param_vec = params.to_vector()
    # Match run_calibration_holoocean.py's CalibrationConfig exactly:
    # dof_scales=None, normalize_by_run_length=True (both defaults)
    residuals = joint_residuals(param_vec, train_runs, vehicle,
                                 dof_scales=None, normalize_by_run_length=True,
                                 simulate_fn=simulate_fn)
    new_cost = 0.5 * float(np.sum(residuals ** 2))  # matches scipy's own cost convention

    print(f"\nOld (pre-fix) cost:  {old_cost:.6f}")
    print(f"New (post-fix) cost: {new_cost:.6f}")
    print(f"Difference: {new_cost - old_cost:+.6f} ({(new_cost - old_cost) / old_cost * 100:+.2f}%)")
    print("\nAs expected, this should be small — the fix only affects the very first instant "
          "of each 150-step run, and real logged forces dominate almost immediately after.\n")

    new_checkpoint = {
        "call_count": checkpoint["call_count"],
        "current_cost": new_cost,
        "current_parameters": checkpoint["best_parameters_so_far"],
        "best_cost_so_far": new_cost,
        "best_parameters_so_far": checkpoint["best_parameters_so_far"],
        "note": "Re-validated under the corrected set_physics_state degrees/sec fix — "
                f"cost updated from the pre-fix value ({old_cost:.6f}) to the true, "
                f"post-fix value ({new_cost:.6f}). Parameters unchanged.",
    }
    with open(args.checkpoint_path, "w", encoding="utf-8") as f:
        yaml.dump(new_checkpoint, f, sort_keys=False)
    print(f"Wrote corrected, internally-consistent checkpoint to {args.checkpoint_path}")
    print("Safe to resume from this checkpoint now.")


if __name__ == "__main__":
    main()