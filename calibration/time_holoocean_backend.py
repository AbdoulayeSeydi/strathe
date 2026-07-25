#!/usr/bin/env python3
"""
Measures real per-step cost of the HoloOcean-backed simulate_fn against
the analytical stand-in, on a small truncated slice of real data, then
extrapolates to what a full calibration run would actually cost — before
committing to running one.

Usage:
    python time_holoocean_backend.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --n-steps-timed 20 \
        --n-runs-timed 3
"""
import argparse
import time
from pathlib import Path

import numpy as np
import holoocean

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, joint_residuals
from optimizer import load_vehicle_spec, initial_parameters
from dynamics_model import TrajectoryInputs
from holoocean_backend import make_holoocean_simulate_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--n-steps-timed", type=int, default=20,
                         help="Truncate each timed run to this many steps — keep small, "
                              "real HoloOcean stepping is the expensive part")
    parser.add_argument("--n-runs-timed", type=int, default=3,
                         help="How many runs to include in the timed joint_residuals call")
    parser.add_argument("--assumed-evals-per-restart", type=int, default=40,
                         help="For the extrapolation estimate — past real single-restart runs "
                              "converged in 33-59 evaluations, 40 is a reasonable middle estimate")
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)

    all_train_runs = [load_run(fp, dataset_config) for fp in split.train_files]
    avg_full_run_length = np.mean([r.n_samples for r in all_train_runs])
    n_train_runs = len(all_train_runs)

    loaded_spec = load_vehicle_spec(args.spec)
    params = initial_parameters(loaded_spec)
    vehicle = loaded_spec.vehicle

    # Build TRUNCATED versions of a few real runs for the actual timed test —
    # keep this small since HoloOcean stepping is the expensive unknown.
    timed_runs = []
    for run in all_train_runs[:args.n_runs_timed]:
        n = min(args.n_steps_timed, run.n_samples)
        truncated = type(run)(
            filepath=run.filepath, forces_torques=run.forces_torques[:n],
            roll_pitch=run.roll_pitch[:n], real_nu=run.real_nu[:n], nu0=run.nu0,
            dt=run.dt, sample_rate_hz=run.sample_rate_hz, date=run.date,
        )  # n_samples is auto-computed in __post_init__ from forces_torques.shape[0] — NOT a constructor arg
        timed_runs.append(truncated)

    print(f"Timing against {len(timed_runs)} run(s), {args.n_steps_timed} steps each\n")

    param_vec = params.to_vector()

    print("=== Timing the STAND-IN (baseline) ===")
    t0 = time.monotonic()
    _ = joint_residuals(param_vec, timed_runs, vehicle)
    standin_elapsed = time.monotonic() - t0
    print(f"Stand-in: {standin_elapsed:.4f}s total, "
          f"{standin_elapsed / (len(timed_runs) * args.n_steps_timed) * 1000:.4f}ms/step\n")

    print("=== Creating HoloOcean environment (one-time cost, not counted in per-step timing) ===")
    t0 = time.monotonic()
    env = holoocean.make("SimpleUnderwater-Hovering")
    env_creation_time = time.monotonic() - t0
    print(f"Environment creation took {env_creation_time:.1f}s (one-time only, NOT repeated per evaluation)\n")

    simulate_fn = make_holoocean_simulate_fn(env)

    print("=== Timing the HOLOOCEAN-BACKED simulate_fn ===")
    t0 = time.monotonic()
    _ = joint_residuals(param_vec, timed_runs, vehicle, simulate_fn=simulate_fn)
    holoocean_elapsed = time.monotonic() - t0
    per_step_ms = holoocean_elapsed / (len(timed_runs) * args.n_steps_timed) * 1000
    print(f"HoloOcean-backed: {holoocean_elapsed:.4f}s total, {per_step_ms:.4f}ms/step\n")

    slowdown = holoocean_elapsed / standin_elapsed if standin_elapsed > 0 else float("inf")
    print("=" * 70)
    print("EXTRAPOLATION TO A FULL CALIBRATION RUN")
    print("=" * 70)
    print(f"HoloOcean is {slowdown:.1f}x slower than the stand-in per step (this measurement)")
    print(f"Real dataset: {n_train_runs} training runs, average {avg_full_run_length:.0f} steps/run")

    per_eval_seconds = per_step_ms / 1000 * avg_full_run_length * n_train_runs
    print(f"\nEstimated cost of ONE full joint_residuals() evaluation "
          f"({n_train_runs} runs x ~{avg_full_run_length:.0f} steps): {per_eval_seconds:.1f}s "
          f"(~{per_eval_seconds/60:.1f} min)")

    for n_evals in [args.assumed_evals_per_restart, args.assumed_evals_per_restart * 5]:
        total_seconds = per_eval_seconds * n_evals
        label = "single restart" if n_evals == args.assumed_evals_per_restart else "5 restarts"
        print(f"Estimated total for ~{n_evals} evaluations ({label}): "
              f"{total_seconds/60:.1f} min ({total_seconds/3600:.2f} hours)")

    print("\nCaveat: this is a rough linear extrapolation from a small, short-run sample —")
    print("real per-step cost may not scale perfectly linearly (Unreal Engine warm-up/caching")
    print("effects, memory growth over a long run, etc. aren't captured by this short test).")
    print("Treat this as a planning estimate, not a guarantee.")


if __name__ == "__main__":
    main()