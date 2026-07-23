#!/usr/bin/env python3
"""
The real stress test: runs simulate_trajectory_holoocean() against REAL
HoloOcean with the DEFAULT (uncalibrated) parameters on ONE real logged
run, and compares it against the stand-in model's result for the SAME
run/params — the two backends should produce qualitatively similar
(not identical — this backend is a genuine forward simulation, not
teacher-forced) trajectories if everything is wired correctly.

This is what actually tests the assumptions the mock environment (used
during development) couldn't catch — particularly set_physics_state's
angular_velocity frame convention, which was assumed rather than
independently confirmed against real HoloOcean.

Usage:
    python3 stress_test_holoocean.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --run odom-12-01-2024-run9.csv \
        --n-steps 50
"""
import argparse
from pathlib import Path

import numpy as np
import holoocean

from data_split import discover_csv_files
from objective import DatasetConfig, load_run
from optimizer import load_vehicle_spec, initial_parameters
from dynamics_model import SimulationConfig, TrajectoryInputs, simulate_trajectory
from holoocean_backend import simulate_trajectory_holoocean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--run", required=True, help="Filename (or substring) of the run to test")
    parser.add_argument("--n-steps", type=int, default=50,
                         help="Truncate to this many steps — real HoloOcean stepping is much "
                              "slower than the analytical stand-in, keep this small for a first test")
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    match = next((p for p in csv_paths if args.run in p), None)
    if match is None:
        raise SystemExit(f"No run matching '{args.run}' found under {args.data_dir}")

    print(f"Loading: {match}")
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    run = load_run(match, dataset_config)

    loaded_spec = load_vehicle_spec(args.spec)
    params = initial_parameters(loaded_spec)  # DEFAULT (uncalibrated) params for this first test
    vehicle = loaded_spec.vehicle

    n = min(args.n_steps, run.n_samples)
    trajectory = TrajectoryInputs(
        nu0=run.nu0,
        forces_torques=run.forces_torques[:n],
        roll_pitch_series=run.roll_pitch[:n],
    )
    sim_config = SimulationConfig(dt=run.dt)

    print(f"\nRunning STAND-IN model for {n} steps...")
    standin_result = simulate_trajectory(params, vehicle, sim_config, trajectory)
    print("Stand-in done.")

    print(f"\nCreating HoloOcean environment (this takes real time — Unreal Engine startup)...")
    env = holoocean.make("SimpleUnderwater-Hovering")
    print("Environment created. Running HoloOcean-backed simulation...")

    holoocean_result = simulate_trajectory_holoocean(params, vehicle, sim_config, trajectory, env)
    print("HoloOcean-backed simulation done.")

    print("\n" + "=" * 70)
    print("COMPARISON: real logged data vs. stand-in vs. HoloOcean-backed")
    print("=" * 70)
    labels = ["u", "v", "w", "p", "q", "r"]
    print(f"{'DOF':<6}{'real[final]':>14}{'stand-in[final]':>18}{'holoocean[final]':>18}")
    for i, lbl in enumerate(labels):
        print(f"{lbl:<6}{run.real_nu[n-1, i]:>14.4f}{standin_result.nu[-1, i]:>18.4f}{holoocean_result.nu[-1, i]:>18.4f}")

    print("\nWhat to look for:")
    print("- HoloOcean's numbers should be in the SAME BALLPARK as the stand-in's, not wildly")
    print("  different (e.g. not off by 10x, not opposite sign, not all zero/frozen)")
    print("- They do NOT need to match exactly — HoloOcean does a genuine forward simulation")
    print("  (compounding its own integration), the stand-in uses teacher-forced orientation")
    print("- If HoloOcean's numbers look nonphysical (huge, NaN, all zero), that points at a")
    print("  frame-convention bug — most likely set_physics_state's angular_velocity assumption")


if __name__ == "__main__":
    main()