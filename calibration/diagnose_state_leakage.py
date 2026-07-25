#!/usr/bin/env python3
"""
Isolates whether HoloOcean's internal state carries over between separate
simulate_trajectory_holoocean() calls, and whether calling env.reset()
before each one fixes it. Uses ONE short run repeated several times, not
the full 25-run set, to isolate fast.

Usage:
    python diagnose_state_leakage.py --data-dir ../data/raw/Bluerov2_trajectory_dataset --spec vehicle_spec.yaml --sampling-rate-hz 10.0 --warm-start-yaml ../calibration_output_10hz/calibrated_parameters.yaml
"""
import argparse
import time
import yaml
import holoocean

from data_split import discover_csv_files
from objective import DatasetConfig, load_run, rmse_for_run
from optimizer import load_vehicle_spec
from dynamics_model import CalibrationParameters
from holoocean_backend import make_holoocean_simulate_fn


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
    parser.add_argument("--warm-start-yaml", required=True)
    parser.add_argument("--n-steps", type=int, default=150)
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    run = truncate(load_run(csv_paths[0], dataset_config), args.n_steps)
    print(f"Using single run: {run.filepath}, {run.n_samples} steps\n")

    with open(args.warm_start_yaml) as f:
        data = yaml.safe_load(f)
    params = CalibrationParameters(**data["parameters"])

    loaded = load_vehicle_spec(args.spec)
    vehicle = loaded.vehicle

    env = holoocean.make("SimpleUnderwater-Hovering")
    simulate_fn = make_holoocean_simulate_fn(env)

    print("=== WITHOUT env.reset() between calls (current behavior) ===")
    for i in range(4):
        rmse = rmse_for_run(params, run, vehicle, simulate_fn=simulate_fn)
        print(f"  Call {i+1}: RMSE = {rmse:.4f}")

    print("\n=== WITH env.reset() before each call ===")
    for i in range(4):
        env.reset()
        rmse = rmse_for_run(params, run, vehicle, simulate_fn=simulate_fn)
        print(f"  Call {i+1}: RMSE = {rmse:.4f}")

    print("\nIf the WITHOUT group drifts/varies but the WITH group stays consistent,")
    print("env.reset() is the fix — needs adding inside simulate_trajectory_holoocean().")


if __name__ == "__main__":
    main()