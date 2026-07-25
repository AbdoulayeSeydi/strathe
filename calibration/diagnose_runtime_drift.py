#!/usr/bin/env python3
"""
Tests whether the SAME long-lived HoloOcean environment gets slower over
many repeated calls — the leading hypothesis for the confirmed but
unexplained gap between standalone per-eval cost (124.2s, measured 3x in
a row, consistent) and real optimizer-run cost (~900s/eval, observed
during an actual calibration run). Uses one short, cheap single-run
evaluation repeated MANY times (not the full 25-run set) so we can get
enough data points to see a real trend without waiting hours.

Usage:
    python diagnose_runtime_drift.py --data-dir ../data/raw/Bluerov2_trajectory_dataset --spec vehicle_spec.yaml --sampling-rate-hz 10.0 --warm-start-yaml ../calibration_output_10hz/calibrated_parameters.yaml --n-calls 60
"""
import argparse
import time
import yaml

from data_split import discover_csv_files
from objective import DatasetConfig, load_run, rmse_for_run
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
    parser.add_argument("--warm-start-yaml", required=True)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--n-calls", type=int, default=60)
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    run = truncate(load_run(csv_paths[0], dataset_config), args.n_steps)

    with open(args.warm_start_yaml) as f:
        data = yaml.safe_load(f)
    params = CalibrationParameters(**data["parameters"])

    loaded = load_vehicle_spec(args.spec)
    vehicle = loaded.vehicle

    print("Creating environment (one time)...")
    env = make_deterministic_environment("SimpleUnderwater-Hovering")
    simulate_fn = make_holoocean_simulate_fn(env)

    print(f"Running {args.n_calls} repeated single-run evaluations through the SAME environment...\n")
    times = []
    for i in range(args.n_calls):
        t0 = time.monotonic()
        rmse_for_run(params, run, vehicle, simulate_fn=simulate_fn)
        elapsed = time.monotonic() - t0
        times.append(elapsed)
        print(f"  Call {i+1:3d}: {elapsed:6.3f}s")

    n = len(times)
    first_10_avg = sum(times[:10]) / 10
    last_10_avg = sum(times[-10:]) / 10
    print(f"\nFirst 10 calls avg: {first_10_avg:.3f}s")
    print(f"Last 10 calls avg:  {last_10_avg:.3f}s")
    print(f"Ratio (last/first): {last_10_avg/first_10_avg:.2f}x")
    print(f"\nIf this ratio is close to 1.0, the SAME-environment-degrades-over-time hypothesis")
    print(f"is RULED OUT. If it's meaningfully >1, that's real, confirmed evidence of drift —")
    print(f"and a practical fix (periodically recreate the environment during long runs) follows.")


if __name__ == "__main__":
    main()