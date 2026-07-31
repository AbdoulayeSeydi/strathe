#!/usr/bin/env python3
"""
Follow-up to diagnose_perturbed_crash.py: reproduces the EXACT crashing
(perturbed parameter vector, real run) combination found there, prints
the FULL parameter vector (not just 4 fields), and tests whether further
tightening physics_dt resolves this specific case - the same class of fix
as before, or something that needs more than just finer resolution.

Usage:
    python diagnose_perturbed_crash_detail.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --seed 42
"""
import argparse
import numpy as np

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run
from optimizer import load_vehicle_spec, initial_parameters, parameter_bounds, _initialize_x0
from dynamics_model import CalibrationParameters, SimulationConfig, compute_acceleration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=["16-02-2024", "19-02-2024"])
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    run = load_run([f for f in split.train_files if "12-01-2024-run1.csv" in f][0], dataset_config)

    loaded_spec = load_vehicle_spec(args.spec)
    vehicle = loaded_spec.vehicle
    x0_base = initial_parameters(loaded_spec).to_vector()
    lb, ub = parameter_bounds(loaded_spec)

    rng = np.random.default_rng(args.seed)
    x0 = _initialize_x0(x0_base, lb, ub, rng, "perturbed", 0.1)  # restart 0's actual vector
    params = CalibrationParameters.from_vector(x0)

    print("FULL perturbed parameter vector for restart 0:")
    for name in params.param_names:
        nominal = CalibrationParameters.from_vector(x0_base)
        print(f"  {name:<18}{getattr(nominal, name):>10.4f} -> {getattr(params, name):>10.4f}")

    print(f"\nTesting how much sub-stepping resolution this SPECIFIC combination needs...")
    for physics_dt in [0.005, 0.002, 0.001, 0.0005, 0.0002]:
        sim_config = SimulationConfig(dt=1.0 / args.sampling_rate_hz, physics_dt=physics_dt)
        n_substeps = sim_config.n_substeps
        sub_dt = sim_config.sub_dt
        v = run.nu0.copy()
        crashed_at = None
        for k in range(run.n_samples - 1):
            tau = run.forces_torques[k]
            roll, pitch = run.roll_pitch[k]
            for _ in range(n_substeps):
                v_dot, _, _ = compute_acceleration(params, vehicle, v, tau, roll, pitch)
                v = v + v_dot * sub_dt
                if not np.all(np.isfinite(v)):
                    crashed_at = k
                    break
            if crashed_at is not None:
                break
        if crashed_at is None:
            print(f"  physics_dt={physics_dt}: OK (n_substeps={n_substeps}), max|v|={np.max(np.abs(v)):.2f}")
        else:
            print(f"  physics_dt={physics_dt}: CRASHED at step {crashed_at} (n_substeps={n_substeps})")


if __name__ == "__main__":
    main()