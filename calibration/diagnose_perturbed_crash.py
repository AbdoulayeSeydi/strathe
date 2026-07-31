#!/usr/bin/env python3
"""
Replicates optimizer.py's EXACT perturbation logic (same seed, same
sequential rng calls) to reproduce the REAL 5 starting parameter vectors
--n-restarts 5 --seed 42 actually used - not the nominal default, which
was already confirmed stable across all 25 full-length training runs.

If one of these ACTUALLY-perturbed vectors crashes against real data
where the nominal default didn't, that confirms perturbation (not the
nominal defaults) is the real trigger.

Usage:
    python diagnose_perturbed_crash.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --seed 42 --n-restarts 5
"""
import argparse
import numpy as np

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run
from optimizer import load_vehicle_spec, initial_parameters, parameter_bounds, _initialize_x0
from dynamics_model import CalibrationParameters, SimulationConfig, compute_acceleration


def check_stability(params, vehicle, run, sim_config):
    n_substeps = sim_config.n_substeps
    sub_dt = sim_config.sub_dt
    v = run.nu0.copy()
    for k in range(run.n_samples - 1):
        tau = run.forces_torques[k]
        roll, pitch = run.roll_pitch[k]
        for _ in range(n_substeps):
            v_dot, _, _ = compute_acceleration(params, vehicle, v, tau, roll, pitch)
            v = v + v_dot * sub_dt
            if not np.all(np.isfinite(v)):
                return False, k, tau, v_dot
    return True, None, None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-restarts", type=int, required=True)
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=["16-02-2024", "19-02-2024"])
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files]

    loaded_spec = load_vehicle_spec(args.spec)
    vehicle = loaded_spec.vehicle
    x0_base = initial_parameters(loaded_spec).to_vector()
    lb, ub = parameter_bounds(loaded_spec)

    # EXACT replication of optimizer.py's rng usage: one rng, seeded once,
    # called once per restart IN SEQUENCE - not re-seeded per restart.
    rng = np.random.default_rng(args.seed)
    sim_config = SimulationConfig(dt=1.0 / args.sampling_rate_hz)

    for restart_idx in range(args.n_restarts):
        x0 = _initialize_x0(x0_base, lb, ub, rng, "perturbed", 0.1)
        params = CalibrationParameters.from_vector(x0)
        print(f"\n=== Restart {restart_idx}: checking this ACTUAL perturbed starting point ===")
        print(f"    Ieff_roll={params.Ieff_roll:.3f}, Xu_dot={params.Xu_dot:.3f}, "
              f"Yv_dot={params.Yv_dot:.3f}, Zw_dot={params.Zw_dot:.3f}")

        for run in train_runs:
            stable, step, tau, v_dot = check_stability(params, vehicle, run, sim_config)
            if not stable:
                print(f"  *** CRASHES on {run.filepath} at step {step} ***")
                print(f"      tau: {tau}")
                print(f"      v_dot: {v_dot}")
                return  # stop at first found crash - this is what we needed to know
        print(f"  All 25 training runs stable with this restart's starting point.")

    print("\nNone of the actual perturbed starting points crashed against any training run.")


if __name__ == "__main__":
    main()