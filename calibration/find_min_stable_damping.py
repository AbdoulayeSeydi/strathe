#!/usr/bin/env python3
"""
Empirically finds the minimum stable |Mq|/|Mqq| magnitude for the EXACT
crashing parameter combination found in diagnose_perturbed_crash_detail.py
- holding every other parameter at that same perturbed vector, only
varying pitch damping, to get a REAL, data-grounded bound rather than an
arbitrary round number.

Usage:
    python find_min_stable_damping.py \
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


def is_stable(params, vehicle, run, sim_config):
    n_substeps = sim_config.n_substeps
    sub_dt = sim_config.sub_dt
    v = run.nu0.copy()
    for k in range(run.n_samples - 1):
        tau = run.forces_torques[k]
        roll, pitch = run.roll_pitch[k]
        for _ in range(n_substeps):
            v_dot, _, _ = compute_acceleration(params, vehicle, v, tau, roll, pitch)
            v = v + v_dot * sub_dt
            if not np.all(np.isfinite(v)) or np.max(np.abs(v)) > 50:  # 50 m/s or rad/s is already absurd for this vehicle
                return False
    return True


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
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files]

    loaded_spec = load_vehicle_spec(args.spec)
    vehicle = loaded_spec.vehicle
    x0_base = initial_parameters(loaded_spec).to_vector()
    lb, ub = parameter_bounds(loaded_spec)

    rng = np.random.default_rng(args.seed)
    x0 = _initialize_x0(x0_base, lb, ub, rng, "perturbed", 0.1)
    params = CalibrationParameters.from_vector(x0)
    sim_config = SimulationConfig(dt=1.0 / args.sampling_rate_hz, physics_dt=0.005)  # our confirmed-good default

    print("Sweeping Mq/Mqq magnitude (holding every other perturbed parameter fixed), "
          "checking stability against ALL 25 training runs at each point...\n")

    # Coarse sweep first, from strongly-damped down toward zero, to find where it breaks.
    for mq_frac in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0]:
        params.Mq = -5.0 * mq_frac if mq_frac > 0 else 0.0  # scale toward the STRONG end of Mq's own bounds [-5,0]
        params.Mqq = -10.0 * mq_frac if mq_frac > 0 else 0.0
        all_stable = True
        for run in train_runs:
            if not is_stable(params, vehicle, run, sim_config):
                all_stable = False
                break
        print(f"  Mq={params.Mq:>8.4f}, Mqq={params.Mqq:>8.4f}  ->  "
              f"{'STABLE (all 25 runs)' if all_stable else 'UNSTABLE'}")


if __name__ == "__main__":
    main()