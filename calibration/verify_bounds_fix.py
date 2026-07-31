#!/usr/bin/env python3
"""
Broader verification of the damping-bounds fix (vehicle_spec.yaml's
upper bounds changed from exactly 0 to a proportional small-magnitude
value) - generates MANY random perturbed starting points under the NEW
bounds (not just the one seed=42/restart=0 instance that originally found
the crash) and checks every one against all 25 real training runs.

Usage:
    python verify_bounds_fix.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --n-trials 50
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
            if not np.all(np.isfinite(v)):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-runs-per-trial", type=int, default=None,
                         help="Cap how many of the 25 training runs each trial checks — "
                              "None means all 25 (thorough but slow: this script's manual "
                              "Python loop is unvectorized, ~104M raw function calls at the "
                              "full 50-trial/25-run/4160-step/20-substep scale). Use a smaller "
                              "number for a much faster first pass.")
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=["16-02-2024", "19-02-2024"])
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files]
    if args.max_runs_per_trial is not None:
        train_runs = train_runs[:args.max_runs_per_trial]

    loaded_spec = load_vehicle_spec(args.spec)
    vehicle = loaded_spec.vehicle
    x0_base = initial_parameters(loaded_spec).to_vector()
    lb, ub = parameter_bounds(loaded_spec)
    sim_config = SimulationConfig(dt=1.0 / args.sampling_rate_hz, physics_dt=0.005)

    print(f"Testing {args.n_trials} random perturbed starting points "
          f"(different seeds, NEW bounds) against {len(train_runs)} training run(s)...\n")

    n_crashes = 0
    for trial in range(args.n_trials):
        t0 = __import__("time").monotonic()
        rng = np.random.default_rng(1000 + trial)
        x0 = _initialize_x0(x0_base, lb, ub, rng, "perturbed", 0.1)
        params = CalibrationParameters.from_vector(x0)

        crashed = False
        for run in train_runs:
            if not is_stable(params, vehicle, run, sim_config):
                n_crashes += 1
                crashed = True
                print(f"  Trial {trial}/{args.n_trials}: CRASH on {run.filepath}")
                print(f"    Kp={params.Kp:.4f}, Mq={params.Mq:.4f}, Nr={params.Nr:.4f}, "
                      f"Kpp={params.Kpp:.4f}, Mqq={params.Mqq:.4f}, Nrr={params.Nrr:.4f}")
                break
        elapsed = __import__("time").monotonic() - t0
        if not crashed:
            print(f"  Trial {trial}/{args.n_trials}: stable ({elapsed:.1f}s)")

    print(f"\n{args.n_trials - n_crashes}/{args.n_trials} trials fully stable across all training runs.")
    if n_crashes == 0:
        print("PASS — the bounds fix holds across many random perturbations, not just the one found.")
    else:
        print(f"FAIL — {n_crashes} trial(s) still crashed. The bounds fix needs a wider margin.")


if __name__ == "__main__":
    main()