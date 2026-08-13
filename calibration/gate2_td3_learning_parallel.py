#!/usr/bin/env python3
"""
Parallel version of gate2_td3_learning.py - runs the 4 independent seeds
as separate OS processes instead of sequentially, given 16 real CPU
cores confirmed available and no GPU. Reuses the exact same real
train_one_seed() and the exact same locked Phase B criteria - this is
purely a wall-clock speedup, not a change to what's being tested or the
bar it's tested against.

Applies the same real lessons already learned tonight building
calibrate_parallel():
  - Worker function is MODULE-LEVEL (not a closure/lambda) - required
    for Windows' 'spawn' multiprocessing start method to pickle it
  - Each worker forces single-threaded BLAS/PyTorch internally, BEFORE
    any tensor operation - otherwise each of the 4 processes independently
    tries to use every CPU core for its own internal linear algebra,
    causing severe oversubscription that can make parallel SLOWER than
    sequential (the exact trap found and fixed earlier this session)
  - MUST be run as a saved script with the __name__ == "__main__" guard
    below, never via `python -c` - Windows' spawn re-executes the whole
    calling script inside each worker, which without this guard can
    cause runaway recursive process spawning

Run from calibration/:
    python gate2_td3_learning_parallel.py
"""
import inspect
import json
import os

import numpy as np


def _train_one_seed_worker(seed, params, vehicle, waypoint, timesteps,
                            n_eval_episodes, task, algorithm):
    # Single-threading FIRST, before torch/numpy do anything in this
    # process - see module docstring for why this is required, not optional.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    from train_multiseed import train_one_seed
    print(f"  Worker starting seed {seed} (PID={os.getpid()})...", flush=True)
    result = train_one_seed(params, vehicle, waypoint, seed, timesteps,
                             n_eval_episodes, task=task, algorithm=algorithm)
    print(f"  Worker finished seed {seed}: reward={result['mean_reward']:.2f}", flush=True)
    return result


def main():
    from stable_baselines3 import TD3
    from train_multiseed import derive_real_waypoint
    from phase2_config import PHASE1_HELDOUT_DATES
    from optimizer import load_vehicle_spec, initial_parameters

    ALGORITHM = "td3"
    TASK = "waypoint"
    N_SEEDS = 4
    LEARNING_BUDGET = 150000
    N_EVAL_EPISODES = 20

    RANDOM_FLOOR = -223.60
    PPO_REFERENCE = -151.41

    loaded = load_vehicle_spec("vehicle_spec.yaml")
    vehicle = loaded.vehicle
    params = initial_parameters(loaded)
    waypoint = derive_real_waypoint("../data/raw/Bluerov2_trajectory_dataset", PHASE1_HELDOUT_DATES)

    sig = inspect.signature(TD3.__init__)
    hyperparameters = {
        "algorithm": ALGORITHM,
        "learning_rate": sig.parameters["learning_rate"].default,
        "learning_starts": sig.parameters["learning_starts"].default,
        "buffer_size": sig.parameters["buffer_size"].default,
        "total_timesteps": LEARNING_BUDGET,
        "task": TASK,
    }

    print("=" * 70)
    print(f"GATE 2 (LEARNING) - {ALGORITHM.upper()}, task={TASK}, n_seeds={N_SEEDS} (PARALLEL)")
    print("=" * 70)
    print(f"Locked criteria: random floor >= {RANDOM_FLOOR}, PPO reference >= {PPO_REFERENCE}")
    print(f"Hyperparameters (real, confirmed SB3 defaults): {hyperparameters}\n")
    print(f"Starting {N_SEEDS} parallel worker processes...\n")

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=N_SEEDS) as executor:
        futures = [
            executor.submit(_train_one_seed_worker, seed, params, vehicle, waypoint,
                             LEARNING_BUDGET, N_EVAL_EPISODES, TASK, ALGORITHM)
            for seed in range(N_SEEDS)
        ]
        seed_results = [f.result() for f in futures]

    seed_results.sort(key=lambda r: r["seed"])
    for r in seed_results:
        r["hyperparameters"] = dict(hyperparameters, seed=r["seed"])

    rewards = [r["mean_reward"] for r in seed_results]
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))

    print("\n" + "=" * 70)
    print("GATE 2 RESULTS - PER SEED (not just aggregate)")
    print("=" * 70)
    for r in seed_results:
        print(f"  seed {r['seed']}: reward={r['mean_reward']:.2f}, "
              f"success_rate={r['success_rate']:.2f}, distance={r['mean_final_distance']:.2f}")
    print(f"\nAggregate: mean={mean_reward:.2f}, std={std_reward:.2f} (n={N_SEEDS} seeds)")

    stability_ok = all(np.isfinite(r["mean_reward"]) and np.isfinite(r["mean_final_distance"])
                        for r in seed_results)
    passes_random_floor = mean_reward >= RANDOM_FLOOR
    passes_ppo_reference = mean_reward >= PPO_REFERENCE

    print(f"\n[{'PASS' if passes_random_floor else 'FAIL'}] Random floor "
          f"(mean_reward >= {RANDOM_FLOOR}): {mean_reward:.2f}")
    print(f"[{'PASS' if passes_ppo_reference else 'FAIL'}] PPO reference "
          f"(mean_reward >= {PPO_REFERENCE}): {mean_reward:.2f}")
    print(f"[{'PASS' if stability_ok else 'FAIL'}] Stability (no NaN/Inf in any seed)")

    gate2_passed = passes_random_floor and passes_ppo_reference and stability_ok
    print(f"\n{'GATE 2 PASSED' if gate2_passed else 'GATE 2 FAILED'}")

    report = {
        "algorithm": ALGORITHM, "task": TASK, "n_seeds": N_SEEDS,
        "hyperparameters": hyperparameters,
        "criteria": {"random_floor": RANDOM_FLOOR, "ppo_reference": PPO_REFERENCE},
        "per_seed": seed_results, "mean_reward": mean_reward, "std_reward": std_reward,
        "gate2_passed": gate2_passed, "ran_in_parallel": True,
    }
    with open("gate2_td3_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nSaved: gate2_td3_report.json")


if __name__ == "__main__":
    main()