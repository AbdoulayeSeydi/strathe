    #!/usr/bin/env python3
"""
Gate 2 (Learning) for TD3 - per the frozen plan. Reuses the REAL, already
Gate-1-verified train_one_seed() from train_multiseed.py directly (not a
reimplementation) - it already performs deterministic evaluation
(predict(obs, deterministic=True), confirmed in the real file), so this
script's only job is running it across real multiple seeds and checking
the result against Phase B's locked-in criteria.

Locked criteria (Phase B, from two FRESH real reference runs this
session - see PROJECT_STATUS.md):
    random floor:   mean_reward >= -223.60  (20% of the random-to-PPO gap closed)
    PPO reference:  mean_reward >= -151.41  (50% of the random-to-PPO gap closed)
    stability:      no NaN/Inf in any seed's result

Learning budget: 150,000 timesteps (Phase C - matches PPO's own real
training scale exactly, not a scaled-down smoke-test number).
n_seeds: 4 (Stage 2.4's own precedent).

Run from calibration/:
    python gate2_td3_learning.py
"""
import inspect
import json

import numpy as np
from stable_baselines3 import TD3

from train_multiseed import train_one_seed, derive_real_waypoint
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

# Real hyperparameters actually in effect (Phase A confirmed these ARE the
# real SB3 defaults - train_one_seed() overrides nothing except seed).
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
print(f"GATE 2 (LEARNING) - {ALGORITHM.upper()}, task={TASK}, n_seeds={N_SEEDS}")
print("=" * 70)
print(f"Locked criteria: random floor >= {RANDOM_FLOOR}, PPO reference >= {PPO_REFERENCE}")
print(f"Hyperparameters (real, confirmed SB3 defaults): {hyperparameters}\n")

seed_results = []
for seed in range(N_SEEDS):
    print(f"Seed {seed}...")
    r = train_one_seed(params, vehicle, waypoint, seed, LEARNING_BUDGET,
                        N_EVAL_EPISODES, task=TASK, algorithm=ALGORITHM)
    r["hyperparameters"] = dict(hyperparameters, seed=seed)
    seed_results.append(r)
    print(f"  reward={r['mean_reward']:.2f}, success_rate={r['success_rate']:.2f}, "
          f"distance={r['mean_final_distance']:.2f}")

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

if not gate2_passed:
    print("\nGate 2 failed - actor_loss/critic_loss training-curve inspection is the "
          "next real diagnostic step (per the frozen plan, a diagnostic aid only, not "
          "a pass/fail criterion itself). Not run automatically here since it requires "
          "configuring SB3's logger before training - rerun with logging enabled if "
          "this happens.")

report = {
    "algorithm": ALGORITHM,
    "task": TASK,
    "n_seeds": N_SEEDS,
    "hyperparameters": hyperparameters,
    "criteria": {"random_floor": RANDOM_FLOOR, "ppo_reference": PPO_REFERENCE},
    "per_seed": seed_results,
    "mean_reward": mean_reward,
    "std_reward": std_reward,
    "gate2_passed": gate2_passed,
}
with open("gate2_td3_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=str)
print("\nSaved: gate2_td3_report.json")