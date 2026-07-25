#!/usr/bin/env python3
"""
Stage 2.2 — the sanity check: can PPO learn anything sensible in Arena at
all, before trusting any default-vs-calibrated comparison (Stage 2.4)?

Trains PPO on a SINGLE, easy, fixed waypoint (straight ahead) for a modest
number of timesteps, then compares its trained performance against:
  - a RANDOM baseline (should clearly beat this if anything was learned)
  - the hand-coded proportional-controller positive control (see
    evaluate_policy_arena.py) — a real, known-reasonable target to
    compare against, not just "better than random"

Uses the CALIBRATED (10Hz stand-in) parameters per phase2_config.py's
Decision 2 — training happens on the FAST stand-in, always.

If PPO doesn't clearly beat random here, the problem is almost certainly
the wrapper/reward/hyperparameters, NOT the physics — fix this before ever
attempting Stage 2.4's real default-vs-calibrated comparison.

Usage:
    python train_sanity_check.py \
        --calibrated-params ../calibration_output_10hz/calibrated_parameters.yaml \
        --timesteps 50000
"""
import argparse

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from arena import Arena
from dynamics_model import CalibrationParameters, VehicleConstants
from evaluate_policy_arena import evaluate_policy, proportional_controller


def make_env(params: CalibrationParameters, vehicle: VehicleConstants) -> Arena:
    # Single, easy, FIXED waypoint for this sanity check — deliberately
    # simple (straight ahead, moderate distance) so "can PPO learn
    # anything at all" isn't confounded by task difficulty.
    return Arena(params, vehicle, fixed_waypoint=[2.0, 0.0, 0.0], max_episode_steps=100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-params", required=True)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    args = parser.parse_args()

    with open(args.calibrated_params, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    params = CalibrationParameters(**data["parameters"])
    vehicle = VehicleConstants(mass_kg=11.2)

    print(f"Training PPO for {args.timesteps} timesteps on Arena "
          f"(fast stand-in physics, fixed waypoint [2,0,0])...")

    train_env = Monitor(make_env(params, vehicle))
    model = PPO("MlpPolicy", train_env, verbose=1, seed=0)
    model.learn(total_timesteps=args.timesteps)

    print("\nTraining complete. Evaluating...\n")

    eval_env = make_env(params, vehicle)

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    random_result = evaluate_policy(eval_env, policy_fn=None, n_episodes=args.n_eval_episodes, seed=1)
    prop_result = evaluate_policy(eval_env, policy_fn=proportional_controller, n_episodes=args.n_eval_episodes, seed=1)
    ppo_result = evaluate_policy(eval_env, policy_fn=ppo_policy, n_episodes=args.n_eval_episodes, seed=1)

    print("=" * 70)
    print("SANITY CHECK RESULTS")
    print("=" * 70)
    print(f"{'Policy':<25}{'Mean reward':>15}{'Mean final dist':>18}{'Success rate':>15}")
    for label, r in [("Random baseline", random_result),
                      ("Proportional (hand-coded)", prop_result),
                      ("PPO (trained)", ppo_result)]:
        print(f"{label:<25}{r.mean_reward:>15.2f}{r.mean_final_distance:>18.2f}{r.success_rate:>15.2f}")

    print()
    reward_improvement = ppo_result.mean_reward - random_result.mean_reward
    # Reward is the PRIMARY signal for "did PPO learn anything real" — a
    # continuous measure of progress. Success rate is a stricter, binary
    # bar (fully solving the task within the radius/step limit) and is
    # reported for context, NOT used as a hard gate — a policy can learn
    # real, substantial behavior (large reward improvement) without yet
    # reliably reaching the tight success radius, especially at a modest
    # timestep budget. Conflating these two was a real bug in an earlier
    # version of this script, caught by a real training run where PPO's
    # reward improved dramatically (nearly matching the hand-coded
    # controller) while success_rate stayed at 0.0 for both PPO AND random
    # — the old all-or-nothing check misreported that as a FAIL.
    clearly_learned = reward_improvement > 0.5 * abs(random_result.mean_reward - prop_result.mean_reward)
    print(f"Reward improvement over random: {reward_improvement:+.2f} "
          f"(hand-coded controller's improvement over random, for scale: "
          f"{prop_result.mean_reward - random_result.mean_reward:+.2f})")
    print(f"Success rate — informational only, NOT a pass/fail gate: "
          f"PPO={ppo_result.success_rate:.2f}, random={random_result.success_rate:.2f}")
    if clearly_learned:
        print("\nPASS — PPO is learning something real in this environment "
              "(reward improvement is a substantial fraction of the hand-coded "
              "controller's own improvement over random). Safe to move to Stage 2.3/2.4's "
              "real comparisons.")
    else:
        print("\nFAIL — PPO's reward improvement over random is too small to be confident "
              "it learned real behavior. Do NOT proceed to Stage 2.4 yet — the problem is "
              "almost certainly the wrapper/reward/hyperparameters, not physics. Consider: "
              "more timesteps, reward shaping, checking action/observation scaling.")


if __name__ == "__main__":
    main()