#!/usr/bin/env python3
"""
Stage 2.4 — Claim B: multi-seed PPO training, default vs. calibrated
physics, reported as a distribution (not one run per side — that mistake
was already caught once this session, finding #7: a single restart isn't
evidence of anything, only multiple independent attempts agreeing is).

Per phase2_config.py's Decision 2, trains on the FAST stand-in (Arena) —
training cost is separate from and shouldn't wait on the slow HoloOcean
validation backend.

Per Stage 2.1's gap #3, the waypoint is GROUNDED IN REAL DATA — derived
from an actual held-out run's real logged displacement (its x,y,z position
columns), not an arbitrary synthetic target.

Explicitly Claim B — complementary to Stage 2.3's Claim A, NOT a
replacement for it (no live hardware to validate a trained policy's real-
world behavior against yet — that's Phase 2.5's job).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor

from data_split import discover_csv_files, split_by_date
from optimizer import load_vehicle_spec, initial_parameters
from dynamics_model import CalibrationParameters, VehicleConstants
from arena import Arena
from evaluate_policy_arena import evaluate_policy
from phase2_config import PHASE1_HELDOUT_DATES, assert_uses_phase1_split, check_result_type, output_dir_for, Claim
from replay_comparison import load_calibrated_params


def derive_real_waypoint(data_dir: str, heldout_dates: list) -> np.ndarray:
    """
    Stage 2.1's gap #3, addressed: derive Arena's training waypoint from
    an ACTUAL held-out run's real logged displacement (final position -
    initial position, read directly from the CSV's x,y,z columns), rather
    than an arbitrary synthetic target — so the task the policies are
    trained on is grounded in real vehicle behavior, not a made-up number.
    """
    csv_paths, _ = discover_csv_files(data_dir)
    split = split_by_date(csv_paths, heldout_dates=heldout_dates)
    df = pd.read_csv(split.heldout_files[0])
    initial = df[["x", "y", "z"]].iloc[0].to_numpy()
    final = df[["x", "y", "z"]].iloc[-1].to_numpy()
    displacement = final - initial
    print(f"Real-data-grounded waypoint derived from {split.heldout_files[0]}: "
          f"displacement={displacement}")
    return displacement


ALGORITHMS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def train_one_seed(params: CalibrationParameters, vehicle: VehicleConstants,
                    waypoint: np.ndarray, seed: int, timesteps: int,
                    n_eval_episodes: int, task: str = "waypoint",
                    algorithm: str = "ppo") -> dict:
    def make_env():
        return Arena(params, vehicle, fixed_waypoint=waypoint, max_episode_steps=100, task=task)

    train_env = Monitor(make_env())
    algo_cls = ALGORITHMS[algorithm]
    model = algo_cls("MlpPolicy", train_env, verbose=0, seed=seed)
    model.learn(total_timesteps=timesteps)

    eval_env = make_env()

    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    result = evaluate_policy(eval_env, policy_fn=policy_fn, n_episodes=n_eval_episodes, seed=seed)
    return {"seed": seed, "mean_reward": result.mean_reward,
            "mean_final_distance": result.mean_final_distance,
            "success_rate": result.success_rate,
            "collision_rate": result.collision_rate,
            "timeout_rate": result.timeout_rate,
            "mean_episode_length": result.mean_episode_length,
            "mean_time_to_goal": result.mean_time_to_goal,
            "mean_min_obstacle_clearance": result.mean_min_obstacle_clearance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--calibrated-params", required=True)
    parser.add_argument("--result-type", required=True, choices=["placeholder", "final"])
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=150000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", default="train_multiseed")
    parser.add_argument("--task", default="waypoint",
                         choices=["waypoint", "station_keeping", "docking", "obstacle_avoidance"],
                         help="Which Arena task to train/evaluate on (item #12). Defaults "
                              "to 'waypoint', matching all of this session's existing "
                              "real Claim B results - changing this changes what the "
                              "reported reward numbers actually mean, so be deliberate "
                              "about it, not incidental. NOTE: 'obstacle_avoidance' "
                              "generates its OWN randomized goal every episode and does "
                              "NOT use the real-data-grounded waypoint below - see the "
                              "printed warning at runtime.")
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "sac", "td3"],
                         help="Which SB3 algorithm to train with. Defaults to 'ppo', "
                              "matching all of this session's existing real Claim B "
                              "results. 'sac'/'td3' are off-policy algorithms with a "
                              "replay buffer - see PROJECT_STATUS.md for their real, "
                              "confirmed learning_starts/buffer_size defaults and the "
                              "validation gates run before trusting them.")
    args = parser.parse_args()

    check_result_type(args.result_type, metrics_path=args.metrics_path)
    output_dir = output_dir_for(args.output_dir, args.result_type)

    assert_uses_phase1_split(PHASE1_HELDOUT_DATES)  # sanity: constants haven't drifted
    waypoint = derive_real_waypoint(args.data_dir, PHASE1_HELDOUT_DATES)
    if args.task == "obstacle_avoidance":
        print("NOTE: --task obstacle_avoidance generates its own randomized goal every "
              "episode (see ObstacleAvoidanceTask.reset()) - the real-data-grounded "
              "waypoint just derived above will be computed but NOT actually used for "
              "this task. This is intentional (obstacle avoidance needs to control "
              "spawn/goal/obstacle together for a valid, solvable episode), not a bug - "
              "flagged here so it's not a silent surprise.")

    loaded_spec = load_vehicle_spec(args.spec)
    vehicle = loaded_spec.vehicle
    default_params = initial_parameters(loaded_spec)
    calibrated_params = load_calibrated_params(args.calibrated_params)

    results = {"default": [], "calibrated": []}
    for variant, params in [("default", default_params), ("calibrated", calibrated_params)]:
        print(f"\n{'='*70}\nTraining {args.n_seeds} seeds — {variant} physics\n{'='*70}")
        for seed in range(args.n_seeds):
            print(f"  Seed {seed}...")
            r = train_one_seed(params, vehicle, waypoint, seed, args.timesteps,
                               args.n_eval_episodes, task=args.task, algorithm=args.algorithm)
            clearance_str = (f", clearance={r['mean_min_obstacle_clearance']:.2f}"
                              if r['mean_min_obstacle_clearance'] is not None else "")
            print(f"    reward={r['mean_reward']:.2f}, "
                  f"distance={r['mean_final_distance']:.2f}, success={r['success_rate']:.2f}, "
                  f"collision={r['collision_rate']:.2f}, timeout={r['timeout_rate']:.2f}, "
                  f"ep_len={r['mean_episode_length']:.1f}{clearance_str}")
            results[variant].append(r)

    def summarize(variant_results):
        rewards = [r["mean_reward"] for r in variant_results]
        successes = [r["success_rate"] for r in variant_results]
        collisions = [r["collision_rate"] for r in variant_results]
        timeouts = [r["timeout_rate"] for r in variant_results]
        ep_lengths = [r["mean_episode_length"] for r in variant_results]
        clearances = [r["mean_min_obstacle_clearance"] for r in variant_results
                      if r["mean_min_obstacle_clearance"] is not None]
        return {
            "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
            "mean_success_rate": float(np.mean(successes)), "std_success_rate": float(np.std(successes)),
            "mean_collision_rate": float(np.mean(collisions)),
            "mean_timeout_rate": float(np.mean(timeouts)),
            "mean_episode_length": float(np.mean(ep_lengths)),
            "mean_min_obstacle_clearance": float(np.mean(clearances)) if clearances else None,
            "per_seed": variant_results,
        }

    summary = {v: summarize(results[v]) for v in ["default", "calibrated"]}

    print("\n" + "=" * 70)
    print(Claim.b_trained_policy_behavior(args.algorithm))
    print("=" * 70)
    for v in ["default", "calibrated"]:
        s = summary[v]
        clearance_str = (f", clearance={s['mean_min_obstacle_clearance']:.2f}"
                          if s['mean_min_obstacle_clearance'] is not None else "")
        print(f"{v:<12} reward={s['mean_reward']:.2f}±{s['std_reward']:.2f}, "
              f"success={s['mean_success_rate']:.2f}±{s['std_success_rate']:.2f}, "
              f"collision={s['mean_collision_rate']:.2f}, timeout={s['mean_timeout_rate']:.2f}, "
              f"ep_len={s['mean_episode_length']:.1f}{clearance_str} "
              f"(n={args.n_seeds} seeds)")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    variants = ["default", "calibrated"]
    means = [summary[v]["mean_reward"] for v in variants]
    stds = [summary[v]["std_reward"] for v in variants]
    ax.bar(variants, means, yerr=stds, capsize=8, color=["#c0392b", "#27ae60"])
    ax.set_ylabel("Mean episode reward")
    ax.set_title(f"Claim B: {args.algorithm.upper()} reward, default vs. calibrated physics "
                 f"(n={args.n_seeds} seeds each, {args.result_type})")
    fig.tight_layout()
    chart_path = out / "claim_b_reward_comparison.png"
    fig.savefig(chart_path, dpi=150)
    print(f"\nSaved chart: {chart_path}")

    report = {
        "claim": Claim.b_trained_policy_behavior(args.algorithm),
        "algorithm": args.algorithm,
        "result_type": args.result_type,
        "n_seeds": args.n_seeds,
        "timesteps_per_seed": args.timesteps,
        "waypoint": waypoint.tolist(),
        "summary": summary,
    }
    with open(out / "claim_b_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report: {out / 'claim_b_report.json'}")


if __name__ == "__main__":
    main()