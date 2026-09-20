#!/usr/bin/env python3
"""
evaluate_baseline_sanity_mujoco.py -- the missing half of C0.3 (see
conversation). The original task named two things to repeat on the
MuJoCo backend: "A1.9 (baseline sanity check), then A1.10
(comm-degradation sweep)". evaluate_comm_degradation_curve_mujoco.py
covers the second; this covers the first, which was skipped.

A1.9's own evaluate_policy() constructs its environment internally
(build_zero_degradation_env(...), a module-level function with no
dynamics_backend parameter) -- unlike A1.10's run_episode(), which
takes an already-built env, this can't be reused unchanged. This file
duplicates ONLY that one construction, adding dynamics_backend="mujoco";
random_policy_fn and _print_eval are reused unchanged via import, since
neither constructs an environment.

Deliberately zero comm degradation, matching A1.9's own explicit scope
("comm degradation OFF... purely a sanity check") -- NOT the L0-L3a
severities A1.10 already covers. This is a different axis
(dynamics backend) from A1.10's own axis (comm severity): does each
already-trained policy still clearly beat a random baseline when
evaluated on real MuJoCo dynamics instead of the stand-in, everything
else held at A1.9's own zero-degradation baseline.

Per Decision 2 (phase2_config.py), no retraining happens here -- same
already-trained models A1.10 used, evaluation only.

C0.5 addition (see conversation): logs each evaluation (random baseline
+ one per trained model) to MLflow via log_fleet_run() -- comm_severity_
level="L0" always (this file's own explicit zero-degradation scope),
dynamics_backend="mujoco" always.

Usage:
    python3 evaluate_baseline_sanity_mujoco.py \
        --model-paths formation_hold_ppo_curriculum_v5 \
                       formation_hold_ppo_curriculum_L1 \
                       formation_hold_ppo_curriculum_L2 \
                       formation_hold_ppo_curriculum_L3a \
        --labels L0 L1 L2 L3a
"""
import argparse
from collections import Counter

import numpy as np

from two_agent_formation_env import TwoAgentFormationEnv, DEFAULT_ACTION_BOUND
from formation_hold_task import TERMINAL_REASON_FORMATION_ACHIEVED, TERMINAL_REASON_COLLISION, FormationHoldConfig
from comm_channel import PacketLossConfig, BandwidthConfig
from two_agent_orchestrator import OrchestratorConfig
from train_formation_baseline import random_policy_fn, _print_eval
from mlflow_run_utils import get_git_commit, compute_per_agent_rewards
from evaluate_comm_degradation_curve import VEHICLE_SPEC_PATH
import mlflow_logging

CALIBRATED_PARAMS_PATH = "calibrated_parameters_a1.yaml"


def build_zero_degradation_env_mujoco(max_episode_steps: int = 200) -> TwoAgentFormationEnv:
    """Mirrors train_formation_baseline.py's own build_zero_degradation_env
    exactly, with dynamics_backend="mujoco" added -- the ONLY intentional
    difference."""
    return TwoAgentFormationEnv(
        calibrated_params_path=CALIBRATED_PARAMS_PATH,
        packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.0, 0.0),
        bandwidth_config=BandwidthConfig(bytes_per_tick=1e9),
        randomize_initial_state=True,
        orchestrator_config=OrchestratorConfig(max_episode_steps=max_episode_steps),
        action_bound=DEFAULT_ACTION_BOUND,
        dynamics_backend="mujoco",
    )


def evaluate_policy_mujoco(policy_fn, n_episodes: int, max_episode_steps: int, seed: int) -> dict:
    """Mirrors train_formation_baseline.py's own evaluate_policy exactly
    -- only the env construction call differs. C0.5 addition: also
    returns reward_generic/reward_calibrated, same decomposition as the
    stand-in version."""
    target_separation = FormationHoldConfig().target_separation
    env = build_zero_degradation_env_mujoco(max_episode_steps=max_episode_steps)
    reward_ctrl_weight = env._orchestrator.task.config.reward_ctrl_weight
    rng = np.random.default_rng(seed)

    episode_rewards, episode_lengths = [], []
    final_separation_errors, min_separations, max_consecutive_in_tolerance = [], [], []
    component_sums = {"separation": [], "control": [], "proximity": [], "collision": [], "success_bonus": []}
    terminal_reasons = []
    reward_generic_means, reward_calibrated_means = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total_reward = 0.0
        steps = 0
        terminal_reason = None
        min_sep = float("inf")
        max_consec = 0
        this_episode_components = {k: 0.0 for k in component_sums}
        last_separation = None
        reward_generic_sum, reward_calibrated_sum = 0.0, 0.0

        for _ in range(max_episode_steps):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            r_g, r_c = compute_per_agent_rewards(reward, action[:6], action[6:], reward_ctrl_weight)
            reward_generic_sum += r_g
            reward_calibrated_sum += r_c

            sep = info.get("true_separation")
            if sep is not None:
                min_sep = min(min_sep, sep)
                last_separation = sep
            max_consec = max(max_consec, info.get("consecutive_steps_in_tolerance", 0))
            comps = info.get("reward_components", {})
            for k in this_episode_components:
                this_episode_components[k] += comps.get(k, 0.0)

            if terminated:
                terminal_reason = info.get("terminal_reason")
                break
            if truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        min_separations.append(min_sep)
        max_consecutive_in_tolerance.append(max_consec)
        for k in component_sums:
            component_sums[k].append(this_episode_components[k])
        if last_separation is not None:
            final_separation_errors.append(abs(last_separation - target_separation))
        terminal_reasons.append(terminal_reason if terminal_reason else "timeout")
        reward_generic_means.append(reward_generic_sum / steps)
        reward_calibrated_means.append(reward_calibrated_sum / steps)

    reason_counts = Counter(terminal_reasons)
    successes = reason_counts.get(TERMINAL_REASON_FORMATION_ACHIEVED, 0)
    collisions = reason_counts.get(TERMINAL_REASON_COLLISION, 0)
    timeouts = reason_counts.get("timeout", 0)

    return {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "success_rate": successes / n_episodes,
        "collision_rate": collisions / n_episodes,
        "timeout_rate": timeouts / n_episodes,
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_final_separation_error": float(np.mean(final_separation_errors)) if final_separation_errors else None,
        "mean_min_separation": float(np.mean(min_separations)),
        "mean_max_consecutive_steps_in_tolerance": float(np.mean(max_consecutive_in_tolerance)),
        "reward_component_means": {k: float(np.mean(v)) for k, v in component_sums.items()},
        "terminal_reason_counts": dict(reason_counts),
        "reward_generic": float(np.mean(reward_generic_means)),
        "reward_calibrated": float(np.mean(reward_calibrated_means)),
    }


def main(model_paths: list, labels: list, eval_episodes: int, seed: int):
    from stable_baselines3 import PPO
    from validate_policy_compatibility import validate_all_policies, PolicyIncompatibleError

    # F3.3: pre-flight check before any compute, matching preflight_check.py's precedent.
    print("[pre-flight] Validating policy compatibility...")
    try:
        validate_all_policies(model_paths, algorithm="ppo")
    except PolicyIncompatibleError as e:
        raise SystemExit(f"Policy compatibility check failed:\n{e}") from e

    git_commit = get_git_commit()
    env_for_space = build_zero_degradation_env_mujoco()
    print(f"Evaluating RANDOM baseline on MuJoCo backend ({eval_episodes} episodes, "
          f"shared across all models below for a fair comparison)...")
    random_result = evaluate_policy_mujoco(
        random_policy_fn(env_for_space.action_space), eval_episodes, max_episode_steps=200, seed=seed)
    _print_eval("RANDOM BASELINE (MuJoCo backend)", random_result)
    mlflow_logging.log_fleet_run(
        script="evaluate_baseline_sanity_mujoco.py",
        run_name="random_baseline_mujoco",
        status="SUCCESS",
        status_reason="random baseline evaluation completed on MuJoCo backend",
        git_commit=git_commit,
        vehicle_spec_path=VEHICLE_SPEC_PATH,
        comm_severity_level="L0",
        dynamics_backend="mujoco",
        coordination_success_rate=random_result["success_rate"],
        reward_generic=random_result["reward_generic"],
        reward_calibrated=random_result["reward_calibrated"],
        params={"n_agents": 2},
        metrics={"collision_rate": random_result["collision_rate"],
                 "mean_max_consecutive_in_tolerance": random_result["mean_max_consecutive_steps_in_tolerance"]},
        tags={"policy_type": "random"},
    )

    for label, path in zip(labels, model_paths):
        print(f"\n{'=' * 70}\n{label}: {path}\n{'=' * 70}")
        model = PPO.load(path)

        def trained_policy_fn(obs, _model=model):
            action, _ = _model.predict(obs, deterministic=True)
            return action

        trained_result = evaluate_policy_mujoco(
            trained_policy_fn, eval_episodes, max_episode_steps=200, seed=seed)
        _print_eval(f"{label} TRAINED POLICY (MuJoCo backend)", trained_result)
        mlflow_logging.log_fleet_run(
            script="evaluate_baseline_sanity_mujoco.py",
            run_name=f"{label}_trained_mujoco",
            status="SUCCESS",
            status_reason=f"trained-model evaluation on MuJoCo backend, model={path}",
            git_commit=git_commit,
            vehicle_spec_path=VEHICLE_SPEC_PATH,
            comm_severity_level="L0",
            dynamics_backend="mujoco",
            coordination_success_rate=trained_result["success_rate"],
            reward_generic=trained_result["reward_generic"],
            reward_calibrated=trained_result["reward_calibrated"],
            params={"n_agents": 2, "trained_model_label": label, "trained_model_path": path},
            metrics={"collision_rate": trained_result["collision_rate"],
                     "mean_max_consecutive_in_tolerance": trained_result["mean_max_consecutive_steps_in_tolerance"]},
            tags={"policy_type": "trained"},
        )

        print(f"\n  COMPARISON ({label}, MuJoCo backend):")
        print(f"    success_rate:  random={random_result['success_rate']:.0%}  "
              f"trained={trained_result['success_rate']:.0%}")
        print(f"    mean_reward:   random={random_result['mean_reward']:.3f}  "
              f"trained={trained_result['mean_reward']:.3f}")
        print(f"    mean_max_consecutive_in_tolerance: random="
              f"{random_result['mean_max_consecutive_steps_in_tolerance']:.1f}  "
              f"trained={trained_result['mean_max_consecutive_steps_in_tolerance']:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-paths", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True,
                         help="Display label per model, same order as --model-paths")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert len(args.model_paths) == len(args.labels), \
        "--model-paths and --labels must be the same length, in matching order"
    main(args.model_paths, args.labels, args.eval_episodes, args.seed)
