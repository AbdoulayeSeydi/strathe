"""
trace_episode.py — runs ONE deterministic episode with an already-trained
model and prints per-step separation/consecutive-count/action-magnitude,
so "why did final_separation_error get so much worse while min_separation
and max_consecutive both improved" can be answered by actually looking at
the trajectory, not guessed at from aggregate statistics.

Specifically built to test one concrete hypothesis: FormationHoldTask's
consecutive-steps counter is a HARD reset to zero on any single
out-of-tolerance step (not a gradual decay) — see formation_hold_task.py's
step(): "if within_tolerance and not collision: += 1 else: = 0". If the
policy gets close to the 20-step hold, drops out on one bad step, and then
fails to re-approach cleanly for the rest of a 200-step episode, that
would directly explain this exact pattern: a near-miss on
max_consecutive_in_tolerance alongside a terrible final_separation_error.
"""
import argparse

import numpy as np

from two_agent_formation_env import TwoAgentFormationEnv
from comm_channel import PacketLossConfig, BandwidthConfig
from two_agent_orchestrator import OrchestratorConfig


def build_env(max_episode_steps=200):
    return TwoAgentFormationEnv(
        calibrated_params_path="calibrated_parameters_a1.yaml",
        packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.0, 0.0),
        bandwidth_config=BandwidthConfig(bytes_per_tick=1e9),
        orchestrator_config=OrchestratorConfig(max_episode_steps=max_episode_steps),
    )


def main(model_path: str, seed: int, print_every: int):
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    env = build_env()
    obs, _ = env.reset(seed=seed)

    print(f"{'step':>5} {'separation':>11} {'consec':>7} {'action_norm':>12} "
          f"{'reward':>9} {'cum_reward':>11} {'terminal_reason':>16}")

    cum_reward = 0.0
    max_consec_seen = 0
    step_of_max_consec = 0
    last_reset_step = None

    for step in range(1, 201):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward

        sep = info["true_separation"]
        consec = info["consecutive_steps_in_tolerance"]
        action_norm = float(np.linalg.norm(action))
        reason = info.get("terminal_reason", "")

        if consec > max_consec_seen:
            max_consec_seen = consec
            step_of_max_consec = step
        if consec == 0 and step > 1:
            last_reset_step = step

        if step % print_every == 0 or terminated or truncated or step <= 5:
            print(f"{step:>5} {sep:>11.3f} {consec:>7} {action_norm:>12.3f} "
                  f"{reward:>9.3f} {cum_reward:>11.2f} {reason:>16}")

        if terminated or truncated:
            break

    print(f"\nMax consecutive-in-tolerance reached: {max_consec_seen} "
          f"(at step {step_of_max_consec})")
    print(f"Last reset-to-zero of the counter: step {last_reset_step}")
    print(f"Final separation: {sep:.3f} (target was 2.0)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True,
                         help="e.g. formation_hold_ppo_post_ctrl_fix_400k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=5)
    args = parser.parse_args()
    main(args.model_path, args.seed, args.print_every)