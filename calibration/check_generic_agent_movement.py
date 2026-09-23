"""
check_generic_agent_movement.py -- verifies whether the generic agent
has actually been stationary or actively moving under PPO's own learned
action, for a real v5 trajectory. Every PPO evaluation this session
(evaluate_recovery_grid.py, check_original_scenario.py, the six-axis
diagnostics) calls model.predict(obs) for the FULL 12-dim action and
passes it straight to env.step() -- nothing zeros the generic half.
"Generic held stationary" was only ever true for the hand-crafted
heuristic scripts (alternative_action explicitly zeros it), not for any
PPO evaluation. This checks which is actually true empirically, since
it's load-bearing for how A1.10's generic-vs-calibrated metrics should
be defined.
"""
import argparse

import numpy as np

from evaluate_recovery import build_env, reset_to_custom_state


def main(model_path: str, d0: float, v0: float, seed: int, max_steps: int):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    env = build_env()

    obs = reset_to_custom_state(env, d0, v0, seed)
    generic_positions = []
    calibrated_positions = []
    terminal_reason = None

    for step in range(1, max_steps + 1):
        core_g = env._orchestrator.core_generic
        core_c = env._orchestrator.core_calibrated
        generic_positions.append(core_g.state.position.copy())
        calibrated_positions.append(core_c.state.position.copy())

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            terminal_reason = info.get("terminal_reason")
            break
        if truncated:
            break

    generic_positions = np.array(generic_positions)
    calibrated_positions = np.array(calibrated_positions)

    generic_displacement = np.linalg.norm(generic_positions - generic_positions[0], axis=1)
    calibrated_displacement = np.linalg.norm(calibrated_positions - calibrated_positions[0], axis=1)

    print(f"Trajectory: {len(generic_positions)} steps, terminal_reason={terminal_reason}")
    print(f"\nGeneric agent displacement from its own starting position:")
    print(f"  max: {generic_displacement.max():.4f} m")
    print(f"  mean: {generic_displacement.mean():.4f} m")
    print(f"  final: {generic_displacement[-1]:.4f} m")
    print(f"\nCalibrated agent displacement from its own starting position:")
    print(f"  max: {calibrated_displacement.max():.4f} m")
    print(f"  mean: {calibrated_displacement.mean():.4f} m")
    print(f"  final: {calibrated_displacement[-1]:.4f} m")
    print(f"\nRatio (generic/calibrated max displacement): "
          f"{generic_displacement.max() / max(calibrated_displacement.max(), 1e-9):.4f}")
    print(f"\nFirst 10 generic positions:\n{generic_positions[:10]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--d0", type=float, default=4.0)
    parser.add_argument("--v0", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    main(args.model_path, args.d0, args.v0, args.seed, args.max_steps)