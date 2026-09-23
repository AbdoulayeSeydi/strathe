"""
overlay_ppo_heuristic_trajectories.py -- runs TWO independent, real
trajectories from the SAME initial (d0, v0): PPO's real trajectory (as
in every other evaluation), and the heuristic's OWN real trajectory --
its actions actually advance a SEPARATE episode, unlike
compare_ppo_vs_heuristic.py, which only ever asks "what would the
heuristic have done at PPO's state" without ever letting it drive.

The heuristic's trajectory, given its established 32/32 grid success,
empirically DEFINES a "recoverable corridor" in (d_dot, pitch, q) space
for this exact starting condition -- the range it needed to pass through
to succeed. separation itself is excluded from the corridor check:
PPO trending toward collision will trivially undercut the heuristic's
minimum separation near the end, and that outcome is what we're trying
to explain, not new information about why it happens.

This finds the first PPO step where d_dot, pitch, or q falls OUTSIDE
that corridor -- the first point PPO enters state territory the
heuristic never needed to visit to solve the identical problem.
"""
import argparse
import csv

import numpy as np

from evaluate_recovery import (build_env, reset_to_custom_state, safe_csv_path,
                                DEFAULT_ACTION_BOUND, SATURATION_THRESHOLD_FRACTION)
from check_alternative_controller import alternative_action

CASES = [
    ("v5_collision_2.0_neg0.7", 2.0, -0.7),
    ("v5_collision_2.5_neg0.7", 2.5, -0.7),
    ("v5_collision_2.5_pos0.7", 2.5, 0.7),
    ("v5_collision_3.0_pos0.15", 3.0, 0.15),
]
CORRIDOR_KEYS = ("d_dot", "pitch", "q")


def _current_state(env):
    core_c = env._orchestrator.core_calibrated
    core_g = env._orchestrator.core_generic
    sep = float(np.linalg.norm(core_c.state.position - core_g.state.position))
    pitch = core_c.state.orientation.as_euler("xyz")[1]
    q = core_c.state.nu[4]
    return sep, pitch, q


def run_heuristic_trajectory(env, d0: float, v0: float, seed: int = 0, max_steps: int = 200):
    """Heuristic's OWN actions actually advance the physics here -- an
    independent real trajectory, not a counterfactual query."""
    obs = reset_to_custom_state(env, d0, v0, seed)
    dt = env._orchestrator.config.dt

    rows = []
    sep_history = []
    terminal_reason = None
    for step in range(1, max_steps + 1):
        current_sep, pitch, q = _current_state(env)
        d_dot = (current_sep - sep_history[-1]) / dt if sep_history else 0.0
        sep_history.append(current_sep)

        action = alternative_action(current_sep, d_dot, pitch, q)
        obs, reward, terminated, truncated, info = env.step(action)

        rows.append({"step": step, "separation": current_sep, "d_dot": d_dot, "pitch": pitch, "q": q})

        if terminated:
            terminal_reason = info.get("terminal_reason")
            break
        if truncated:
            break
    return rows, terminal_reason


def run_ppo_trajectory(env, model, original_step, d0: float, v0: float, seed: int = 0, max_steps: int = 200):
    obs = reset_to_custom_state(env, d0, v0, seed)

    captured_tau = {}

    def spy_step(action_generic, action_calibrated, **kwargs):
        captured_tau["calibrated"] = np.array(action_calibrated, dtype=float)
        return original_step(action_generic, action_calibrated, **kwargs)

    env._orchestrator.step = spy_step
    saturation_threshold = SATURATION_THRESHOLD_FRACTION * DEFAULT_ACTION_BOUND
    dt = env._orchestrator.config.dt

    rows = []
    sep_history = []
    terminal_reason = None
    try:
        for step in range(1, max_steps + 1):
            current_sep, pitch, q = _current_state(env)
            d_dot = (current_sep - sep_history[-1]) / dt if sep_history else 0.0
            sep_history.append(current_sep)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            tau_c = captured_tau["calibrated"]
            sat = bool(np.max(np.abs(tau_c)) >= saturation_threshold)

            rows.append({"step": step, "separation": current_sep, "d_dot": d_dot, "pitch": pitch, "q": q,
                         "ppo_saturated": sat})

            if terminated:
                terminal_reason = info.get("terminal_reason")
                break
            if truncated:
                break
    finally:
        env._orchestrator.step = original_step
    return rows, terminal_reason


def compute_corridor(heuristic_rows):
    return {key: (min(r[key] for r in heuristic_rows), max(r[key] for r in heuristic_rows))
            for key in CORRIDOR_KEYS}


def find_first_corridor_exit(ppo_rows, corridor):
    for row in ppo_rows:
        for key in CORRIDOR_KEYS:
            lo, hi = corridor[key]
            if not (lo <= row[key] <= hi):
                return row, key, corridor[key]
    return None, None, None


def main(model_path: str, seed: int, csv_prefix: str):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    env = build_env()
    true_original_step = env._orchestrator.step

    for case_name, d0, v0 in CASES:
        print(f"\n{'='*90}\n{case_name} (d0={d0}, v0={v0})\n{'='*90}")

        heuristic_rows, heuristic_terminal = run_heuristic_trajectory(env, d0, v0, seed)
        corridor = compute_corridor(heuristic_rows)
        print(f"Heuristic's OWN trajectory: {len(heuristic_rows)} steps, terminal_reason={heuristic_terminal}")
        print("Heuristic's demonstrated corridor for this starting condition:")
        for key, (lo, hi) in corridor.items():
            print(f"  {key}: [{lo:.3f}, {hi:.3f}]")

        ppo_rows, ppo_terminal = run_ppo_trajectory(env, model, true_original_step, d0, v0, seed)
        print(f"\nPPO's REAL trajectory: {len(ppo_rows)} steps, terminal_reason={ppo_terminal}")

        exit_row, exit_key, exit_bounds = find_first_corridor_exit(ppo_rows, corridor)
        if exit_row is None:
            print("PPO never left the heuristic's demonstrated (d_dot/pitch/q) corridor.")
        else:
            print(f"\nFirst corridor exit at PPO step {exit_row['step']} (of {len(ppo_rows)}): "
                  f"{exit_key}={exit_row[exit_key]:.3f} is outside "
                  f"[{exit_bounds[0]:.3f}, {exit_bounds[1]:.3f}]")
            print(f"  Full state at exit: separation={exit_row['separation']:.3f}, "
                  f"d_dot={exit_row['d_dot']:.3f}, pitch={exit_row['pitch']:.3f}, q={exit_row['q']:.3f}, "
                  f"ppo_saturated={exit_row['ppo_saturated']}")
            print(f"  fraction of PPO trace remaining after exit: "
                  f"{(len(ppo_rows) - exit_row['step']) / len(ppo_rows):.2f}")

        combined_path = safe_csv_path(f"{csv_prefix}_{case_name}.csv")
        with open(combined_path, "w", newline="") as f:
            fieldnames = ["source", "step", "separation", "d_dot", "pitch", "q", "ppo_saturated"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in heuristic_rows:
                writer.writerow({"source": "heuristic", **row, "ppo_saturated": ""})
            for row in ppo_rows:
                writer.writerow({"source": "ppo", **row})
        print(f"  Combined trajectory CSV written to {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-prefix", default="overlay")
    args = parser.parse_args()
    main(args.model_path, args.seed, args.csv_prefix)