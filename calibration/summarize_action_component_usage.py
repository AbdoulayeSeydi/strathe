"""
summarize_action_component_usage.py -- aggregates per-axis control usage
(mean |action|, max |action|, fraction >=0.99 limit, fraction >=0.8
limit, sign switches) across the first N steps of each of the four
collision cases, to answer the specific remaining question from the
six-axis saturation check: is PPO systematically spending control
authority on Fy/Fz/Tx/Tz, or were those isolated artifacts of the
handful of trajectories already inspected by eye?

Reuses run_ppo_trajectory_full_action from diagnose_six_axis_saturation.py
directly -- no new environment-stepping logic, just aggregation over
data that script already produces per-step.
"""
import argparse

import numpy as np

from evaluate_recovery import build_env, DEFAULT_ACTION_BOUND
from diagnose_six_axis_saturation import CASES, AXIS_NAMES, SATURATION_THRESHOLD, run_ppo_trajectory_full_action

NEAR_SATURATION_THRESHOLD = 0.8 * DEFAULT_ACTION_BOUND


def compute_axis_stats(rows, axis_name: str, n_steps: int) -> dict:
    limited = rows[:n_steps]
    values = np.array([r[axis_name] for r in limited])
    abs_values = np.abs(values)
    signs = np.sign(values)
    sign_switches = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

    return {
        "axis": axis_name,
        "n_steps": len(limited),
        "mean_abs": float(np.mean(abs_values)),
        "max_abs": float(np.max(abs_values)),
        "frac_saturated": float(np.mean(abs_values >= SATURATION_THRESHOLD)),
        "frac_near_saturated": float(np.mean(abs_values >= NEAR_SATURATION_THRESHOLD)),
        "sign_switches": sign_switches,
        "sign_switch_rate": sign_switches / max(len(limited) - 1, 1),
    }


def print_case_table(case_name: str, rows, n_steps: int):
    print(f"\n{case_name}  (first {min(n_steps, len(rows))} of {len(rows)} steps)")
    header = (f"  {'axis':>4} {'mean|a|':>8} {'max|a|':>7} {'frac>=99%':>10} "
              f"{'frac>=80%':>10} {'sign_switches':>14}")
    print(header)
    for axis in AXIS_NAMES:
        s = compute_axis_stats(rows, axis, n_steps)
        print(f"  {s['axis']:>4} {s['mean_abs']:>8.2f} {s['max_abs']:>7.2f} "
              f"{s['frac_saturated']:>10.2f} {s['frac_near_saturated']:>10.2f} "
              f"{s['sign_switches']:>7}/{s['n_steps']-1} ({s['sign_switch_rate']:.2f})")


def main(model_path: str, seed: int, n_steps: int):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    env = build_env()
    true_original_step = env._orchestrator.step

    all_case_rows = {}
    for case_name, d0, v0 in CASES:
        ppo_rows, terminal_reason = run_ppo_trajectory_full_action(env, model, true_original_step, d0, v0, seed)
        all_case_rows[case_name] = ppo_rows
        print_case_table(case_name, ppo_rows, n_steps)

    print(f"\n{'='*80}\nCOMBINED ACROSS ALL FOUR CASES (first {n_steps} steps each)\n{'='*80}")
    print(f"  {'axis':>4} {'mean|a|':>8} {'max|a|':>7} {'frac>=99%':>10} "
          f"{'frac>=80%':>10} {'sign_switches':>14}")
    for axis in AXIS_NAMES:
        pooled = []
        for case_name, rows in all_case_rows.items():
            pooled.extend(rows[:n_steps])
        s = compute_axis_stats(pooled, axis, len(pooled))
        print(f"  {s['axis']:>4} {s['mean_abs']:>8.2f} {s['max_abs']:>7.2f} "
              f"{s['frac_saturated']:>10.2f} {s['frac_near_saturated']:>10.2f} "
              f"{s['sign_switches']:>7}/{s['n_steps']-1} ({s['sign_switch_rate']:.2f})")
    print(f"\n(pooled naively across case boundaries -- the transition between the "
          f"last step of one case and the first step of the next inflates sign_switches "
          f"by at most 3, negligible against {sum(len(r[:n_steps]) for r in all_case_rows.values())} total steps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=15)
    args = parser.parse_args()
    main(args.model_path, args.seed, args.n_steps)