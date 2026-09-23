"""
compare_ppo_vs_heuristic.py — runs PPO's REAL trajectory (its own actions
advance the physics, exactly as every other evaluation in this
investigation) and, at EVERY step, ALSO queries what the heuristic
controller (check_alternative_controller.py) would have chosen at that
EXACT same state -- without applying it. A pure counterfactual query,
giving a genuine "same state, different action" comparison rather than
two trajectories that diverge after step one.

Full per-step CSVs for the three focus cases (collision, robust_failure,
success); a compact summary table across the full 32-point grid.
"""
import argparse
import csv

import numpy as np

from evaluate_recovery import (build_env, reset_to_custom_state, safe_csv_path,
                                DEFAULT_ACTION_BOUND, SATURATION_THRESHOLD_FRACTION)
from check_alternative_controller import alternative_action

FOCUS_CASES = [
    ("v5_collision_2.0_neg0.7", 2.0, -0.7),
    ("v5_collision_2.5_neg0.7", 2.5, -0.7),
    ("v5_collision_2.5_pos0.7", 2.5, 0.7),
    ("v5_collision_3.0_pos0.15", 3.0, 0.15),
]
GRID_SEPARATIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
GRID_VELOCITIES = [-0.7, -0.15, 0.15, 0.7]


def run_comparison(env, model, original_step, d0: float, v0: float, seed: int = 0, max_steps: int = 200):
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
            core_c = env._orchestrator.core_calibrated
            core_g = env._orchestrator.core_generic
            current_sep = float(np.linalg.norm(core_c.state.position - core_g.state.position))
            d_dot = (current_sep - sep_history[-1]) / dt if sep_history else 0.0
            sep_history.append(current_sep)
            pitch = core_c.state.orientation.as_euler("xyz")[1]
            q = core_c.state.nu[4]

            # PPO's REAL action -- this is what actually advances the physics.
            ppo_action, _ = model.predict(obs, deterministic=True)

            # Heuristic's COUNTERFACTUAL action at this EXACT same state -- computed, not applied.
            heuristic_full_action = alternative_action(current_sep, d_dot, pitch, q)
            heuristic_fx = heuristic_full_action[6] * DEFAULT_ACTION_BOUND
            heuristic_ty = heuristic_full_action[10] * DEFAULT_ACTION_BOUND

            obs, reward, terminated, truncated, info = env.step(ppo_action)

            tau_c = captured_tau["calibrated"]
            ppo_fx, ppo_ty = float(tau_c[0]), float(tau_c[4])
            sat = bool(np.max(np.abs(tau_c)) >= saturation_threshold)

            new_sep = info["true_separation"]
            next_d_dot = (new_sep - current_sep) / dt

            rows.append({
                "step": step, "separation": current_sep, "d_dot": d_dot, "pitch": pitch, "q": q,
                "ppo_Fx": ppo_fx, "ppo_Ty": ppo_ty,
                "heuristic_Fx": heuristic_fx, "heuristic_Ty": heuristic_ty,
                "Fx_diff": ppo_fx - heuristic_fx, "Ty_diff": ppo_ty - heuristic_ty,
                "ppo_saturated": sat, "next_d_dot": next_d_dot,
                "consecutive_steps_in_tolerance": info["consecutive_steps_in_tolerance"],
            })

            if terminated:
                terminal_reason = info.get("terminal_reason")
                break
            if truncated:
                break
    finally:
        env._orchestrator.step = original_step

    return rows, terminal_reason


def main(model_path: str, seed: int, csv_prefix: str):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    env = build_env()
    true_original_step = env._orchestrator.step
    hold_target = env._orchestrator.task.config.hold_steps

    print("=== FOCUS CASES: full per-step PPO vs heuristic comparison ===\n")
    for case_name, d0, v0 in FOCUS_CASES:
        rows, terminal_reason = run_comparison(env, model, true_original_step, d0, v0, seed)
        path = safe_csv_path(f"{csv_prefix}_{case_name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        mean_fx_diff = np.mean([abs(r["Fx_diff"]) for r in rows])
        mean_ty_diff = np.mean([abs(r["Ty_diff"]) for r in rows])
        frac_sat = np.mean([r["ppo_saturated"] for r in rows])
        print(f"{case_name}: {len(rows)} steps, terminal_reason={terminal_reason}")
        print(f"  mean|Fx_diff|={mean_fx_diff:.2f}  mean|Ty_diff|={mean_ty_diff:.2f}  "
              f"PPO saturated_fraction={frac_sat:.2f}")
        print(f"  written to {path}\n")

    print("=== FULL GRID: summary comparison ===\n")
    print(f"{'d0':>6} {'v0':>6} {'PPO':>4} {'mean|Fx_diff|':>14} {'mean|Ty_diff|':>14} {'ppo_sat_frac':>13}")
    for d0 in GRID_SEPARATIONS:
        for v0 in GRID_VELOCITIES:
            rows, terminal_reason = run_comparison(env, model, true_original_step, d0, v0, seed)
            max_consec = max(r["consecutive_steps_in_tolerance"] for r in rows)
            if terminal_reason == "collision":
                outcome = "C"
            elif max_consec >= hold_target:
                outcome = "S"
            elif max_consec > 0:
                outcome = "R"
            else:
                outcome = "F"
            mean_fx_diff = np.mean([abs(r["Fx_diff"]) for r in rows])
            mean_ty_diff = np.mean([abs(r["Ty_diff"]) for r in rows])
            frac_sat = np.mean([r["ppo_saturated"] for r in rows])
            print(f"{d0:>6.2f} {v0:>6.2f} {outcome:>4} {mean_fx_diff:>14.2f} {mean_ty_diff:>14.2f} {frac_sat:>13.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-prefix", default="ppo_vs_heuristic")
    args = parser.parse_args()
    main(args.model_path, args.seed, args.csv_prefix)