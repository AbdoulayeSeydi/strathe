"""
diagnose_six_axis_saturation.py -- extends the corridor-exit diagnostic
with the FULL 6-dimensional calibrated action vector (Fx,Fy,Fz,Tx,Ty,Tz),
not just Fx/Ty. Every prior script in this investigation -- including
overlay_ppo_heuristic_trajectories.py and compare_ppo_vs_heuristic.py --
only displayed Fx/Ty, mirroring the heuristic's own convention of
zeroing the other four axes. But ppo_saturated is computed as
np.max(np.abs(tau_c)) over ALL SIX components, and several traces show
ppo_saturated=True while Fx/Ty are both small (e.g. Fx=3.45, Ty=3.29 at
step 2 of the 2.0,-0.7 case) -- meaning PPO must be driving one of the
other four axes to near-saturation at exactly these points, unexamined
until now.

This does NOT assume Fx/Ty are responsible; it reports whichever
component(s) are actually at or above the saturation threshold, for
every one of the first N steps of each case PLUS the exact corridor-exit
step already found by overlay_ppo_heuristic_trajectories.py (reusing its
CASES list and corridor logic directly, so the exit step reported here
is identical to what was already found, not a re-derivation).
"""
import argparse

import numpy as np

from evaluate_recovery import build_env, reset_to_custom_state, DEFAULT_ACTION_BOUND, SATURATION_THRESHOLD_FRACTION
from overlay_ppo_heuristic_trajectories import CASES, run_heuristic_trajectory, compute_corridor

AXIS_NAMES = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
SATURATION_THRESHOLD = SATURATION_THRESHOLD_FRACTION * DEFAULT_ACTION_BOUND


def run_ppo_trajectory_full_action(env, model, original_step, d0: float, v0: float,
                                    seed: int = 0, max_steps: int = 200):
    obs = reset_to_custom_state(env, d0, v0, seed)

    captured_tau = {}

    def spy_step(action_generic, action_calibrated, **kwargs):
        captured_tau["calibrated"] = np.array(action_calibrated, dtype=float)
        return original_step(action_generic, action_calibrated, **kwargs)

    env._orchestrator.step = spy_step
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

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            tau_c = captured_tau["calibrated"]  # full 6-dim: [Fx, Fy, Fz, Tx, Ty, Tz]
            abs_tau = np.abs(tau_c)
            max_abs = float(np.max(abs_tau))
            saturated_axes = [AXIS_NAMES[i] for i in range(6) if abs_tau[i] >= SATURATION_THRESHOLD]

            row = {"step": step, "separation": current_sep, "d_dot": d_dot, "pitch": pitch, "q": q,
                   "max_abs_tau": max_abs, "saturated_axes": saturated_axes}
            for i, name in enumerate(AXIS_NAMES):
                row[name] = float(tau_c[i])
                row[f"norm_{name}"] = float(tau_c[i]) / DEFAULT_ACTION_BOUND
            rows.append(row)

            if terminated:
                terminal_reason = info.get("terminal_reason")
                break
            if truncated:
                break
    finally:
        env._orchestrator.step = original_step
    return rows, terminal_reason


def find_corridor_exit_step(ppo_rows, corridor):
    """Same check as overlay_ppo_heuristic_trajectories.find_first_corridor_exit,
    reproduced here since that function's rows don't carry the 6-axis fields
    this script adds -- kept logically identical (same keys, same order,
    same inequality) so the step number matches exactly."""
    for row in ppo_rows:
        for key in ("d_dot", "pitch", "q"):
            lo, hi = corridor[key]
            if not (lo <= row[key] <= hi):
                return row["step"]
    return None


def print_table(rows, n_steps, highlight_step=None):
    header = (f"{'step':>5} {'sep':>7} {'d_dot':>8} {'pitch':>8} {'q':>8} "
              f"{'Fx':>8} {'Fy':>8} {'Fz':>8} {'Tx':>8} {'Ty':>8} {'Tz':>8} "
              f"{'max|tau|':>9} saturated_axes")
    print(header)
    shown_steps = set(r["step"] for r in rows[:n_steps])
    if highlight_step is not None:
        shown_steps.add(highlight_step)
    for row in rows:
        if row["step"] not in shown_steps:
            continue
        marker = "  <-- CORRIDOR EXIT" if row["step"] == highlight_step else ""
        axes_str = ",".join(row["saturated_axes"]) if row["saturated_axes"] else "none"
        print(f"{row['step']:>5} {row['separation']:>7.3f} {row['d_dot']:>8.3f} "
              f"{row['pitch']:>8.3f} {row['q']:>8.3f} "
              f"{row['Fx']:>8.2f} {row['Fy']:>8.2f} {row['Fz']:>8.2f} "
              f"{row['Tx']:>8.2f} {row['Ty']:>8.2f} {row['Tz']:>8.2f} "
              f"{row['max_abs_tau']:>9.2f} {axes_str}{marker}")


def main(model_path: str, seed: int, n_steps: int):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    env = build_env()
    true_original_step = env._orchestrator.step

    print(f"Saturation threshold: {SATURATION_THRESHOLD_FRACTION} * {DEFAULT_ACTION_BOUND} "
          f"= {SATURATION_THRESHOLD:.2f} N/Nm on any single component\n")

    for case_name, d0, v0 in CASES:
        print(f"\n{'='*118}\n{case_name} (d0={d0}, v0={v0})\n{'='*118}")

        heuristic_rows, _ = run_heuristic_trajectory(env, d0, v0, seed)
        corridor = compute_corridor(heuristic_rows)

        ppo_rows, terminal_reason = run_ppo_trajectory_full_action(env, model, true_original_step, d0, v0, seed)
        exit_step = find_corridor_exit_step(ppo_rows, corridor)

        print(f"PPO trajectory: {len(ppo_rows)} steps, terminal_reason={terminal_reason}")
        print(f"Corridor exit previously identified at step: {exit_step}\n")
        print_table(ppo_rows, n_steps, highlight_step=exit_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=15,
                         help="How many initial steps to print per case, plus the corridor-exit step")
    args = parser.parse_args()
    main(args.model_path, args.seed, args.n_steps)