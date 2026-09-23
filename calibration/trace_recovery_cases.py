"""
trace_recovery_cases.py — merged instrumentation: custom-state injection
(from evaluate_recovery.py) + full per-step state/action/reward capture
(extending trace_episode_detailed.py's approach), run across three named
cases for direct mechanistic contrast:

  collision      (3.75,  0.0)  -- identify the actual safety-failure mechanism
  robust_failure (5.00, +0.70) -- identify the far-recovery failure mechanism
                                   (jitter-confirmed: 25/25 neighbors are F)
  success        (3.00,  0.0)  -- what the controller does when recovery works

Captures, every step, for the calibrated agent:
  relative state:  separation d, d_dot, relative position vector, relative velocity vector
  vehicle state:   pitch, pitch rate, surge/sway/heave velocity, all angular rates
  actions:         raw (real-Newton) Fx/Fy/Fz/Tx/Ty/Tz, the NORMALIZED [-1,1]
                   action driving them, and a per-component saturation flag
  outcome:         reward, all reward COMPONENTS (separation/control/proximity/
                   collision/success_bonus), consecutive-hold counter,
                   running minimum separation, terminal reason
  derived:         closing_speed = max(0, -d_dot); time_to_collision =
                   (d - d_safe) / closing_speed when d_dot < 0, else None

Output: one combined CSV (all three cases, a "case" column distinguishes
them) plus three PNG plots, each overlaying all three cases:
  1. separation (+ safety_radius line, + tolerance band)
  2. d_dot (+ commanded Fx)
  3. pitch (+ commanded Ty)
"""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_recovery import build_env, reset_to_custom_state, DEFAULT_ACTION_BOUND
from evaluate_recovery import SATURATION_THRESHOLD_FRACTION

CASES = [
    ("collision", 3.75, 0.0),
    ("robust_failure", 5.00, 0.70),
    ("success", 3.00, 0.0),
]

FIELDNAMES = [
    "case", "step", "separation", "d_dot", "rel_pos_x", "rel_pos_y", "rel_pos_z",
    "rel_vel_x", "rel_vel_y", "rel_vel_z",
    "cal_pitch", "cal_roll", "cal_yaw", "cal_u", "cal_v", "cal_w", "cal_p", "cal_q", "cal_r",
    "cal_Fx_raw", "cal_Fy_raw", "cal_Fz_raw", "cal_Tx_raw", "cal_Ty_raw", "cal_Tz_raw",
    "cal_action_norm_0", "cal_action_norm_1", "cal_action_norm_2",
    "cal_action_norm_3", "cal_action_norm_4", "cal_action_norm_5",
    "any_component_saturated", "n_components_saturated",
    "reward", "reward_separation", "reward_control", "reward_proximity",
    "reward_collision", "reward_success_bonus",
    "consecutive_steps_in_tolerance", "running_min_separation", "terminal_reason",
    "closing_speed", "time_to_collision",
]


def trace_one_case(env, model, original_step, case_name: str, d0: float, v0: float,
                    seed: int = 0, max_steps: int = 200, safety_radius: float = 0.5):
    obs = reset_to_custom_state(env, d0, v0, seed)

    captured_tau = {}

    def spy_step(action_generic, action_calibrated, **kwargs):
        captured_tau["calibrated"] = np.array(action_calibrated, dtype=float)
        return original_step(action_generic, action_calibrated, **kwargs)

    env._orchestrator.step = spy_step
    saturation_threshold = SATURATION_THRESHOLD_FRACTION * DEFAULT_ACTION_BOUND

    rows = []
    running_min_sep = float("inf")
    prev_sep = None
    try:
        for step in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            core_g = env._orchestrator.core_generic
            core_c = env._orchestrator.core_calibrated

            sep = info["true_separation"]
            d_dot = (sep - prev_sep) / env._orchestrator.config.dt if prev_sep is not None else 0.0
            prev_sep = sep
            running_min_sep = min(running_min_sep, sep)

            rel_pos = core_c.state.position - core_g.state.position
            g_world_v = core_g.state.orientation.apply(core_g.state.nu[:3])
            c_world_v = core_c.state.orientation.apply(core_c.state.nu[:3])
            rel_vel = c_world_v - g_world_v

            c_roll, c_pitch, c_yaw = core_c.state.orientation.as_euler("xyz")
            tau_c = captured_tau["calibrated"]
            normalized_action = tau_c / DEFAULT_ACTION_BOUND
            saturated_mask = np.abs(tau_c) >= saturation_threshold

            closing_speed = max(0.0, -d_dot)
            ttc = (sep - safety_radius) / closing_speed if closing_speed > 1e-9 else None

            rc = info["reward_components"]
            row = {
                "case": case_name, "step": step, "separation": sep, "d_dot": d_dot,
                "rel_pos_x": rel_pos[0], "rel_pos_y": rel_pos[1], "rel_pos_z": rel_pos[2],
                "rel_vel_x": rel_vel[0], "rel_vel_y": rel_vel[1], "rel_vel_z": rel_vel[2],
                "cal_pitch": c_pitch, "cal_roll": c_roll, "cal_yaw": c_yaw,
                "cal_u": core_c.state.nu[0], "cal_v": core_c.state.nu[1], "cal_w": core_c.state.nu[2],
                "cal_p": core_c.state.nu[3], "cal_q": core_c.state.nu[4], "cal_r": core_c.state.nu[5],
                "cal_Fx_raw": tau_c[0], "cal_Fy_raw": tau_c[1], "cal_Fz_raw": tau_c[2],
                "cal_Tx_raw": tau_c[3], "cal_Ty_raw": tau_c[4], "cal_Tz_raw": tau_c[5],
                "cal_action_norm_0": normalized_action[0], "cal_action_norm_1": normalized_action[1],
                "cal_action_norm_2": normalized_action[2], "cal_action_norm_3": normalized_action[3],
                "cal_action_norm_4": normalized_action[4], "cal_action_norm_5": normalized_action[5],
                "any_component_saturated": bool(saturated_mask.any()),
                "n_components_saturated": int(saturated_mask.sum()),
                "reward": reward,
                "reward_separation": rc.get("separation"), "reward_control": rc.get("control"),
                "reward_proximity": rc.get("proximity"), "reward_collision": rc.get("collision"),
                "reward_success_bonus": rc.get("success_bonus"),
                "consecutive_steps_in_tolerance": info["consecutive_steps_in_tolerance"],
                "running_min_separation": running_min_sep,
                "terminal_reason": info.get("terminal_reason"),
                "closing_speed": closing_speed, "time_to_collision": ttc,
            }
            rows.append(row)

            if terminated or truncated:
                break
    finally:
        env._orchestrator.step = original_step

    return rows


def make_plots(all_rows: list, output_prefix: str):
    cases = sorted(set(r["case"] for r in all_rows), key=lambda c: [n for n, *_ in CASES].index(c))
    colors = {"collision": "tab:red", "robust_failure": "tab:orange", "success": "tab:green"}

    # Plot 1: separation + safety boundary + tolerance band
    fig, ax = plt.subplots(figsize=(10, 5))
    for case in cases:
        case_rows = [r for r in all_rows if r["case"] == case]
        steps = [r["step"] for r in case_rows]
        seps = [r["separation"] for r in case_rows]
        ax.plot(steps, seps, label=case, color=colors.get(case), linewidth=1.5)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="safety_radius=0.5")
    ax.axhspan(1.8, 2.2, color="green", alpha=0.1, label="tolerance band [1.8,2.2]")
    ax.set_xlabel("step")
    ax.set_ylabel("separation (m)")
    ax.set_title("Separation over time, three diagnostic cases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_separation.png", dpi=120)
    plt.close(fig)

    # Plot 2: d_dot + commanded Fx
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for case in cases:
        case_rows = [r for r in all_rows if r["case"] == case]
        steps = [r["step"] for r in case_rows]
        axes[0].plot(steps, [r["d_dot"] for r in case_rows], label=case, color=colors.get(case))
        axes[1].plot(steps, [r["cal_Fx_raw"] for r in case_rows], label=case, color=colors.get(case))
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_ylabel("d_dot (m/s)")
    axes[0].set_title("Separation rate and commanded surge force")
    axes[0].legend()
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("commanded Fx (N)")
    axes[1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_ddot_fx.png", dpi=120)
    plt.close(fig)

    # Plot 3: pitch + commanded Ty
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for case in cases:
        case_rows = [r for r in all_rows if r["case"] == case]
        steps = [r["step"] for r in case_rows]
        axes[0].plot(steps, [r["cal_pitch"] for r in case_rows], label=case, color=colors.get(case))
        axes[1].plot(steps, [r["cal_Ty_raw"] for r in case_rows], label=case, color=colors.get(case))
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_ylabel("pitch (rad)")
    axes[0].set_title("Pitch and commanded pitch torque")
    axes[0].legend()
    axes[1].axhline(30, color="red", linestyle="--", alpha=0.4, label="+/-30 limit")
    axes[1].axhline(-30, color="red", linestyle="--", alpha=0.4)
    axes[1].set_ylabel("commanded Ty (N*m)")
    axes[1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_pitch_ty.png", dpi=120)
    plt.close(fig)


def main(model_path: str, seed: int, csv_path: str, output_prefix: str):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    env = build_env()
    true_original_step = env._orchestrator.step

    all_rows = []
    for case_name, d0, v0 in CASES:
        print(f"Tracing case '{case_name}' (d0={d0}, v0={v0})...")
        rows = trace_one_case(env, model, true_original_step, case_name, d0, v0, seed)
        all_rows.extend(rows)
        final = rows[-1]
        print(f"  {len(rows)} steps, terminal_reason={final['terminal_reason']}, "
              f"final_separation={final['separation']:.3f}, "
              f"running_min_separation={final['running_min_separation']:.3f}")

    from evaluate_recovery import safe_csv_path
    actual_csv_path = safe_csv_path(csv_path)
    with open(actual_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCombined trace ({len(all_rows)} rows) written to {actual_csv_path}")

    make_plots(all_rows, output_prefix)
    print(f"Plots written to {output_prefix}_separation.png, {output_prefix}_ddot_fx.png, "
          f"{output_prefix}_pitch_ty.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-path", default="trace_recovery_cases.csv")
    parser.add_argument("--output-prefix", default="trace_recovery_cases")
    args = parser.parse_args()
    main(args.model_path, args.seed, args.csv_path, args.output_prefix)