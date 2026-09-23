"""
evaluate_comm_degradation_2x2.py -- A1.10's core evaluation: runs the
2x2 (policy trained under L0/L2) x (evaluated under L0/L2) matrix,
reporting all six agreed metric categories with generic-vs-calibrated
broken out separately, plus the Delta_cal = M_generic - M_calibrated
gap for each per-agent metric.

Uses the SAME 32-point x n_seeds grid established throughout A1.9 for
comparability. Each cell gets its own fresh environment with a fixed
comm_seed -- no attempt at exact matched-pair RNG synchronization
across cells (fragile, since episode lengths differ across models);
aggregate statistics over 32 x n_seeds episodes are the right level of
comparison here, not per-episode pairing.

Per-agent "formation/separation error" is operationalized as off-axis
(y,z) drift from each agent's own starting position -- the task only
requires x-axis motion to close the separation gap, so y/z drift is
wasted/off-task motion, directly comparable to what
check_generic_agent_movement.py found (generic wandering in all three
axes despite only x-motion being task-relevant).

*** UNVERIFIED against a real model, same as every SB3-dependent script
in this investigation -- run_episode/run_cell/the aggregation logic are
fully tested below against a fake model and the real environment;
model.predict() itself is not. ***
"""
import argparse
import csv

import numpy as np

from two_agent_formation_env import TwoAgentFormationEnv
from comm_channel import PacketLossConfig, BandwidthConfig
from evaluate_recovery import reset_to_custom_state, safe_csv_path

GRID_SEPARATIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
GRID_VELOCITIES = [-0.7, -0.15, 0.15, 0.7]

COMM_CONFIGS = {
    "L0": dict(packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.0, 0.0),
               bandwidth_config=BandwidthConfig(bytes_per_tick=1e9)),
    "L2": dict(packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.3, 0.3),
               bandwidth_config=BandwidthConfig(bytes_per_tick=1e9)),
}


def build_env_with_comm(comm_level: str, comm_seed: int = 42):
    cfg = COMM_CONFIGS[comm_level]
    return TwoAgentFormationEnv(
        packet_loss_config=cfg["packet_loss_config"],
        bandwidth_config=cfg["bandwidth_config"],
        comm_seed=comm_seed,
    )


def run_episode(env, model, d0: float, v0: float, seed: int, max_steps: int = 200) -> dict:
    obs = reset_to_custom_state(env, d0, v0, seed)

    positions_generic, positions_calibrated = [], []
    belief_errors_generic, belief_errors_calibrated = [], []
    delivered_generic_flags, delivered_calibrated_flags = [], []
    terminal_reason = None
    steps_run = 0

    for step in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps_run = step

        pos_g = info["position_generic"]
        pos_c = info["position_calibrated"]
        positions_generic.append(np.array(pos_g, dtype=float))
        positions_calibrated.append(np.array(pos_c, dtype=float))

        revealed_g = info["revealed_to_generic"]
        revealed_c = info["revealed_to_calibrated"]
        if revealed_g is not None:
            belief_errors_generic.append(float(np.linalg.norm(np.asarray(revealed_g) - pos_c)))
        if revealed_c is not None:
            belief_errors_calibrated.append(float(np.linalg.norm(np.asarray(revealed_c) - pos_g)))

        delivered_generic_flags.append(info["delivered_to_generic_this_tick"])
        delivered_calibrated_flags.append(info["delivered_to_calibrated_this_tick"])

        if terminated:
            terminal_reason = info.get("terminal_reason")
            break
        if truncated:
            break

    positions_generic = np.array(positions_generic)
    positions_calibrated = np.array(positions_calibrated)
    off_axis_generic = np.linalg.norm(positions_generic[:, 1:] - positions_generic[0, 1:], axis=1)
    off_axis_calibrated = np.linalg.norm(positions_calibrated[:, 1:] - positions_calibrated[0, 1:], axis=1)

    return {
        "steps_run": steps_run,
        "terminal_reason": terminal_reason,
        "success": terminal_reason == "formation_achieved",
        "collision": terminal_reason == "collision",
        "mean_off_axis_generic": float(np.mean(off_axis_generic)),
        "mean_off_axis_calibrated": float(np.mean(off_axis_calibrated)),
        "mean_belief_error_generic": float(np.mean(belief_errors_generic)) if belief_errors_generic else float("nan"),
        "frac_no_belief_generic": 1.0 - len(belief_errors_generic) / steps_run,
        "mean_belief_error_calibrated": float(np.mean(belief_errors_calibrated)) if belief_errors_calibrated else float("nan"),
        "frac_no_belief_calibrated": 1.0 - len(belief_errors_calibrated) / steps_run,
        "delivery_rate_generic": float(np.mean(delivered_generic_flags)),
        "delivery_rate_calibrated": float(np.mean(delivered_calibrated_flags)),
    }


def run_cell(model, comm_level: str, n_seeds: int, csv_path: str):
    env = build_env_with_comm(comm_level)
    rows = []
    for d0 in GRID_SEPARATIONS:
        for v0 in GRID_VELOCITIES:
            for seed in range(n_seeds):
                result = run_episode(env, model, d0, v0, seed)
                result.update({"d0": d0, "v0": v0, "seed": seed})
                rows.append(result)

    actual_path = safe_csv_path(csv_path)
    with open(actual_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    summary = {
        "n_episodes": n,
        "success_rate": sum(r["success"] for r in rows) / n,
        "collision_rate": sum(r["collision"] for r in rows) / n,
        "mean_off_axis_generic": float(np.mean([r["mean_off_axis_generic"] for r in rows])),
        "mean_off_axis_calibrated": float(np.mean([r["mean_off_axis_calibrated"] for r in rows])),
        "mean_belief_error_generic": float(np.nanmean([r["mean_belief_error_generic"] for r in rows])),
        "mean_belief_error_calibrated": float(np.nanmean([r["mean_belief_error_calibrated"] for r in rows])),
        "frac_no_belief_generic": float(np.mean([r["frac_no_belief_generic"] for r in rows])),
        "frac_no_belief_calibrated": float(np.mean([r["frac_no_belief_calibrated"] for r in rows])),
        "delivery_rate_generic": float(np.mean([r["delivery_rate_generic"] for r in rows])),
        "delivery_rate_calibrated": float(np.mean([r["delivery_rate_calibrated"] for r in rows])),
    }
    summary["delta_cal_off_axis"] = summary["mean_off_axis_generic"] - summary["mean_off_axis_calibrated"]
    summary["delta_cal_belief_error"] = summary["mean_belief_error_generic"] - summary["mean_belief_error_calibrated"]
    return summary, actual_path


def print_summary_table(results: dict):
    print(f"\n{'='*145}\n2x2 COMM-DEGRADATION SUMMARY\n{'='*145}")
    header = (f"{'cell':<20} {'success':>8} {'collision':>10} "
              f"{'off_axis_gen':>13} {'off_axis_cal':>13} {'Δ_off_axis':>11} "
              f"{'belief_gen':>11} {'belief_cal':>11} {'Δ_belief':>9} "
              f"{'deliv_gen':>10} {'deliv_cal':>10}")
    print(header)
    for name, s in results.items():
        print(f"{name:<20} {s['success_rate']:>8.2%} {s['collision_rate']:>10.2%} "
              f"{s['mean_off_axis_generic']:>13.4f} {s['mean_off_axis_calibrated']:>13.4f} "
              f"{s['delta_cal_off_axis']:>11.4f} "
              f"{s['mean_belief_error_generic']:>11.4f} {s['mean_belief_error_calibrated']:>11.4f} "
              f"{s['delta_cal_belief_error']:>9.4f} "
              f"{s['delivery_rate_generic']:>10.2%} {s['delivery_rate_calibrated']:>10.2%}")


def main(l0_model_path: str, l2_model_path: str, n_seeds: int, csv_prefix: str):
    from stable_baselines3 import PPO
    l0_model = PPO.load(l0_model_path)
    l2_model = PPO.load(l2_model_path)

    cells = [
        ("L0-trained@L0", l0_model, "L0"),
        ("L0-trained@L2", l0_model, "L2"),
        ("L2-trained@L2", l2_model, "L2"),
        ("L2-trained@L0", l2_model, "L0"),
    ]

    results = {}
    for name, model, comm_level in cells:
        print(f"\nRunning {name}...")
        summary, csv_path = run_cell(model, comm_level, n_seeds, f"{csv_prefix}_{name}.csv")
        results[name] = summary
        print(f"  {summary['n_episodes']} episodes written to {csv_path}")

    print_summary_table(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l0-model-path", default="formation_hold_ppo_curriculum_v5")
    parser.add_argument("--l2-model-path", default="formation_hold_ppo_curriculum_L2")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--csv-prefix", default="comm_degradation")
    args = parser.parse_args()
    main(args.l0_model_path, args.l2_model_path, args.n_seeds, args.csv_prefix)