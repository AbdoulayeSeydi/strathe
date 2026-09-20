import argparse
import csv

import numpy as np

from two_agent_formation_env import TwoAgentFormationEnv
from comm_channel import PacketLossConfig, BandwidthConfig
from evaluate_recovery import reset_to_custom_state, safe_csv_path
from mlflow_run_utils import get_git_commit, compute_per_agent_rewards
import mlflow_logging

GRID_SEPARATIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
GRID_VELOCITIES = [-0.7, -0.15, 0.15, 0.7]
POSITION_BEACON_SIZE_BYTES = 24
VEHICLE_SPEC_PATH = "vehicle_spec.yaml"  # matches build_two_agent_cores.py's own constant

COMM_CONFIGS = {
    "L0": dict(packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.0, 0.0),
               bandwidth_config=BandwidthConfig(bytes_per_tick=1e9)),
    "L1": dict(packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.1, 0.1),
               bandwidth_config=BandwidthConfig(bytes_per_tick=1e9)),
    "L2": dict(packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.3, 0.3),
               bandwidth_config=BandwidthConfig(bytes_per_tick=1e9)),
    "L3a": dict(packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.6, 0.6),
                bandwidth_config=BandwidthConfig(bytes_per_tick=1e9)),
    "L3b": dict(packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.6, 0.6),
                bandwidth_config=BandwidthConfig(bytes_per_tick=POSITION_BEACON_SIZE_BYTES - 1)),
}


def build_env_with_comm(comm_level: str, comm_seed: int,
                        speed_of_sound_mps: float = 1500.0):
    cfg = COMM_CONFIGS[comm_level]
    return TwoAgentFormationEnv(
        packet_loss_config=cfg["packet_loss_config"],
        bandwidth_config=cfg["bandwidth_config"],
        comm_seed=comm_seed,
        speed_of_sound_mps=speed_of_sound_mps,
    )


def run_episode(env, model, d0: float, v0: float, seed: int, max_steps: int = 200) -> dict:
    obs = reset_to_custom_state(env, d0, v0, seed)

    positions_generic, positions_calibrated = [], []
    belief_errors_generic, belief_errors_calibrated = [], []
    delivered_generic_flags, delivered_calibrated_flags = [], []
    terminal_reason = None
    steps_run = 0
    # C0.5 addition (see conversation): per-agent reward accumulators.
    # reward_ctrl_weight read LIVE from the real task config in use, not
    # hardcoded -- stays correct even if a caller ever customizes
    # FormationHoldConfig.
    reward_ctrl_weight = env._orchestrator.task.config.reward_ctrl_weight
    reward_generic_sum, reward_calibrated_sum = 0.0, 0.0

    for step in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps_run = step

        action_generic, action_calibrated = action[:6], action[6:]
        r_g, r_c = compute_per_agent_rewards(reward, action_generic, action_calibrated, reward_ctrl_weight)
        reward_generic_sum += r_g
        reward_calibrated_sum += r_c

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
        "mean_belief_error_calibrated": float(np.mean(belief_errors_calibrated)) if belief_errors_calibrated else float("nan"),
        "delivery_rate_generic": float(np.mean(delivered_generic_flags)),
        "delivery_rate_calibrated": float(np.mean(delivered_calibrated_flags)),
        # C0.5 addition -- mean per-STEP reward (not the raw sum), so
        # this stays comparable across episodes of different lengths.
        "reward_generic": reward_generic_sum / steps_run,
        "reward_calibrated": reward_calibrated_sum / steps_run,
    }


def aggregate_rows(rows: list) -> dict:
    n = len(rows)
    summary = {
        "n_episodes": n,
        "success_rate": sum(r["success"] for r in rows) / n,
        "collision_rate": sum(r["collision"] for r in rows) / n,
        "mean_off_axis_generic": float(np.mean([r["mean_off_axis_generic"] for r in rows])),
        "mean_off_axis_calibrated": float(np.mean([r["mean_off_axis_calibrated"] for r in rows])),
        "mean_belief_error_generic": float(np.nanmean([r["mean_belief_error_generic"] for r in rows])),
        "mean_belief_error_calibrated": float(np.nanmean([r["mean_belief_error_calibrated"] for r in rows])),
        "delivery_rate_generic": float(np.mean([r["delivery_rate_generic"] for r in rows])),
        "delivery_rate_calibrated": float(np.mean([r["delivery_rate_calibrated"] for r in rows])),
        # C0.5 addition -- mean across episodes of each episode's own
        # mean-per-step reward computed in run_episode() above.
        "reward_generic": float(np.mean([r["reward_generic"] for r in rows])),
        "reward_calibrated": float(np.mean([r["reward_calibrated"] for r in rows])),
    }
    summary["delta_physics_off_axis"] = summary["mean_off_axis_calibrated"] - summary["mean_off_axis_generic"]
    summary["delta_physics_belief_error"] = summary["mean_belief_error_calibrated"] - summary["mean_belief_error_generic"]
    return summary


def run_grid_once(model, comm_level: str, comm_seed: int, n_seeds_per_grid: int,
                   grid_separations: list, grid_velocities: list,
                   speed_of_sound_mps: float = 1500.0):
    env = build_env_with_comm(comm_level, comm_seed,
                               speed_of_sound_mps=speed_of_sound_mps)
    rows = []
    for d0 in grid_separations:
        for v0 in grid_velocities:
            for seed in range(n_seeds_per_grid):
                result = run_episode(env, model, d0, v0, seed)
                result.update({"d0": d0, "v0": v0, "seed": seed, "comm_seed": comm_seed})
                rows.append(result)
    return aggregate_rows(rows), rows


def run_cell_multi_seed(model, comm_level: str, comm_seeds: list, n_seeds_per_grid: int,
                         grid_separations: list, grid_velocities: list, csv_path_prefix: str,
                         speed_of_sound_mps: float = 1500.0):
    per_seed_summaries = []
    all_rows = []
    for comm_seed in comm_seeds:
        summary, rows = run_grid_once(model, comm_level, comm_seed, n_seeds_per_grid,
                                       grid_separations, grid_velocities,
                                       speed_of_sound_mps=speed_of_sound_mps)
        summary["comm_seed"] = comm_seed
        per_seed_summaries.append(summary)
        all_rows.extend(rows)

    actual_path = safe_csv_path(f"{csv_path_prefix}.csv")
    with open(actual_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    metric_keys = [k for k in per_seed_summaries[0].keys() if k not in ("n_episodes", "comm_seed")]
    across_seed_stats = {}
    for key in metric_keys:
        values = np.array([s[key] for s in per_seed_summaries])
        across_seed_stats[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                   "min": float(np.min(values)), "max": float(np.max(values))}

    # C0.5 addition -- one MLflow fleet run per (model, comm_level) cell,
    # matching log_calibration_run()'s own granularity (one run per
    # calibration attempt, not per internal iteration). Uses the
    # across-comm-seed MEAN for each metric, since that's this function's
    # own "the actual noise check" result, not any single comm_seed's run.
    mlflow_logging.log_fleet_run(
        script="evaluate_comm_degradation_curve.py",
        run_name=f"{comm_level}_standin_{csv_path_prefix}",
        status="SUCCESS",
        status_reason=f"comm-degradation sweep completed, {len(comm_seeds)} comm_seeds",
        git_commit=get_git_commit(),
        vehicle_spec_path=VEHICLE_SPEC_PATH,
        comm_severity_level=comm_level,
        dynamics_backend="stand_in",
        coordination_success_rate=across_seed_stats["success_rate"]["mean"],
        reward_generic=across_seed_stats["reward_generic"]["mean"],
        reward_calibrated=across_seed_stats["reward_calibrated"]["mean"],
        params={"comm_seeds": str(list(comm_seeds)), "n_agents": 2,
                "comm_near_distance_m": COMM_CONFIGS[comm_level]["packet_loss_config"].near_distance_m,
                "comm_far_distance_m": COMM_CONFIGS[comm_level]["packet_loss_config"].far_distance_m,
                "comm_near_field_prob": COMM_CONFIGS[comm_level]["packet_loss_config"].near_field_prob,
                "comm_far_field_prob": COMM_CONFIGS[comm_level]["packet_loss_config"].far_field_prob,
                "comm_bandwidth_bytes_per_tick": COMM_CONFIGS[comm_level]["bandwidth_config"].bytes_per_tick},
        # NOTE: PARAM_KEYS' own "comm_seed" entry is singular, better suited to a
        # single-seed-per-run granularity. This function's own granularity (one run
        # per cell, aggregated ACROSS multiple comm_seeds) doesn't have one scalar
        # comm_seed value to report -- "comm_seeds" (plural, the full list) is
        # logged as an extra param instead, rather than forcing a misleading single
        # value into the schema's own singular key.
        metrics={"collision_rate": across_seed_stats["collision_rate"]["mean"],
                 "delta_physics_off_axis": across_seed_stats["delta_physics_off_axis"]["mean"],
                 "delta_physics_belief_error": across_seed_stats["delta_physics_belief_error"]["mean"]},
        artifacts=[actual_path],
    )

    return per_seed_summaries, across_seed_stats, actual_path


_METRIC_GUIDE = {
    "success_rate": (
        "fraction of episodes where both agents held formation for 20+ consecutive steps"
    ),
    "collision_rate": (
        "fraction of episodes ending in collision (agents closer than safety_radius)"
    ),
    "delta_physics_off_axis": (
        "calibrated MINUS generic mean off-axis drift\n"
        "      negative = calibrated agent stays more on-axis; near-zero at low severity is expected"
    ),
    "delta_physics_belief_error": (
        "*** KEY RESULT *** calibrated MINUS generic belief error about teammate position\n"
        "      negative = calibrated agent has a better model of where its teammate actually is\n"
        "      expected pattern: near-zero/noisy at L0/L1 (comm is fine, both agents know where\n"
        "      the other is); meaningfully negative at L2/L3a (comm is degraded, calibrated\n"
        "      physics helps maintain an accurate belief about teammate position despite packet loss)\n"
        "      this is the metric the physics calibration specifically exists to improve"
    ),
    "delivery_rate_generic": (
        "fraction of comm ticks where a message from the calibrated agent reached the generic agent\n"
        "      should match the configured packet-loss probability: L0~1.0, L1~0.9, L2~0.7, L3a~0.4"
    ),
    "delivery_rate_calibrated": (
        "fraction of comm ticks where a message from the generic agent reached the calibrated agent"
    ),
}


def print_cell_stats(name: str, across_seed_stats: dict, n_comm_seeds: int):
    """G7 fix (see conversation): added inline metric explanations and key-result
    labeling. Previously this printed raw numbers with no guidance on interpretation,
    meaning a stranger had to cross-reference the README to understand what they
    were looking at. Now self-contained: explains each metric, marks the key result,
    and states the expected direction for delta_ metrics."""
    print(f"\n{'='*72}")
    print(f"  {name}  (averaged across {n_comm_seeds} comm_seeds)")
    print(f"{'='*72}")
    print(f"  Reading guide: negative delta_physics values = calibrated doing better than generic")
    print(f"  Format: mean  std  [min, max] across comm_seeds\n")
    for key in ("success_rate", "collision_rate", "delta_physics_off_axis", "delta_physics_belief_error",
                "delivery_rate_generic", "delivery_rate_calibrated"):
        s = across_seed_stats[key]
        sign = "+" if s["mean"] >= 0 else ""
        print(f"  {key:<32} {sign}{s['mean']:.4f}  ±{s['std']:.4f}  "
              f"[{s['min']:+.4f}, {s['max']:+.4f}]")
        guide = _METRIC_GUIDE.get(key, "")
        if guide:
            for line in guide.split("\n"):
                print(f"      {line}")
    print()


def main(severities: list, model_paths: list, comm_seeds: list, n_seeds_per_grid: int,
         full_matrix: bool, csv_prefix: str):
    from stable_baselines3 import PPO
    from validate_policy_compatibility import validate_all_policies, PolicyIncompatibleError

    # F3.3: pre-flight compatibility check before any compute -- matching
    # preflight_check.py's own "check before spending compute" precedent.
    # A shape mismatch surfaces here as a named error, not a raw numpy
    # exception three layers into the first episode's step().
    print("[pre-flight] Validating policy compatibility...")
    try:
        validate_all_policies(model_paths, algorithm="ppo")
    except PolicyIncompatibleError as e:
        raise SystemExit(f"Policy compatibility check failed:\n{e}") from e

    models = {sev: PPO.load(path) for sev, path in zip(severities, model_paths)}

    if full_matrix:
        cells = [(f"{train_sev}-trained@{eval_sev}", models[train_sev], eval_sev)
                 for train_sev in severities for eval_sev in severities]
    else:
        cells = [(f"{sev}-trained@{sev}", models[sev], sev) for sev in severities]

    results = {}
    for name, model, eval_sev in cells:
        print(f"\nRunning {name} across comm_seeds {comm_seeds}...")
        _, across_seed_stats, csv_path = run_cell_multi_seed(
            model, eval_sev, comm_seeds, n_seeds_per_grid, GRID_SEPARATIONS, GRID_VELOCITIES,
            f"{csv_prefix}_{name}")
        results[name] = across_seed_stats
        print(f"  written to {csv_path}")

    print(f"\n{'='*90}\nRESULTS ACROSS COMM_SEEDS (mean, std, range per metric)\n{'='*90}")
    for name, stats in results.items():
        print_cell_stats(name, stats, len(comm_seeds))

    # G8 fix (see conversation): MLflow access was never mentioned at the point
    # of running scripts, meaning a stranger had to know independently that
    # results were being logged and how to view them. Printed here, immediately
    # after results appear, so the next step is always visible.
    print(f"\n{'='*72}")
    print("Results also logged to MLflow. To explore and compare runs:")
    print("  cd calibration && mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("  Then open http://127.0.0.1:5000 in a browser")
    print("(If MLflow is not installed: pip install mlflow)")
    print(f"{'='*72}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--severities", nargs="+", required=True,
                         help="e.g. L0 L2, or L0 L1 L2 L3 once all four exist")
    parser.add_argument("--model-paths", nargs="+", required=True,
                         help="Model path per severity, same order as --severities")
    parser.add_argument("--comm-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                         help="comm_seed values to average over -- this is the noise check itself")
    parser.add_argument("--n-seeds-per-grid", type=int, default=5,
                         help="Existing per-grid-point seed count, unchanged from A1.9's convention")
    parser.add_argument("--full-matrix", action="store_true",
                         help="Evaluate every model at every severity (diagnostic). "
                              "Without this flag, only the matched diagonal runs "
                              "(each model at its own training severity -- the official deliverable).")
    parser.add_argument("--csv-prefix", default="comm_curve")
    args = parser.parse_args()
    assert len(args.severities) == len(args.model_paths), \
        "--severities and --model-paths must be the same length, in matching order"
    main(args.severities, args.model_paths, args.comm_seeds, args.n_seeds_per_grid,
         args.full_matrix, args.csv_prefix)
