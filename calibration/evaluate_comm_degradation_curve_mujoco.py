#!/usr/bin/env python3
"""
evaluate_comm_degradation_curve_mujoco.py -- C0.3's sim-to-sim
validation (see conversation): re-runs A1.10's exact matched-diagonal
severity sweep (L0/L1/L2/L3a) using the SAME already-trained,
stand-in-trained policies, with ONLY dynamics_backend switched from
"stand_in" to "mujoco". Per phase2_config.py's own Decision 2, no
retraining happens here -- these are the exact same .zip models A1.9
already produced.

Deliberately does NOT modify evaluate_comm_degradation_curve.py at all
-- that file stays completely frozen, per this session's own agreed
principle ("everything else stays frozen, only stand-in -> MuJoCo
changes"). Reuses its run_episode, aggregate_rows, GRID_SEPARATIONS,
GRID_VELOCITIES, and COMM_CONFIGS directly via import -- none of those
construct an environment themselves, so none of them need to change.
Only a MuJoCo-aware build_env_with_comm/run_grid_once/
run_cell_multi_seed exist here, mirroring the original three functions
of the same name exactly, with dynamics_backend="mujoco" threaded
through the one line that actually constructs a TwoAgentFormationEnv.

Usage (same CLI shape as evaluate_comm_degradation_curve.py):
    python3 evaluate_comm_degradation_curve_mujoco.py \\
        --severities L0 L1 L2 L3a \\
        --model-paths formation_hold_ppo_curriculum_v5 \\
                       formation_hold_ppo_curriculum_L1 \\
                       formation_hold_ppo_curriculum_L2 \\
                       formation_hold_ppo_curriculum_L3a \\
        --csv-prefix comm_curve_mujoco

Expect similar qualitative shape to the stand-in's own comm_curve
results, not necessarily identical numbers -- the same way MuJoCo's
held-out RMSE was never expected to exactly match the stand-in's (B2.1).
A separate script (compare_stand_in_vs_mujoco_curve.py, not yet built)
is the right place to do that comparison once both CSVs exist.

*** UNVERIFIED against a real model in this sandbox -- run_grid_once/
run_cell_multi_seed's own logic mirrors the already-real, already-used
originals exactly; model.predict() itself and MuJoCoArenaCore's
behavior under this exact evaluation harness are not verified here.
"""
import argparse

from two_agent_formation_env import TwoAgentFormationEnv
from evaluate_comm_degradation_curve import (
    GRID_SEPARATIONS, GRID_VELOCITIES, COMM_CONFIGS, VEHICLE_SPEC_PATH,
    run_episode, aggregate_rows, print_cell_stats,
)
from evaluate_recovery import safe_csv_path
from mlflow_run_utils import get_git_commit
import mlflow_logging
import csv


def build_env_with_comm_mujoco(comm_level: str, comm_seed: int,
                               speed_of_sound_mps: float = 1500.0):
    """Mirrors evaluate_comm_degradation_curve.py's own
    build_env_with_comm exactly, with dynamics_backend="mujoco" added --
    the ONLY intentional difference from the original."""
    cfg = COMM_CONFIGS[comm_level]
    return TwoAgentFormationEnv(
        packet_loss_config=cfg["packet_loss_config"],
        bandwidth_config=cfg["bandwidth_config"],
        comm_seed=comm_seed,
        dynamics_backend="mujoco",
        speed_of_sound_mps=speed_of_sound_mps,
    )


def run_grid_once_mujoco(model, comm_level: str, comm_seed: int, n_seeds_per_grid: int,
                           grid_separations: list, grid_velocities: list,
                           speed_of_sound_mps: float = 1500.0):
    """Mirrors evaluate_comm_degradation_curve.py's own run_grid_once
    exactly -- only the env-construction call differs."""
    env = build_env_with_comm_mujoco(comm_level, comm_seed,
                                      speed_of_sound_mps=speed_of_sound_mps)
    rows = []
    for d0 in grid_separations:
        for v0 in grid_velocities:
            for seed in range(n_seeds_per_grid):
                result = run_episode(env, model, d0, v0, seed)
                result.update({"d0": d0, "v0": v0, "seed": seed, "comm_seed": comm_seed})
                rows.append(result)
    return aggregate_rows(rows), rows


def run_cell_multi_seed_mujoco(model, comm_level: str, comm_seeds: list, n_seeds_per_grid: int,
                                 grid_separations: list, grid_velocities: list, csv_path_prefix: str,
                                 speed_of_sound_mps: float = 1500.0):
    """Mirrors evaluate_comm_degradation_curve.py's own
    run_cell_multi_seed exactly -- only calls run_grid_once_mujoco
    instead of run_grid_once."""
    per_seed_summaries = []
    all_rows = []
    for comm_seed in comm_seeds:
        summary, rows = run_grid_once_mujoco(model, comm_level, comm_seed, n_seeds_per_grid,
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
        import numpy as np
        values = np.array([s[key] for s in per_seed_summaries])
        across_seed_stats[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                   "min": float(np.min(values)), "max": float(np.max(values))}

    # C0.5 addition -- mirrors evaluate_comm_degradation_curve.py's own
    # run_cell_multi_seed logging exactly, dynamics_backend="mujoco" the
    # only intentional difference.
    mlflow_logging.log_fleet_run(
        script="evaluate_comm_degradation_curve_mujoco.py",
        run_name=f"{comm_level}_mujoco_{csv_path_prefix}",
        status="SUCCESS",
        status_reason=f"comm-degradation sweep completed, {len(comm_seeds)} comm_seeds",
        git_commit=get_git_commit(),
        vehicle_spec_path=VEHICLE_SPEC_PATH,
        comm_severity_level=comm_level,
        dynamics_backend="mujoco",
        coordination_success_rate=across_seed_stats["success_rate"]["mean"],
        reward_generic=across_seed_stats["reward_generic"]["mean"],
        reward_calibrated=across_seed_stats["reward_calibrated"]["mean"],
        params={"comm_seeds": str(list(comm_seeds)), "n_agents": 2,
                "comm_near_distance_m": COMM_CONFIGS[comm_level]["packet_loss_config"].near_distance_m,
                "comm_far_distance_m": COMM_CONFIGS[comm_level]["packet_loss_config"].far_distance_m,
                "comm_near_field_prob": COMM_CONFIGS[comm_level]["packet_loss_config"].near_field_prob,
                "comm_far_field_prob": COMM_CONFIGS[comm_level]["packet_loss_config"].far_field_prob,
                "comm_bandwidth_bytes_per_tick": COMM_CONFIGS[comm_level]["bandwidth_config"].bytes_per_tick},
        metrics={"collision_rate": across_seed_stats["collision_rate"]["mean"],
                 "delta_physics_off_axis": across_seed_stats["delta_physics_off_axis"]["mean"],
                 "delta_physics_belief_error": across_seed_stats["delta_physics_belief_error"]["mean"]},
        artifacts=[actual_path],
    )

    return per_seed_summaries, across_seed_stats, actual_path


def main(severities: list, model_paths: list, comm_seeds: list, n_seeds_per_grid: int,
         csv_prefix: str):
    """Deliberately NO --full-matrix option here, unlike the original --
    C0.3's scope is specifically the matched-diagonal comparison against
    the stand-in's own matched-diagonal result (each model at its own
    training severity), not the full diagnostic cross matrix."""
    from stable_baselines3 import PPO
    models = {sev: PPO.load(path) for sev, path in zip(severities, model_paths)}

    results = {}
    for sev in severities:
        name = f"{sev}-trained@{sev}"
        print(f"\nRunning {name} (MuJoCo backend) across comm_seeds {comm_seeds}...")
        _, across_seed_stats, csv_path = run_cell_multi_seed_mujoco(
            models[sev], sev, comm_seeds, n_seeds_per_grid, GRID_SEPARATIONS, GRID_VELOCITIES,
            f"{csv_prefix}_{name}")
        results[name] = across_seed_stats
        print(f"  written to {csv_path}")

    print(f"\n{'=' * 90}\nMUJOCO-BACKEND RESULTS ACROSS COMM_SEEDS (mean, std, range per metric)\n{'=' * 90}")
    for name, stats in results.items():
        print_cell_stats(name, stats, len(comm_seeds))

    print(f"\n{'='*72}")
    print("Results also logged to MLflow. To explore and compare runs:")
    print("  cd calibration && mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("  Then open http://127.0.0.1:5000 in a browser")
    print("(If MLflow is not installed: pip install mlflow)")
    print(f"{'='*72}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--severities", nargs="+", required=True,
                         help="e.g. L0 L1 L2 L3a -- must match the stand-in comparison run's own list")
    parser.add_argument("--model-paths", nargs="+", required=True,
                         help="Same already-trained (stand-in) models A1.10 used -- no retraining here")
    parser.add_argument("--comm-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                         help="Same comm_seed values as the stand-in comparison run, for a fair comparison")
    parser.add_argument("--n-seeds-per-grid", type=int, default=5)
    parser.add_argument("--csv-prefix", default="comm_curve_mujoco")
    args = parser.parse_args()
    assert len(args.severities) == len(args.model_paths), \
        "--severities and --model-paths must be the same length, in matching order"
    main(args.severities, args.model_paths, args.comm_seeds, args.n_seeds_per_grid, args.csv_prefix)
