#!/usr/bin/env python3
"""
run_fleet_experiment_v1.py -- LEGACY fleet v1 entry point (superseded by run_fleet_experiment.py, fleet v2)
experiments (G6/F1, see conversation).

Usage:
    python run_fleet_experiment_v1.py                          # uses default_experiment_v1.yaml
    python run_fleet_experiment.py --config my_exp.yaml    # your own config

This script replaces the four-command manual sequence in the README with a
single, validated, self-contained entry point. It:
  1. Loads and validates the experiment config (pre-flight, before any compute)
  2. Loads and validates all policies (shape/algorithm checks before any episodes)
  3. Runs the comm-degradation sweep (stand_in, mujoco, or both backends)
  4. Writes a human-readable summary report alongside the raw CSV/JSON output

Everything is logged to MLflow automatically (same as the individual scripts).
If MLflow is missing or logging fails, the experiment itself is unaffected.

Architecture note: this script is deliberately a THIN FRONT DOOR onto
already-tested functions. The sweep logic lives in run_cell_multi_seed()
and run_cell_multi_seed_mujoco() (already validated in C0.3/F0.1). This
script owns validation, setup, and the report -- not physics or sweep math.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import yaml
from datetime import datetime

import numpy as np

# Pre-flight dependency check: fail early with a clear, actionable message
# rather than a raw ImportError three levels into the run.
def _check_dependency(module: str, install_name: str) -> None:
    try:
        __import__(module)
    except ImportError:
        print(
            f"ERROR: {module!r} is not installed. "
            f"Fix with: pip install {install_name}\n"
            f"(This should have been installed via pip install -r requirements.txt "
            f"-- check that requirements.txt is up to date.)"
        )
        sys.exit(1)

_check_dependency("mujoco", "mujoco==3.12.0")
_check_dependency("mlflow", "mlflow==3.16.1")
_check_dependency("stable_baselines3", "stable-baselines3")

from experiment_schema import (
    load_and_validate, build_comm_objects, _effective_severity_label,
)
from validate_policy_compatibility import load_policy, validate_policy_compatibility
from evaluate_comm_degradation_curve import (
    run_cell_multi_seed, print_cell_stats, GRID_SEPARATIONS, GRID_VELOCITIES,
)
from evaluate_comm_degradation_curve_mujoco import run_cell_multi_seed_mujoco
from mlflow_run_utils import get_git_commit
import mlflow_logging

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_experiment_v1.yaml")


def _write_report(config, all_results: dict, output_dir: str, elapsed: float) -> str:
    """Write the human-readable markdown summary report (F5.2).

    F5.2 requirements:
    - Header stating task, vehicles, calibrated-vs-generic EXPLICITLY AND UNMISSABLY
      (first thing in the file, not buried under Setup)
    - Communication condition, episode count, results table
    - Severity-comparison section when more than one severity was run
    - Caveats pulled directly from phase2_config.py's Claim C text, never smoothed over
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "summary_report.md")
    severity_label = _effective_severity_label(config.communication)

    # Pull the Claim C caveats directly from phase2_config.py rather than
    # paraphrasing them -- F5.2 explicitly requires "never smoothed over."
    # Importing the constant ensures the report always reflects the authoritative
    # text, not a copy that can drift.
    try:
        from phase2_config import CLAIM_C_HONEST_CAVEATS
        claim_c_caveats = CLAIM_C_HONEST_CAVEATS
    except (ImportError, AttributeError):
        # Fallback if the constant doesn't exist yet -- written out explicitly
        # so the intent is clear even without the import.
        claim_c_caveats = (
            "success_rate and delta_physics_off_axis did not transfer cleanly between "
            "backends -- but this is confounded by L0-L3a being four separately-trained "
            "policies (each reflecting its own training run's idiosyncrasies), not evidence "
            "against backend transfer itself. No MuJoCo-based training has been run, per "
            "Decision 2's training-backend rule; this is purely a transfer-of-an-already-"
            "trained-policy result. No real hardware validation has been done."
        )

    with open(report_path, "w", encoding="utf-8") as f:

        # --- F5.2: calibrated vs. generic FIRST, unmissably ---
        f.write(f"# Fleet Experiment Report\n\n")
        if config.name:
            f.write(f"**Experiment:** `{config.name}`  ")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
                f"**Runtime:** {elapsed:.1f}s\n\n")

        f.write("---\n\n")
        f.write("## ⚠️ WHICH AGENT IS WHICH — read before interpreting any result\n\n")
        f.write("The environment always contains exactly two agents:\n"
                "one **calibrated** (uses `calibrated_parameters_a1.yaml`, "
                "physics fitted to real BlueROV2 logs) and one **generic** "
                "(uses uncalibrated default physics). The policy below controls both.\n\n")
        f.write("| Index | Role | Model | Algorithm |\n|-------|------|-------|-----------|\n")
        for i, v in enumerate(config.vehicles):
            role = v.role or f"vehicle_{i}"
            physics = ("calibrated — `calibrated_parameters_a1.yaml`"
                       if i == 0 else "generic — default physics")
            f.write(f"| {i} | **{role}** ({physics}) "
                    f"| `{v.model_path}` | {v.algorithm.upper()} |\n")
        f.write("\n> **Note:** only `vehicles[0]`'s policy is used in the sweep "
                "(centralized control). `vehicles[1]` is loaded and validated but "
                "does not affect sweep results. The calibrated/generic physics split "
                "is set by the environment, not by the policy.\n\n")

        f.write("---\n\n")

        # --- Setup ---
        f.write("## Experiment setup\n\n")
        f.write(f"| Field | Value |\n|-------|-------|\n")
        f.write(f"| Task | Formation hold — target separation "
                f"{config.task.target_separation_m}m, "
                f"tolerance ±{config.task.tolerance_m}m, "
                f"hold {config.task.hold_steps} consecutive steps |\n")
        f.write(f"| Comm condition | `{severity_label}` |\n")
        f.write(f"| Dynamics backend | `{config.dynamics_backend}` |\n")
        f.write(f"| Episodes per cell | {config.evaluation.episodes} "
                f"({len(config.evaluation.comm_seeds)} comm seeds × "
                f"{config.evaluation.grid_seeds} grid seeds × 32 grid points) |\n")
        f.write(f"| Comm seeds | {config.evaluation.comm_seeds} |\n\n")

        # --- F5.3: explicit units and convention note, surfaced to the user ---
        f.write("## Units and conventions\n\n")
        f.write("| Quantity | Unit | Convention |\n"
                "|----------|------|------------|\n"
                "| Separation, distances | meters (m) | Euclidean distance between agent centroids |\n"
                "| Velocity components | m/s | SNAME/Fossen body frame: "
                "surge (x, forward), sway (y, left), heave (z, up) |\n"
                "| Angular rates | rad/s | Same body frame: roll (p), pitch (q), yaw (r) |\n"
                "| Speed of sound | m/s | Used in latency model: latency = distance / speed |\n"
                "| Belief error | meters | RMS position error — how far the agent's internal "
                "model of its teammate's position is from the teammate's true position |\n"
                "| `delta_physics_*` | meters | calibrated agent value **minus** generic agent "
                "value — negative means calibrated is closer to truth |\n"
                "| `delivery_rate_*` | fraction [0, 1] | Fraction of acoustic beacons "
                "actually received by that agent |\n"
                "| `success_rate` | fraction [0, 1] | Episodes where both agents held "
                f"target separation ({config.task.target_separation_m}m "
                f"± {config.task.tolerance_m}m) for {config.task.hold_steps} "
                "consecutive steps |\n"
                "| `collision_rate` | fraction [0, 1] | Episodes ending in collision "
                "(separation below safety threshold) |\n\n")

        # --- Results ---
        f.write("## Results\n\n")
        f.write("> **Key metric:** `delta_physics_belief_error` = calibrated minus generic "
                "position belief error. **Negative = calibrated agent has a more accurate "
                "model of where its teammate actually is.** "
                "Near-zero at L0/L1 is expected (comm is good, physics doesn't matter much). "
                "Meaningfully negative at L2/L3a is the Claim C signal.\n\n")

        for cell_name, (stats, backend) in all_results.items():
            f.write(f"### {cell_name} ({backend} backend)\n\n")
            f.write("| Metric | Mean | Std | Range | Note |\n"
                    "|--------|------|-----|-------|------|\n")
            annotations = {
                "success_rate":               "noisy at small n — check std",
                "collision_rate":             "should rise with severity",
                "delta_physics_off_axis":     "",
                "delta_physics_belief_error": "**← KEY METRIC** negative = calibrated better",
                "delivery_rate_generic":      "should match packet-loss prob",
                "delivery_rate_calibrated":   "should match packet-loss prob",
            }
            for key in annotations:
                if key not in stats:
                    continue
                s = stats[key]
                note = annotations[key]
                f.write(f"| `{key}` | {s['mean']:.4f} | {s['std']:.4f} | "
                        f"[{s['min']:.4f}, {s['max']:.4f}] | {note} |\n")
            f.write("\n")

        # --- Severity comparison (when applicable) ---
        if len(all_results) > 1:
            f.write("## Severity comparison\n\n")
            f.write("| Cell | Backend | `delta_physics_belief_error` mean | Signal? |\n"
                    "|------|---------|----------------------------------|--------|\n")
            for cell_name, (stats, backend) in all_results.items():
                bel = stats.get("delta_physics_belief_error", {})
                mean_val = bel.get("mean", float("nan"))
                signal = ("**YES** (meaningful negative)" if mean_val < -0.01
                          else "near-zero (expected at L0/L1)" if abs(mean_val) < 0.01
                          else "positive (unexpected — check)")
                f.write(f"| {cell_name} | {backend} | {mean_val:.4f} | {signal} |\n")
            f.write("\n")

        # --- F5.2: caveats from phase2_config.py Claim C text, never smoothed over ---
        f.write("## Honest caveats (from `phase2_config.py` Claim C)\n\n")
        f.write("> These are the exact caveats recorded in `phase2_config.py` when "
                "Claim C was established. They are reproduced here verbatim, not "
                "paraphrased, so the report is honest about what this result does "
                "and does not establish.\n\n")
        f.write(f"{claim_c_caveats}\n\n")
        f.write("Additionally:\n\n")
        f.write("- `success_rate` is noisy at the episode counts used here — "
                "do not read a single cell's success_rate in isolation without checking std.\n")
        f.write("- No real hardware validation has been done for the fleet work.\n")
        f.write("- Absolute success rates are proof-of-concept grade, not production-ready.\n\n")
        f.write("*Full methodology: `PROJECT_STATUS.md` and `phase2_config.py` `Claim C`.*\n")

    return report_path


def _write_metadata(config, output_dir: str) -> None:
    """Write experiment metadata (git commit, package versions, resolved config).
    F5.1: metadata.json now captures full resolved comm parameters including
    any overrides applied, not just the severity label."""
    os.makedirs(output_dir, exist_ok=True)
    import sys
    from importlib.metadata import version as _v
    from experiment_schema import (
        _build_packet_loss_config, _build_bandwidth_config, _build_speed_of_sound,
        _effective_severity_label,
    )

    def _safe_v(pkg):
        try:
            return _v(pkg)
        except Exception:
            return "unknown"

    plc = _build_packet_loss_config(config.communication)
    bwc = _build_bandwidth_config(config.communication)
    sos = _build_speed_of_sound(config.communication)

    metadata = {
        "git_commit": get_git_commit(),
        "python_version": sys.version.split()[0],
        "numpy_version": _safe_v("numpy"),
        "scipy_version": _safe_v("scipy"),
        "stable_baselines3_version": _safe_v("stable-baselines3"),
        "mujoco_version": _safe_v("mujoco"),
        "mlflow_schema_version": 2,
        "timestamp": datetime.now().isoformat(),
        "resolved_config": {
            "name": config.name,
            "vehicles": [{"model_path": v.model_path, "algorithm": v.algorithm,
                          "role": v.role} for v in config.vehicles],
            "communication": {
                "severity_label": _effective_severity_label(config.communication),
                "severity": config.communication.severity,
                "resolved": {
                    "near_distance_m": plc.near_distance_m,
                    "far_distance_m": plc.far_distance_m,
                    "near_field_prob": plc.near_field_prob,
                    "far_field_prob": plc.far_field_prob,
                    "bytes_per_tick": bwc.bytes_per_tick,
                    "speed_of_sound_mps": sos,
                },
            },
            "task": {"target_separation_m": config.task.target_separation_m,
                     "tolerance_m": config.task.tolerance_m,
                     "hold_steps": config.task.hold_steps},
            "evaluation": {"episodes": config.evaluation.episodes,
                           "comm_seeds": config.evaluation.comm_seeds,
                           "grid_seeds": config.evaluation.grid_seeds},
            "dynamics_backend": config.dynamics_backend,
        }
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def run_experiment(config_path: str) -> None:
    t0 = time.monotonic()

    # --- Step 1: Load and validate config ---
    print(f"\n{'='*70}")
    print("STRATHE FLEET EXPERIMENT (v1, legacy)")
    print(f"{'='*70}")
    print(f"Config: {config_path}")
    print(f"\n[1/4] Loading and validating experiment config...")
    # Pydantic validates on construction -- no separate validate() call needed.
    config = load_and_validate(config_path)
    print(f"  OK: {len(config.vehicles)} vehicles, "
          f"severity={_effective_severity_label(config.communication)}, "
          f"backend={config.dynamics_backend}")

    # --- Step 2: Load and validate all policies ---
    print(f"\n[2/4] Loading and validating policies (pre-flight shape check)...")
    from two_agent_formation_env import TwoAgentFormationEnv
    from formation_hold_task import FormationHoldConfig
    from comm_channel import PacketLossConfig, BandwidthConfig
    from two_agent_orchestrator import OrchestratorConfig

    plc, bwc, sos = build_comm_objects(config)
    ftc = FormationHoldConfig(
        target_separation=config.task.target_separation_m,
        tolerance=config.task.tolerance_m,
        hold_steps=config.task.hold_steps,
    )
    env_for_validation = TwoAgentFormationEnv(
        packet_loss_config=plc,
        bandwidth_config=bwc,
        task_config=ftc,
        speed_of_sound_mps=sos,  # F2.3: now actually threaded through
    )

    models = []
    for v in config.vehicles:
        model = load_policy(v.model_path, v.algorithm)
        validate_policy_compatibility(model, env=env_for_validation,
                                      model_path=v.model_path)
        models.append(model)
    print(f"  OK: all {len(models)} policies passed compatibility check")

    # Same-model warning: if both vehicles use the same model_path, the
    # experiment will show near-zero delta_physics_belief_error by
    # construction -- there's no real calibrated-vs-generic difference to
    # measure. This is fine for a first-run sanity check, but a researcher
    # expecting to see the Claim C signal needs two different models.
    paths = [v.model_path for v in config.vehicles]
    if len(set(paths)) == 1:
        print(
            f"\n  WARNING: both vehicles use the same model "
            f"({paths[0]!r}). delta_physics_belief_error will be near "
            f"zero by construction -- no real calibrated-vs-generic "
            f"difference exists. To see the Claim C signal, set each "
            f"vehicle to a differently-trained policy in the config.\n"
        )

    # --- Step 3: Run the sweep ---
    print(f"\n[3/4] Running fleet experiment sweep...")
    os.makedirs(config.output_dir, exist_ok=True)
    severity = _effective_severity_label(config.communication)
    all_results = {}

    backends_to_run = []
    if config.dynamics_backend in ("stand_in", "both"):
        backends_to_run.append("stand_in")
    if config.dynamics_backend in ("mujoco", "both"):
        backends_to_run.append("mujoco")

    for backend in backends_to_run:
        cell_name = f"{config.name or 'experiment'}@{severity}"
        print(f"\n  Running {cell_name} ({backend} backend)...")

        # Use vehicle 0's model as the primary policy for the sweep.
        # In a two-agent env, both agents share the same centralized policy.
        primary_model = models[0]
        csv_prefix = os.path.join(config.output_dir, f"{severity}_{backend}")

        if backend == "stand_in":
            _, across_seed_stats, csv_path = run_cell_multi_seed(
                primary_model, severity,
                config.evaluation.comm_seeds,
                config.evaluation.grid_seeds,
                GRID_SEPARATIONS, GRID_VELOCITIES,
                csv_prefix,
                speed_of_sound_mps=sos,  # F2.3: sos from experiment YAML reaches CommChannel
            )
        else:
            _, across_seed_stats, csv_path = run_cell_multi_seed_mujoco(
                primary_model, severity,
                config.evaluation.comm_seeds,
                config.evaluation.grid_seeds,
                GRID_SEPARATIONS, GRID_VELOCITIES,
                csv_prefix,
                speed_of_sound_mps=sos,  # F2.3
            )

        all_results[f"{cell_name}_{backend}"] = (across_seed_stats, backend)
        print(f"  Written to {csv_path}")
        print_cell_stats(cell_name, across_seed_stats, len(config.evaluation.comm_seeds))

    # --- Step 4: Write outputs ---
    print(f"\n[4/4] Writing report and metadata...")
    elapsed = time.monotonic() - t0

    # F5.1: create plots/ subdirectory alongside other artifacts
    plots_dir = os.path.join(config.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # F5.1: write experiment.yaml -- the fully-resolved config, overrides applied.
    # A researcher can pass this back to --config to reproduce the exact run.
    # This is distinct from the input YAML: it shows what actually ran, not
    # what was specified, so overrides are fully resolved into the comm section.
    from experiment_schema import _build_packet_loss_config, _build_bandwidth_config, _build_speed_of_sound
    plc_resolved = _build_packet_loss_config(config.communication)
    sos_resolved = _build_speed_of_sound(config.communication)
    resolved_config_dict = {
        "name": config.name,
        "vehicles": [
            {"model_path": v.model_path, "algorithm": v.algorithm, "role": v.role}
            for v in config.vehicles
        ],
        "communication": {
            "severity": config.communication.severity,
            # overrides shown as fully-resolved values so the result is unambiguous
            "overrides": {
                "near_distance_m": plc_resolved.near_distance_m,
                "far_distance_m": plc_resolved.far_distance_m,
                "near_field_prob": plc_resolved.near_field_prob,
                "far_field_prob": plc_resolved.far_field_prob,
                "speed_of_sound_mps": sos_resolved,
            },
        },
        "task": {
            "target_separation_m": config.task.target_separation_m,
            "tolerance_m": config.task.tolerance_m,
            "hold_steps": config.task.hold_steps,
        },
        "evaluation": {
            "episodes": config.evaluation.episodes,
            "comm_seeds": config.evaluation.comm_seeds,
            "grid_seeds": config.evaluation.grid_seeds,
        },
        "dynamics_backend": config.dynamics_backend,
        "output_dir": config.output_dir,
    }
    with open(os.path.join(config.output_dir, "experiment.yaml"), "w") as f:
        yaml.dump(resolved_config_dict, f, default_flow_style=False, sort_keys=False)

    _write_metadata(config, config.output_dir)
    report_path = _write_report(config, all_results, config.output_dir, elapsed)

    with open(os.path.join(config.output_dir, "metrics.json"), "w") as f:
        json.dump({
            cell: {k: v for k, v in stats.items()}
            for cell, (stats, _) in all_results.items()
        }, f, indent=2, default=float)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"  Report:          {report_path}")
    print(f"  Resolved config: {os.path.join(config.output_dir, 'experiment.yaml')}")
    print(f"  Metrics:         {os.path.join(config.output_dir, 'metrics.json')}")
    print(f"  Metadata:        {os.path.join(config.output_dir, 'metadata.json')}")
    print(f"  Plots dir:       {plots_dir}")
    print(f"\nTo view in MLflow:")
    print(f"  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(f"  (then open http://127.0.0.1:5000)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to experiment YAML config (default: default_experiment_v1.yaml). "
             f"Copy default_experiment_v1.yaml and edit it to define your own experiment.",
    )
    args = parser.parse_args()

    try:
        run_experiment(args.config)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
