#!/usr/bin/env python3
"""
run_fleet_experiment.py -- single entry point for Mera fleet coordination
experiments (G6/F1, see conversation).

Usage:
    python run_fleet_experiment.py                          # uses default_experiment.yaml
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

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_experiment.yaml")


def _write_report(config, all_results: dict, output_dir: str, elapsed: float) -> str:
    """Write the human-readable markdown summary report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "summary_report.md")
    severity_label = _effective_severity_label(config.communication)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Fleet Experiment Report\n\n")
        if config.name:
            f.write(f"**Experiment:** {config.name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Runtime:** {elapsed:.1f}s ({elapsed/60:.1f} min)\n\n")

        f.write(f"## Setup\n\n")
        f.write(f"| Field | Value |\n|-------|-------|\n")
        f.write(f"| Task | Formation hold: target separation {config.task.target_separation_m}m, "
                f"tolerance ±{config.task.tolerance_m}m, hold for {config.task.hold_steps} steps |\n")
        f.write(f"| Comm severity | `{severity_label}` |\n")
        f.write(f"| Dynamics backend | `{config.dynamics_backend}` |\n")
        f.write(f"| Episodes per cell | {config.evaluation.episodes} |\n")
        f.write(f"| Comm seeds | {config.evaluation.comm_seeds} |\n\n")

        f.write(f"**⚠️ Calibrated vs. generic agents — which is which:**\n\n")
        for i, v in enumerate(config.vehicles):
            label = v.role or f"vehicle_{i}"
            f.write(f"- **{label}** (`{v.model_path}`, {v.algorithm.upper()})")
            if i == 0:
                f.write(" — first vehicle (index 0)\n")
            else:
                f.write(" — second vehicle (index 1)\n")
        f.write("\n")

        f.write(f"## Results\n\n")
        f.write(f"**KEY METRIC: `delta_physics_belief_error`** = calibrated minus generic "
                f"position belief error. **Negative = calibrated agent knows where its "
                f"teammate is better.** Near-zero at L0/L1 is expected (comm is fine). "
                f"Meaningfully negative at L2/L3a is the signal this project exists to show.\n\n")

        for cell_name, (stats, backend) in all_results.items():
            f.write(f"### {cell_name} ({backend} backend)\n\n")
            f.write(f"| Metric | Mean | Std | Range |\n|--------|------|-----|-------|\n")
            for key in ("success_rate", "collision_rate", "delta_physics_off_axis",
                        "delta_physics_belief_error", "delivery_rate_generic",
                        "delivery_rate_calibrated"):
                if key not in stats:
                    continue
                s = stats[key]
                marker = " ← **KEY**" if key == "delta_physics_belief_error" else ""
                f.write(f"| `{key}` | {s['mean']:.4f} | {s['std']:.4f} | "
                        f"[{s['min']:.4f}, {s['max']:.4f}] |{marker}\n")
            f.write("\n")

        f.write(f"## Honest caveats\n\n")
        f.write("- `success_rate` is noisy at small episode counts — do not read a single "
                "cell's success rate as a definitive result without checking the std.\n")
        f.write("- No MuJoCo-based *training* has been done — all policies above were trained "
                "on the stand-in backend. This result is about transfer, not about what "
                "MuJoCo-trained policies would produce.\n")
        f.write("- No real hardware validation has been done for the fleet work.\n")
        f.write("- Absolute success rates are proof-of-concept grade (15-30%), not "
                "production-ready.\n\n")
        f.write(f"*For methodology details, see `PROJECT_STATUS.md` and `phase2_config.py` "
                f"`Claim C`.*\n")

    return report_path


def _write_metadata(config, output_dir: str) -> None:
    """Write experiment metadata (git commit, package versions, resolved config)."""
    os.makedirs(output_dir, exist_ok=True)
    import sys
    from importlib.metadata import version as _v

    def _safe_v(pkg):
        try:
            return _v(pkg)
        except Exception:
            return "unknown"

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
            "communication": {"severity": config.communication.severity},
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
    print("MERA FLEET EXPERIMENT")
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
    print(f"  Report:   {report_path}")
    print(f"  Metrics:  {os.path.join(config.output_dir, 'metrics.json')}")
    print(f"  Metadata: {os.path.join(config.output_dir, 'metadata.json')}")
    print(f"\nTo view in MLflow:")
    print(f"  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(f"  (then open http://127.0.0.1:5000)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to experiment YAML config (default: default_experiment.yaml). "
             f"Copy default_experiment.yaml and edit it to define your own experiment.",
    )
    args = parser.parse_args()

    try:
        run_experiment(args.config)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
